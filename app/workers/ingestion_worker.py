"""Ingestion orchestration worker."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.models.document import DocumentStatus
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.chunking.semantic_chunker import SemanticChunkerService
from app.services.embeddings.providers import EmbeddingProvider
from app.services.extraction.document_intelligence import DocumentIntelligence
from app.services.extraction.document_extractor import DocumentExtractor
from app.services.extraction.date_extractor import DateExtractor
from app.services.storage.base import StorageProvider
from app.services.metadata.tagger import GeminiTagger
from app.utils.cleaners import build_metadata_search_text

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(
        self,
        storage: StorageProvider,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        extractor: DocumentExtractor,
        chunker: SemanticChunkerService,
        embedding_provider: EmbeddingProvider,
    ):
        self.storage = storage
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.extractor = extractor
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.document_intelligence = DocumentIntelligence()

    def run(self, document_id: str) -> None:
        logger.info("IngestionWorker started for document_id=%s", document_id)
        record = self.document_repository.get(document_id)
        if record is None:
            logger.error("Document not found for ingestion: %s", document_id)
            return

        try:
            self.document_repository.update_status(
                document_id, DocumentStatus.processing
            )
            logger.info("Reading payload from storage: %s", record.get("storage_path"))
            payload = self.storage.read(record["storage_path"])
            extraction = self.extractor.extract(
                payload=payload,
                filename=record["filename"],
                mime_type=record["mime_type"],
            )
            if not extraction.text.strip():
                logger.warning(
                    "Extraction returned empty text for %s (method=%s)",
                    document_id,
                    extraction.method,
                )
                raise ValueError(
                    f"Extraction produced empty text (method={extraction.method})."
                )

            chunks = self.chunker.chunk(
                document_id=document_id,
                text=extraction.text,
                source=record["filename"],
            )
            logger.info(
                "Generated %d chunks for document_id=%s", len(chunks), document_id
            )
            embeddings = self.embedding_provider.embed_texts([c.text for c in chunks])

            # Extract date metadata from document
            uploaded_at = record.get("uploaded_at")
            if isinstance(uploaded_at, str):
                try:
                    uploaded_at = datetime.fromisoformat(uploaded_at)
                except (ValueError, TypeError):
                    uploaded_at = datetime.now(timezone.utc)
            elif uploaded_at is None:
                uploaded_at = datetime.now(timezone.utc)

            date_metadata = DateExtractor.infer_metadata_dates(
                extraction.text,
                uploaded_at,
                record["filename"],
            )

            document_tags = [
                tag.strip().lower()
                for tag in (record.get("tags") or [])
                if isinstance(tag, str) and tag.strip()
            ]
            source_type = str(record.get("source_type") or "file").strip().lower()
            title = record.get("title") or record["filename"]
            sensitivity = str(record.get("sensitivity") or "normal").strip().lower()

            intelligence = self.document_intelligence.classify(
                extraction.text,
                filename=record["filename"],
                title=title,
                source_type=source_type,
                mime_type=record["mime_type"],
            )
            structured_fields = intelligence.structured_fields
            if structured_fields:
                logger.info(
                    "Structured fields extracted for %s (%s): %s",
                    document_id,
                    intelligence.document_type,
                    sorted(structured_fields.keys()),
                )

            # Initialize Gemini tagger once and generate per-chunk tags
            try:
                tagger = GeminiTagger()
                # document-level tags (fallback / global)
                generated_doc_tags = tagger.generate_tags(
                    {
                        "title": title,
                        "source_type": source_type,
                        "document_type": intelligence.document_type,
                        "domain": intelligence.domain,
                        "structure": intelligence.structure,
                        "sensitivity": sensitivity,
                        "tags": document_tags,
                        "structured_fields": structured_fields,
                        "date": date_metadata.get("date", uploaded_at),
                        "document_date": date_metadata.get("document_date"),
                        "filename_date": date_metadata.get("filename_date"),
                    },
                    max_tags=6,
                )
                for t in generated_doc_tags:
                    if t not in document_tags:
                        document_tags.append(t)
            except Exception:
                logger.exception(
                    "Tag generation failed for document; proceeding without generated tags"
                )

            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                chunk.metadata["extraction_method"] = extraction.method
                chunk.metadata["source_type"] = source_type
                chunk.metadata["title"] = title
                # Per-chunk tags: prefer Gemini-generated tags for this chunk, merged with document tags
                try:
                    # keep chunk prompt short to limit tokens
                    chunk_tags = tagger.generate_tags(
                        {
                            "title": title,
                            "source_type": source_type,
                            "document_type": intelligence.document_type,
                            "domain": intelligence.domain,
                            "structure": intelligence.structure,
                            "sensitivity": sensitivity,
                            "tags": document_tags,
                            "structured_fields": structured_fields,
                            "date": date_metadata.get("date", uploaded_at),
                            "document_date": date_metadata.get("document_date"),
                            "filename_date": date_metadata.get("filename_date"),
                            "chunk_index": (
                                getattr(chunk.metadata, "chunk_index", None)
                                if hasattr(chunk.metadata, "chunk_index")
                                else None
                            ),
                            "page": (
                                getattr(chunk.metadata, "page", None)
                                if hasattr(chunk.metadata, "page")
                                else None
                            ),
                        },
                        max_tags=6,
                    )
                except Exception:
                    logger.debug("Per-chunk tag generation failed; using document tags")
                    chunk_tags = []

                merged_tags = list(document_tags)
                for t in chunk_tags:
                    if t not in merged_tags:
                        merged_tags.append(t)
                chunk.metadata["tags"] = merged_tags
                chunk.metadata["sensitivity"] = sensitivity
                chunk.metadata["document_type"] = intelligence.document_type
                chunk.metadata["domain"] = intelligence.domain
                chunk.metadata["structure"] = intelligence.structure
                if structured_fields:
                    chunk.metadata["structured_fields"] = structured_fields
                # Add date metadata to each chunk
                chunk.metadata["date"] = date_metadata.get("date", uploaded_at)
                chunk.metadata["extracted_dates"] = [
                    d.isoformat() if isinstance(d, datetime) else d
                    for d in date_metadata.get("extracted_dates", [])
                ]
                if date_metadata.get("document_date"):
                    chunk.metadata["document_date"] = date_metadata[
                        "document_date"
                    ].isoformat()
                if date_metadata.get("filename_date"):
                    chunk.metadata["filename_date"] = date_metadata[
                        "filename_date"
                    ].isoformat()
                chunk.metadata["search_text"] = build_metadata_search_text(
                    chunk.metadata
                )

            self.chunk_repository.upsert_many(chunks)
            self.document_repository.update_status(
                document_id,
                DocumentStatus.completed,
                metadata={
                    "page_count": extraction.page_count,
                    "chunk_count": len(chunks),
                    "extraction_method": extraction.method,
                    "tags": document_tags,
                    "source_type": source_type,
                    "sensitivity": sensitivity,
                    "document_type": intelligence.document_type,
                    "domain": intelligence.domain,
                    "structure": intelligence.structure,
                    "structured_fields": structured_fields,
                },
            )
            logger.info("IngestionWorker completed for document_id=%s", document_id)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("Ingestion failed for %s: %s", document_id, exc)
            self.document_repository.update_status(
                document_id,
                DocumentStatus.failed,
                error_message=str(exc),
            )
