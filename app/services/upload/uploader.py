"""Upload orchestration service."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.security import sha256_bytes
from app.models.document import DocumentRecord, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.services.storage.base import StorageProvider
from app.services.upload.validator import UploadValidator


class UploadService:
    def __init__(
        self,
        validator: UploadValidator,
        storage_provider: StorageProvider,
        document_repository: DocumentRepository,
    ):
        self.validator = validator
        self.storage = storage_provider
        self.document_repository = document_repository

    def upload(
        self,
        filename: str,
        content_type: str | None,
        payload: bytes,
        source_type: str = "file",
        title: str | None = None,
        tags: list[str] | None = None,
        sensitivity: str = "normal",
        source_url: str | None = None,
    ) -> DocumentRecord:
        validation = self.validator.validate(filename, content_type, payload)
        document_id = str(uuid.uuid4())

        # Compute hash from payload (works for local and cloud providers)
        file_hash = sha256_bytes(payload)
        storage_url, storage_path = self.storage.save(document_id, filename, payload)

        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            storage_url=storage_url,
            storage_path=storage_path,
            mime_type=validation.mime_type,
            sha256=file_hash,
            title=title.strip() if title and title.strip() else Path(filename).stem,
            source_type=source_type.strip().lower() or "file",
            source_url=source_url,
            tags=[tag.strip().lower() for tag in (tags or []) if tag and tag.strip()],
            sensitivity=sensitivity.strip().lower() or "normal",
            status=DocumentStatus.uploaded,
            metadata={"signature": validation.signature},
        )
        self.document_repository.create(record)
        return record

    def ingest_text_entry(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        source_type: str = "text",
        source_url: str | None = None,
        sensitivity: str = "normal",
    ) -> DocumentRecord:
        cleaned_content = content.strip()
        if not cleaned_content:
            raise ValueError("Content is required for text ingestion.")

        payload = cleaned_content.encode("utf-8")
        if len(payload) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(
                f"Text size exceeds limit of {settings.MAX_UPLOAD_SIZE_BYTES} bytes"
            )

        document_id = str(uuid.uuid4())
        normalized_title = title.strip() or "untitled"
        safe_title = re.sub(r"[^a-zA-Z0-9._-]+", "_", normalized_title).strip("._-")
        safe_title = safe_title or "entry"
        filename = f"{safe_title[:48]}_{document_id[:8]}.txt"

        file_hash = sha256_bytes(payload)
        storage_url, storage_path = self.storage.save(document_id, filename, payload)

        normalized_tags = [
            tag.strip().lower() for tag in (tags or []) if tag and tag.strip()
        ]

        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            storage_url=storage_url,
            storage_path=storage_path,
            mime_type="text/plain",
            sha256=file_hash,
            title=normalized_title,
            source_type=source_type.strip().lower() or "text",
            source_url=source_url,
            tags=normalized_tags,
            sensitivity=sensitivity.strip().lower() or "normal",
            status=DocumentStatus.uploaded,
            metadata={
                "signature": "text/plain",
                "content_type": "text/plain",
            },
        )
        self.document_repository.create(record)
        return record
