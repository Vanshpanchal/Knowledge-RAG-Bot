"""Retrieval service with hybrid ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from collections import defaultdict
from typing import Any

from app.core.config import settings
from app.repositories.chunk_repository import ChunkRepository
from app.services.embeddings.providers import EmbeddingProvider
from app.services.retrieval.query_router import QueryRouteDecision, QueryRouter
from app.services.retrieval.query_rewriter import QueryRewriter
from app.services.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalResolution:
    intent: str
    strategy: str
    contexts: list[dict[str, Any]] = field(default_factory=list)
    structured_answer: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    route: QueryRouteDecision | None = None


class RetrievalService:
    def __init__(
        self,
        chunk_repository: ChunkRepository,
        embedding_provider: EmbeddingProvider,
        vector_index_name: str,
        query_rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        query_router: QueryRouter | None = None,
    ):
        self.chunk_repository = chunk_repository
        self.embedding_provider = embedding_provider
        self.vector_index_name = vector_index_name
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.query_router = query_router or QueryRouter()

    def resolve(
        self, question: str, top_k: int, filters: dict | None = None
    ) -> RetrievalResolution:
        """Resolve a query into the most appropriate retrieval strategy."""
        route = self.query_router.classify(question)
        merged_filters = dict(filters or {})

        if route.strategy == "structured_field":
            structured_contexts = self.chunk_repository.structured_field_search(
                field_candidates=route.field_candidates,
                limit=top_k,
                filters=merged_filters if merged_filters else None,
            )
            structured_answer = self._extract_structured_answer(
                route, structured_contexts
            )
            if structured_answer is not None:
                return RetrievalResolution(
                    intent=route.intent,
                    strategy=route.strategy,
                    contexts=structured_contexts,
                    structured_answer=structured_answer,
                    citations=self._build_structured_citations(
                        route, structured_contexts, structured_answer
                    ),
                    route=route,
                )

        fallback_strategy = (
            route.strategy if route.strategy != "structured_field" else "hybrid"
        )
        contexts = self.retrieve(question, top_k=top_k, filters=merged_filters)
        return RetrievalResolution(
            intent=route.intent,
            strategy=fallback_strategy,
            contexts=contexts,
            route=route,
        )

    def retrieve(
        self, question: str, top_k: int, filters: dict | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for a question with temporal filtering support.

        Args:
            question: User question.
            top_k: Number of top results to return.
            filters: Optional MongoDB filters.

        Returns:
            List of relevant chunks ranked by relevance.
        """
        return self._retrieve_hybrid(question, top_k, filters)

    def _retrieve_hybrid(
        self, question: str, top_k: int, filters: dict | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks using a layered hybrid pipeline."""
        # Optionally rewrite the query and extract temporal constraints
        rewritten = question
        temporal_filter = None

        if self.query_rewriter and hasattr(
            self.query_rewriter, "rewrite_with_temporal"
        ):
            try:
                rewritten, temporal_filter = self.query_rewriter.rewrite_with_temporal(
                    question
                )
                if temporal_filter:
                    logger.debug(f"Temporal filter applied: {temporal_filter}")
            except Exception as e:
                # Log error but continue with original question
                logger.warning(
                    f"Query rewriting with temporal failed: {e}, using original question"
                )
                rewritten = question
                temporal_filter = None

        # Merge temporal filter with provided filters
        merged_filters = filters or {}
        if temporal_filter:
            # Ensure temporal filter is properly formatted
            try:
                merged_filters.update(temporal_filter)
            except Exception as e:
                # If merging fails, skip temporal filter
                logger.warning(f"Failed to apply temporal filter: {e}")

        logger.debug(f"Using filters: {merged_filters}")

        metadata_results: list[dict[str, Any]] = []
        keyword_results: list[dict[str, Any]] = []
        vector_results: list[dict[str, Any]] = []

        try:
            metadata_results = self.chunk_repository.metadata_search(
                query=question,
                limit=top_k,
                filters=merged_filters if merged_filters else None,
            )
        except Exception as e:
            logger.exception(f"Metadata search failed: {e}")

        if settings.ATLAS_SEARCH_ENABLED:
            try:
                keyword_results = self.chunk_repository.atlas_search(
                    query=rewritten,
                    limit=top_k,
                    index_name=settings.ATLAS_SEARCH_INDEX_NAME,
                    filters=merged_filters if merged_filters else None,
                )
            except Exception as e:
                logger.warning(
                    "Atlas Search failed, falling back to keyword search: %s", e
                )

        if not keyword_results:
            try:
                keyword_results = self.chunk_repository.keyword_search(
                    query=question,
                    limit=top_k,
                    filters=merged_filters if merged_filters else None,
                )
            except Exception as e:
                logger.exception(f"Keyword search failed: {e}")
                keyword_results = []

        try:
            query_vector = self.embedding_provider.embed_texts([rewritten])[0]
            vector_results = self.chunk_repository.vector_search(
                query_vector=query_vector,
                index_name=self.vector_index_name,
                limit=top_k,
                filters=merged_filters if merged_filters else None,
            )
        except Exception as e:
            logger.exception(f"Vector search failed: {e}")

        merged = self._rrf_merge(
            metadata_results, keyword_results, vector_results, top_k
        )

        # Optionally rerank using embedding-based reranker
        if self.reranker:
            try:
                return self.reranker.rerank(rewritten, merged, top_k)
            except Exception:
                return merged
        return merged

    @staticmethod
    def _extract_structured_answer(
        route: QueryRouteDecision, contexts: list[dict[str, Any]]
    ) -> str | None:
        field_candidates = route.field_candidates
        if not field_candidates:
            return None

        for context in contexts:
            metadata = context.get("metadata") or {}
            structured_fields = metadata.get("structured_fields") or {}
            for field_name in field_candidates:
                value = structured_fields.get(field_name)
                if value in (None, "", [], {}):
                    value = metadata.get(field_name)
                if value in (None, "", [], {}):
                    continue
                return RetrievalService._format_structured_value(value)
        return None

    @staticmethod
    def _build_structured_citations(
        route: QueryRouteDecision,
        contexts: list[dict[str, Any]],
        answer: str,
    ) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for context in contexts:
            metadata = context.get("metadata") or {}
            structured_fields = metadata.get("structured_fields") or {}
            field_name = next(
                (
                    name
                    for name in route.field_candidates
                    if name in structured_fields or name in metadata
                ),
                None,
            )
            if not field_name:
                field_name = (
                    route.field_candidates[0] if route.field_candidates else "field"
                )
            citations.append(
                {
                    "chunk_id": context.get("chunk_id", ""),
                    "document_id": context.get("document_id", ""),
                    "score": float(context.get("score", 0.0)),
                    "source": metadata.get("source"),
                    "page": context.get("page"),
                    "text": f"{field_name}: {answer}",
                }
            )
        return citations

    @staticmethod
    def _format_structured_value(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        if isinstance(value, dict):
            return ", ".join(f"{key}: {val}" for key, val in value.items())
        return str(value)

    @staticmethod
    def _rrf_merge(
        metadata_results: list[dict],
        keyword_results: list[dict],
        vector_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        rank_scores: dict[str, float] = defaultdict(float)
        merged_map: dict[str, dict] = {}

        for rank, item in enumerate(metadata_results, start=1):
            key = item["chunk_id"]
            rank_scores[key] += 1.0 / (80 + rank)
            merged_map[key] = item

        for rank, item in enumerate(keyword_results, start=1):
            key = item["chunk_id"]
            rank_scores[key] += 1.0 / (70 + rank)
            if key not in merged_map:
                merged_map[key] = item

        for rank, item in enumerate(vector_results, start=1):
            key = item["chunk_id"]
            rank_scores[key] += 1.0 / (60 + rank)
            merged_map[key] = item

        ranked = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        output: list[dict] = []
        for chunk_id, score in ranked:
            record = merged_map[chunk_id]
            record["score"] = score
            output.append(record)
        return output
