"""Chunk repository with vector and keyword search access."""

from __future__ import annotations

import re
from typing import Any

from pymongo.collection import Collection
from pymongo import ReplaceOne

from app.models.chunk import ChunkRecord


class ChunkRepository:
    def __init__(self, collection: Collection):
        self.collection = collection

    @staticmethod
    def _project_fields() -> dict[str, int]:
        return {
            "_id": 0,
            "chunk_id": 1,
            "document_id": 1,
            "chunk_index": 1,
            "text": 1,
            "page": 1,
            "metadata": 1,
        }

    def upsert_many(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        operations: list[ReplaceOne] = []
        for chunk in chunks:
            operations.append(
                ReplaceOne(
                    {"chunk_id": chunk.chunk_id},
                    chunk.model_dump(),
                    upsert=True,
                )
            )
        self.collection.bulk_write(operations)

    def vector_search(
        self,
        query_vector: list[float],
        index_name: str,
        limit: int,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        vector_stage: dict[str, Any] = {
            "index": index_name,
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": max(limit * 15, 30),
            "limit": limit,
        }
        if filters:
            vector_stage["filter"] = filters

        pipeline = [
            {"$vectorSearch": vector_stage},
            {
                "$project": {
                    **self._project_fields(),
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return list(self.collection.aggregate(pipeline))

    def structured_field_search(
        self,
        field_candidates: list[str],
        limit: int,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        if not field_candidates:
            return []

        or_clauses: list[dict[str, Any]] = []
        for field_name in field_candidates:
            or_clauses.append(
                {f"metadata.structured_fields.{field_name}": {"$exists": True}}
            )
            or_clauses.append({f"metadata.{field_name}": {"$exists": True}})

        search_filter: dict[str, Any] = {"$or": or_clauses}
        if filters:
            search_filter = {"$and": [filters, search_filter]}

        cursor = self.collection.find(search_filter, self._project_fields()).limit(
            limit
        )
        results = list(cursor)
        for idx, result in enumerate(results):
            result["score"] = 1.0 / (idx + 1)
        return results

    def metadata_search(
        self,
        query: str,
        limit: int,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        tokens = [
            token for token in re.findall(r"\w+", query.lower()) if len(token) > 2
        ]
        if not tokens:
            tokens = [query.lower().strip()]

        regex_pattern = "|".join(re.escape(token) for token in tokens)

        search_filter: dict[str, Any] = {
            "$or": [
                {"metadata.title": {"$regex": regex_pattern, "$options": "i"}},
                {"metadata.tags": {"$regex": regex_pattern, "$options": "i"}},
                {"metadata.source_type": {"$regex": regex_pattern, "$options": "i"}},
                {"metadata.document_type": {"$regex": regex_pattern, "$options": "i"}},
                {"metadata.domain": {"$regex": regex_pattern, "$options": "i"}},
                {"metadata.structure": {"$regex": regex_pattern, "$options": "i"}},
                {"metadata.search_text": {"$regex": regex_pattern, "$options": "i"}},
                {
                    "metadata.structured_fields": {
                        "$regex": regex_pattern,
                        "$options": "i",
                    }
                },
            ]
        }

        if filters:
            search_filter = {"$and": [filters, search_filter]}

        cursor = self.collection.find(
            search_filter,
            {
                "_id": 0,
                "chunk_id": 1,
                "document_id": 1,
                "chunk_index": 1,
                "text": 1,
                "page": 1,
                "metadata": 1,
            },
        ).limit(limit)
        results = list(cursor)
        for idx, result in enumerate(results):
            result["score"] = 1.0 / (idx + 1)
        return results

    def atlas_search(
        self,
        query: str,
        limit: int,
        index_name: str,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Optional Atlas Search layer using $search if the index exists.

        Falls back to keyword/metadata search outside Atlas Search deployment.
        """
        should_clauses: list[dict[str, Any]] = []
        fields = [
            "text",
            "metadata.title",
            "metadata.tags",
            "metadata.source_type",
            "metadata.document_type",
            "metadata.domain",
            "metadata.structure",
            "metadata.search_text",
        ]
        for field_path in fields:
            should_clauses.append(
                {
                    "text": {
                        "query": query,
                        "path": field_path,
                        "fuzzy": {"maxEdits": 1},
                    }
                }
            )

        compound: dict[str, Any] = {"should": should_clauses, "minimumShouldMatch": 1}
        if filters:
            compound["filter"] = []
            for key, value in filters.items():
                compound["filter"].append({"equals": {"path": key, "value": value}})

        pipeline: list[dict[str, Any]] = [
            {"$search": {"index": index_name, "compound": compound}},
            {
                "$project": {
                    **self._project_fields(),
                    "score": {"$meta": "searchScore"},
                }
            },
            {"$limit": limit},
        ]

        return list(self.collection.aggregate(pipeline))

    def keyword_search(
        self,
        query: str,
        limit: int,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        tokens = [
            token for token in re.findall(r"\w+", query.lower()) if len(token) > 2
        ]
        if not tokens:
            tokens = [query.lower().strip()]

        regex_pattern = "|".join(re.escape(token) for token in tokens)
        field_paths = [
            "text",
            "metadata.search_text",
        ]

        search_filter: dict[str, Any] = {
            "$or": [
                {field_path: {"$regex": regex_pattern, "$options": "i"}}
                for field_path in field_paths
            ]
        }
        if filters:
            search_filter = {"$and": [filters, search_filter]}

        cursor = self.collection.find(
            search_filter,
            {
                "_id": 0,
                "chunk_id": 1,
                "document_id": 1,
                "chunk_index": 1,
                "text": 1,
                "page": 1,
                "metadata": 1,
            },
        ).limit(limit)
        results = list(cursor)
        for idx, result in enumerate(results):
            result["score"] = 1.0 / (idx + 1)
        return results
