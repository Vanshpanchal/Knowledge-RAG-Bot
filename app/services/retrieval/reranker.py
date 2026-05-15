"""Reranker service using embedding similarity (Gemini embeddings recommended)."""

from __future__ import annotations

from typing import Iterable, List
import math

from app.services.embeddings.providers import EmbeddingProvider


class Reranker:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def rerank(self, query: str, candidates: Iterable[dict], top_k: int) -> List[dict]:
        candidates = list(candidates)
        if not candidates:
            return []

        texts = [c.get("text", "") for c in candidates]
        # embed query + candidate texts in a single batch
        vectors = self.embedding_provider.embed_texts([query] + texts)
        query_vec = vectors[0]
        text_vecs = vectors[1:]

        scored = []
        for cand, vec in zip(candidates, text_vecs):
            score = self._cosine(query_vec, vec)
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = score
            scored.append(cand_copy)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        for idx, item in enumerate(scored[:top_k], start=1):
            item["rank"] = idx
        return scored[:top_k]


__all__ = ["Reranker"]
