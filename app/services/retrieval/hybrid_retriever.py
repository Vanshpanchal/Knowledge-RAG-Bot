"""Hybrid retrieval system combining vector, keyword, and metadata search."""
from typing import List
from pydantic import BaseModel
from app.core.config import retrieval_config
from app.services.vector_store import VectorStore
from app.services.keyword_search import KeywordSearch
from app.services.query_rewriter import QueryRewriter


class RetrievalResult(BaseModel):
    document: str
    page: int | None = None
    chunk_score: float
    content: str
    metadata: dict = {}


class HybridRetriever:
    """Combines vector, keyword, and metadata retrieval strategies."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.keyword_search = KeywordSearch()

    async def retrieve(self, query: str) -> List[RetrievalResult]:
        """Perform hybrid retrieval with reciprocal rank fusion."""
        # Rewrite query for optimized retrieval
        rewritten = await QueryRewriter.rewrite(query)
        
        # Parallel retrieval
        vector_results = await self.vector_store.search(
            query=rewritten.rewritten_query,
            top_k=retrieval_config.retrieval_top_k
        )
        
        keyword_results = await self.keyword_search.search(
            query=rewritten.rewritten_query,
            top_k=retrieval_config.retrieval_top_k
        )
        
        # Reciprocal rank fusion merge
        return self._rrf_merge(vector_results, keyword_results)

    def _rrf_merge(self, vector_results: List, keyword_results: List) -> List[RetrievalResult]:
        """Merge results using reciprocal rank fusion."""
        fused_scores = {}
        
        # Score vector results
        for rank, result in enumerate(vector_results):
            doc_id = (result.document, result.page)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + \
                (retrieval_config.vector_weight / (rank + 60))
        
        # Score keyword results
        for rank, result in enumerate(keyword_results):
            doc_id = (result.document, result.page)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + \
                (retrieval_config.keyword_weight / (rank + 60))
        
        # Combine and return top results
        all_results = vector_results + keyword_results
        unique_results = {}
        
        for result in all_results:
            doc_id = (result.document, result.page)
            if doc_id not in unique_results:
                unique_results[doc_id] = RetrievalResult(
                    document=result.document,
                    page=result.page,
                    chunk_score=fused_scores[doc_id],
                    content=result.content,
                    metadata=result.metadata
                )
        
        return sorted(
            unique_results.values(),
            key=lambda x: x.chunk_score,
            reverse=True
        )[:retrieval_config.retrieval_top_k]