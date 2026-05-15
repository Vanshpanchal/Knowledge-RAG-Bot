"""Hybrid retrieval system combining vector and keyword search."""
from typing import List, Dict
from app.services.embeddings import MongoDBVectorStore
from app.services.keyword_search import BM25Search
from app.core.logging import get_logger
import asyncio

logger = get_logger(__name__)


class HybridRetriever:
    """Combines vector search (semantic) + BM25 (keyword)."""

    def __init__(self):
        self.vector_store = MongoDBVectorStore()
        self.bm25 = BM25Search()
        
    async def initialize(self, show_logs: bool = False) -> None:
        """Initialize the BM25 index with documents from MongoDB."""
        if show_logs:
            logger.info("Initializing HybridRetriever and building BM25 index")
        
        try:
            # Get documents from MongoDB
            documents = await self.vector_store.collection.find({}).to_list(None)
            
            # Prepare documents for BM25
            bm25_docs = []
            for doc in documents:
                bm25_docs.append({
                    "chunk": doc.get("chunk", ""),
                    "document": doc.get("document", ""),
                    "page": doc.get("page", None)
                })
            
            # Add documents to BM25 index
            await self.bm25.add_documents(bm25_docs, show_logs=show_logs)
            
            if show_logs:
                logger.info(f"HybridRetriever initialized with {len(bm25_docs)} documents")
                
        except Exception as e:
            logger.error(f"Failed to initialize HybridRetriever: {str(e)}")
            raise

    async def retrieve(self, query: str, top_k: int = 5, show_logs: bool = False) -> List[Dict]:
        """Perform hybrid retrieval."""
        if show_logs:
            logger.info(f"Performing hybrid retrieval for query: {query}")
        
        # Parallel retrieval
        vector_results = await self.vector_store.search(query, top_k, show_logs=show_logs)
        keyword_results = await self.bm25.search(query, show_logs=show_logs)
        
        # Merge and rerank results
        merged = self._rrf_merge(vector_results, keyword_results, top_k, show_logs)
        
        if show_logs:
            logger.info(f"Final merged results: {len(merged)} items")
        
        return merged

    def _rrf_merge(self, vector_results: List[Dict], keyword_results: List[Dict], top_k: int, show_logs: bool = False) -> List[Dict]:
        """Merge results using Reciprocal Rank Fusion (RRF)."""
        if show_logs:
            logger.info("Merging results using RRF")
        
        # Create score dictionaries
        vector_scores = {i: result["score"] for i, result in enumerate(vector_results)}
        keyword_scores = {i: 1.0 for i in range(len(keyword_results))}  # BM25 doesn't always provide scores
        
        # Calculate RRF scores
        fused_scores = {}
        all_results = vector_results + keyword_results
        
        for rank, result in enumerate(all_results):
            doc_id = (result.get("document"), result.get("page"))
            
            # Calculate vector score contribution
            vector_rank = rank if result in vector_results else len(vector_results) + rank
            vector_contribution = vector_scores.get(vector_rank, 0) * 0.6  # Weight for vector
            
            # Calculate keyword score contribution
            keyword_rank = rank if result in keyword_results else len(keyword_results) + rank
            keyword_contribution = keyword_scores.get(keyword_rank, 0) * 0.4  # Weight for keyword
            
            # RRF formula
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (1 / (60 + vector_rank)) * 0.6 + (1 / (60 + keyword_rank)) * 0.4
        
        # Create unique results with fused scores
        unique_results = {}
        for rank, result in enumerate(all_results):
            doc_id = (result.get("document"), result.get("page"))
            if doc_id not in unique_results:
                fused_score = fused_scores[doc_id]
                unique_results[doc_id] = {
                    "chunk": result["chunk"],
                    "document": result.get("document"),
                    "page": result.get("page"),
                    "score": fused_score
                }
        
        # Return top results sorted by score
        sorted_results = sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]