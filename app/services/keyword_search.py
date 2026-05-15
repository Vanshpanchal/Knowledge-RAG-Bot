"""Basic BM25 keyword search implementation."""
import math
from collections import defaultdict
from typing import List, Dict
from app.core.logging import get_logger

logger = get_logger(__name__)


class BM25Search:
    """Basic BM25 keyword search implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict] = []
        self.corpus_size = 0
        self.avg_doc_len = 0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.term_freqs: List[Dict[str, int]] = []
        self.index_built = False

    async def add_documents(self, documents: List[Dict], show_logs: bool = False) -> None:
        """Add documents to the index."""
        if show_logs:
            logger.info(f"Adding {len(documents)} documents to BM25 index")
        
        for doc in documents:
            self._add_document(doc)
        
        self._finalize_index()
        self.index_built = True
        if show_logs:
            logger.info(f"BM25 index built with {self.corpus_size} documents")

    def _add_document(self, doc: Dict) -> None:
        """Add a single document to the index."""
        if "chunk" not in doc:
            return
            
        text = doc["chunk"].lower()
        terms = text.split()
        
        self.doc_len.append(len(terms))
        self.corpus_size += 1
        
        # Update document frequency
        freq = defaultdict(int)
        for term in terms:
            freq[term] += 1
        
        self.term_freqs.append(freq)
        
        # Update document frequencies
        for term in freq:
            self.doc_freqs[term] += 1

    def _finalize_index(self) -> None:
        """Finalize the index and compute IDF."""
        self.avg_doc_len = sum(self.doc_len) / self.corpus_size
        
        # Compute IDF
        for term, freq in self.doc_freqs.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = idf

    async def search(self, query: str, top_k: int = 5, show_logs: bool = False) -> List[Dict]:
        """Search the index for documents matching the query."""
        if not self.index_built:
            if show_logs:
                logger.warning("BM25 index not built, returning empty results")
            return []
        
        if show_logs:
            logger.info(f"Searching BM25 index for query: {query}")
        
        query_terms = query.lower().split()
        scores = defaultdict(float)
        
        for term in query_terms:
            if term not in self.idf:
                continue
                
            idf = self.idf[term]
            
            for i, doc_len in enumerate(self.doc_len):
                if term in self.term_freqs[i]:
                    tf = self.term_freqs[i][term]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                    scores[i] += idf * (numerator / denominator)
        
        # Get top documents
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Return formatted results
        results = []
        for idx, score in ranked:
            doc = {
                "chunk": self.documents[idx]["chunk"],
                "document": self.documents[idx].get("document"),
                "page": self.documents[idx].get("page"),
                "score": score
            }
            results.append(doc)
            
        if show_logs:
            logger.info(f"BM25 search returned {len(results)} results")
        
        return results