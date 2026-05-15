"""Text embedding generation using Gemini."""
import httpx
from typing import List
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiEmbedder:
    """Generates embeddings using Gemini's text-embedding-004."""
    MODEL = "text-embedding-004"

    @classmethod
    async def embed(cls, text: str, show_logs: bool = False) -> List[float]:
        """Generate embeddings for a given text."""
        if show_logs:
            logger.info(f"Generating embedding for text of length {len(text)} characters")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cls.MODEL}:embedText?key={settings.GEMINI_API_KEY}"
        payload = {
            "text": text
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                embedding = response.json()["embedding"]
                if show_logs:
                    logger.info(f"Generated embedding of dimension {len(embedding)}")
                
                return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise


class MongoDBVectorStore:
    """Stores and retrieves embeddings in MongoDB Atlas."""
    
    def __init__(self):
        from pymongo import MongoClient
        self.client = MongoClient(settings.MONGODB_URI)
        self.db = self.client[settings.DATABASE_NAME]
        self.collection = self.db["knowledge_chunks"]
        
    async def store_chunks(self, chunks: List[str], metadata: dict, show_logs: bool = False) -> None:
        """Store chunks with their embeddings in MongoDB."""
        if show_logs:
            logger.info(f"Storing {len(chunks)} chunks in MongoDB")
        
        from tqdm import tqdm
        for i, chunk in enumerate(tqdm(chunks, disable=not show_logs)):
            embedding = await GeminiEmbedder.embed(chunk, show_logs=show_logs)
            
            document = {
                "chunk": chunk,
                "embedding": embedding,
                **metadata
            }
            
            await self.collection.insert_one(document)
            
            if show_logs and (i + 1) % 10 == 0:
                logger.info(f"Inserted {i + 1} chunks...")
        
        if show_logs:
            logger.info(f"Successfully stored all {len(chunks)} chunks")

    async def search(self, query: str, top_k: int = 5, show_logs: bool = False) -> List[dict]:
        """Perform vector search in MongoDB Atlas."""
        if show_logs:
            logger.info(f"Searching for top {top_k} results for query: {query}")
        
        query_embedding = await GeminiEmbedder.embed(query, show_logs=show_logs)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    "index": "vector_index"
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "chunk": 1,
                    "document": 1,
                    "page": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        if show_logs:
            logger.info(f"Retrieved {len(results)} results from vector search")
        
        return results