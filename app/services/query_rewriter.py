"""Query rewriting service for optimized retrieval."""
from pydantic import BaseModel
from app.core.config import retrieval_config
from app.services.gemini_client import generate_text


class RewrittenQuery(BaseModel):
    original_query: str
    rewritten_query: str


class QueryRewriter:
    """
    Query rewriting service that transforms user queries into optimized
    search queries for vector and keyword retrieval.
    """

    QUERY_REWRITE_PROMPT = """
    You are a retrieval query optimizer for a personal knowledge system. 
    Your task is to rewrite user queries into concise semantic search queries 
    optimized for:
    - vector retrieval
    - keyword retrieval
    - document search
    
    Rules:
    - preserve intent
    - include technical keywords
    - remove conversational filler
    - include synonymous technical terminology when useful
    - output ONLY the rewritten query
    - do not explain
    """

    @classmethod
    async def rewrite(cls, query: str) -> RewrittenQuery:
        """Rewrite a user query for optimized retrieval."""
        prompt = f"""{cls.QUERY_REWRITE_PROMPT}

        Input: {query}
        Output:"""

        rewritten = await generate_text(
            model=retrieval_config.query_rewriting.provider,
            prompt=prompt,
            temperature=0.1
        )

        return RewrittenQuery(
            original_query=query,
            rewritten_query=rewritten.strip()
        )