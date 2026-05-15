"""Query rewriting for RAG optimization."""

from __future__ import annotations


QUERY_REWRITING_PROMPT = (
    "You are a query rewriting assistant for a technical knowledge base. " 
    "Rewrite the user's question to:"
    "1. Preserve intent and technical meaning"
    "2. Include domain-specific keywords for retrieval"
    "3. Remove conversational filler, ambiguity, and pronouns"
    "4. Output ONLY the rewritten query without commentary."
    "Examples:"
    "Question: How do I fix the error when the TensorFlow model doesn't train?"
    "Rewritten: TensorFlow model training error troubleshooting steps"
    "Question: What's the best way to optimize SQL queries in Postgres?"
    "Rewritten: PostgreSQL query optimization techniques and best practices"
)


def rewrite_query(question: str) -> str:
    """Rewrite user query for optimized retrieval."""
    # TODO: Integrate with LLM provider or deploy as a microservice
    return question  # Placeholder