"""Query rewriting service using an LLM to improve retrieval queries."""

from __future__ import annotations

from typing import Protocol, Optional, Dict, Any

from app.services.llm.providers import LlmProvider
from app.services.temporal_processor import TemporalProcessor, TemporalExpression


class QueryRewriter:
    def __init__(self, llm: LlmProvider):
        self.llm = llm
        self.temporal_processor = TemporalProcessor()

    def rewrite(self, query: str) -> str:
        """Rewrite query for optimized retrieval.

        Args:
            query: Original user query.

        Returns:
            Rewritten query optimized for retrieval.
        """
        system = (
            "You are a retrieval query optimizer for a personal knowledge system.\n"
            "Rewrite the user's query into a concise semantic search query optimized for vector and keyword retrieval.\n"
            "Output only the rewritten query with no explanation."
        )
        prompt = (
            "Rewrite this query for retrieval:\n\n" + query + "\n\nRewritten query:"
        )
        return self.llm.generate(system, prompt).strip()

    def extract_temporal_context(
        self, query: str
    ) -> tuple[str, Optional[TemporalExpression]]:
        """Extract temporal context from query.

        Args:
            query: User query that may contain temporal references.

        Returns:
            Tuple of (cleaned_query, temporal_expression).
        """
        try:
            temporal_expr = self.temporal_processor.parse_temporal_expression(query)
            cleaned_query = self.temporal_processor.remove_temporal_terms(query)
        except Exception:
            # If temporal processing fails, return original query and no temporal expression
            temporal_expr = None
            cleaned_query = query

        return cleaned_query, temporal_expr

    def rewrite_with_temporal(self, query: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """Rewrite query and extract temporal constraints.

        Args:
            query: Original user query.

        Returns:
            Tuple of (rewritten_query, temporal_filter_dict or None).
        """
        # Extract temporal context
        cleaned_query, temporal_expr = self.extract_temporal_context(query)

        # Rewrite the non-temporal part for better retrieval
        rewritten = query  # Default to original
        if cleaned_query.strip():
            try:
                rewritten = self.rewrite(cleaned_query)
            except Exception:
                # Fall back to cleaned query if rewriting fails
                rewritten = cleaned_query

        # Generate temporal filter if temporal expression found
        temporal_filter = None
        if temporal_expr:
            temporal_filter = temporal_expr.to_filter()

        return rewritten, temporal_filter


__all__ = ["QueryRewriter"]
