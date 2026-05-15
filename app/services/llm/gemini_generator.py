"""Gemini 2.5 Pro integration for RAG pipeline."""

from __future__ import annotations

import logging

from google.genai import client as genai_client

from app.core.config import settings
from app.models.response import Citation

logger = logging.getLogger(__name__)


class GeminiGenerator:
    def __init__(self):
        self.client = genai_client.Client(api_key=settings.GEMINI_API_KEY)
        self.config = settings

    def generate_response(self, question: str, contexts: list[dict]) -> str:
        """Generate a response with Gemini using retrieval-augmented prompting."""
        from app.services.llm.prompt_builder import build_retrieval_prompt

        prompt = build_retrieval_prompt(question, contexts)
        logger.info(
            "Gemini generation endpoint called: model=gemini-3.1-flash-lite contexts=%d",
            len(contexts),
        )

        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config={
                "temperature": getattr(self.config, "LLM_TEMPERATURE", 0.2),
                "top_p": getattr(self.config, "LLM_TOP_P", 0.8),
                "top_k": getattr(self.config, "LLM_TOP_K", 32),
            },
        )
        logger.info("Gemini generation response received")

        return self._post_process_response(response.text or "")

    def answer(self, question: str, contexts: list[dict]):
        """Generate response and return structured answer."""
        answer_text = self.generate_response(question, contexts)
        from app.models.response import QueryResponse

        citations: list[Citation] = [
            Citation(
                chunk_id=str(context.get("chunk_id") or ""),
                document_id=str(context.get("document_id") or ""),
                score=float(context.get("score", 0.0)),
                source=(context.get("metadata") or {}).get("source"),
                page=context.get("page"),
                text=str(context.get("text") or ""),
            )
            for context in contexts
        ]
        return QueryResponse(answer=answer_text, citations=citations)

    def _post_process_response(self, response_text: str) -> str:
        """Enforce citation format and technical precision."""
        from app.services.llm.prompt_builder import handle_insufficient_context

        return response_text if response_text.strip() else handle_insufficient_context()
