"""Grounded response generation service."""

from __future__ import annotations

from app.models.response import Citation, QueryResponse
from app.services.llm.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from app.services.llm.providers import LlmProvider


class GenerationService:
    def __init__(self, provider: LlmProvider):
        self.provider = provider

    def answer(self, question: str, contexts: list[dict]) -> QueryResponse:
        prompt = build_user_prompt(question, contexts)
        answer = self.provider.generate(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
        citations = [
            Citation(
                chunk_id=context["chunk_id"],
                document_id=context["document_id"],
                score=float(context.get("score", 0.0)),
                source=(context.get("metadata") or {}).get("source"),
                page=context.get("page"),
                text=context.get("text", ""),
            )
            for context in contexts
        ]
        return QueryResponse(answer=answer, citations=citations)
