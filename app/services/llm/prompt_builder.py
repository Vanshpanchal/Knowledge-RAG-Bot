"""Prompt construction."""

from __future__ import annotations

GEMINI_SYSTEM_PROMPT = (
    "You are a technical knowledge assistant powered by Gemini 2.5 Pro. "
    "Rules:"
    "1. Answer ONLY using the provided retrieval context. "
    "2. NEVER fabricate facts, speculate, or use prior knowledge. "
    "3. If context lacks the answer, state: 'Information not found in provided documents.' "
    "4. Preserve technical accuracy and domain-specific terminology. "
    "5. Cite sources using [1], [2], etc., based on context ordering. "
    "6. If multiple sources support a claim, cite all relevant ones. "
    "7. Use concise, professional language without filler."
)

# Backward-compatible aliases used by GenerationService.
SYSTEM_PROMPT = GEMINI_SYSTEM_PROMPT

CONTEXT_INJECTION_TEMPLATE = (
    "Document Metadata: name={source} | page={page} | chunk_score={score:.4f}\n"
    "Content:\n{text}\n"
)

INSUFFICIENT_CONTEXT_RESPONSE = (
    "Information not found in provided documents. "
    "Suggestions: "
    "- Rephrase your question with more technical details\n"
    "- Check document coverage for this topic\n"
)


def build_retrieval_prompt(question: str, contexts: list[dict]) -> str:
    """Build a structured retrieval prompt with metadata and context limits."""
    context_blocks = []
    for idx, context in enumerate(contexts[:10], start=1):  # Enforce 10-chunk limit
        metadata = context.get("metadata", {})
        source = metadata.get("source") or context.get("document_id")
        page = metadata.get("page") or context.get("page")
        score = context.get("score", 0.0)
        text = context.get("text", "")[:800]  # Enforce 800-token chunk limit

        # Inject metadata and enforce template
        context_blocks.append(
            CONTEXT_INJECTION_TEMPLATE.format(
                source=source, page=page, score=score, text=text
            )
        )

    # Combine SYSTEM + CONTEXT + USER QUESTION
    return (
        GEMINI_SYSTEM_PROMPT
        + "\n\nRetrieved context:\n"
        + "\n---\n".join(context_blocks)
        + "\n---\n\nUser question: "
        + question
        + "\n\nAnswer:"
    )


def build_user_prompt(question: str, contexts: list[dict]) -> str:
    """Backward-compatible wrapper for the generation service."""
    return build_retrieval_prompt(question, contexts)


def handle_insufficient_context() -> str:
    """Return standardized response for insufficient context."""
    return INSUFFICIENT_CONTEXT_RESPONSE
