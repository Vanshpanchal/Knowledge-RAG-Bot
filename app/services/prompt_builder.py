"""Prompt construction for Gemini-based knowledge system."""
from typing import List, Dict
from app.core.logging import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Constructs prompts for Gemini with context injection."""

    SYSTEM_PROMPT = """
    You are an AI knowledge assistant operating on a private personal knowledge system.
    Your responsibilities:
    - Answer questions ONLY using the supplied retrieval context
    - Provide accurate and grounded responses
    - Avoid hallucinations
    - Cite relevant document references when possible
    - Preserve technical accuracy
    - Synthesize information across multiple retrieved chunks
    - Explicitly state when information is missing or uncertain
    
    Rules:
    - NEVER fabricate information
    - NEVER invent citations
    - NEVER assume missing context
    - If the answer is not present in the context, say: "The retrieved knowledge base 
      does not contain sufficient information to answer this."
    - Prioritize factual correctness over completeness
    - Prefer concise but technically accurate responses
    - Maintain structured formatting when appropriate
    - Preserve code snippets exactly when present
    - When possible, cite the source document name used for the answer.
    """

    @classmethod
    def build_prompt(cls, context: List[Dict], question: str, show_logs: bool = False) -> str:
        """Construct the full prompt with context injection."""
        if show_logs:
            logger.info(f"Building prompt for question: {question}")
            logger.info(f"Using {len(context)} context chunks")
        
        context_str = "\n".join(
            f"[Document: {chunk['document']}]"
            f"{' [Page: ' + str(chunk['page']) + ']' if chunk.get('page') else ''}"
            f" [Score: {chunk['score']:.2f}]\n"
            f"{chunk['chunk']}\n"
            "------------------------------------------------------------"
            for chunk in context
        )
        
        prompt = f"""{cls.SYSTEM_PROMPT}

==================== RETRIEVED CONTEXT ====================
{context_str}
============================================================

USER QUESTION: {question}
ANSWER:"""
        
        if show_logs:
            logger.info(f"Prompt size: {len(prompt)} characters")
        
        return prompt