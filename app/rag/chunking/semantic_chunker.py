import re
import uuid
from typing import List, Optional
import tiktoken


from app.rag.chunking.chunk_models import Chunk, ChunkMetadata


class SemanticChunker:
    """Create semantically coherent chunks by splitting on sentence boundaries
    and enforcing a token limit (uses `tiktoken` when available for accurate
    token counts, otherwise falls back to whitespace-based estimation).

    Methods
    - `chunk_text(text, max_tokens=512, overlap=50)` -> List[Chunk]
    """

    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.tokenizer_name = tokenizer_name
        self._enc = None
        if tiktoken:
            try:
                self._enc = tiktoken.get_encoding(tokenizer_name)
            except Exception:
                try:
                    # fallback for older tiktoken versions
                    self._enc = tiktoken.encoding_for_model(tokenizer_name)
                except Exception:
                    self._enc = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._enc:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        # fallback: approximate tokens by whitespace
        return len(text.split())

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with fallback strategies for unpunctuated text."""
        if not text.strip():
            return []

        # Strategy 1: Split on sentence-ending punctuation (. ! ?)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # If we got multiple sentences, use them
        if len(sentences) > 1:
            return sentences

        # Strategy 2: If only one long sentence, try splitting on newlines
        if sentences:
            text_to_split = sentences[0]
        else:
            text_to_split = text

        lines = [line.strip() for line in text_to_split.split("\n") if line.strip()]
        if len(lines) > 1:
            return lines

        # Strategy 3: Fallback - split by word count (rough sentence proxy)
        # If text is still one long string, split into ~100-word chunks
        words = text_to_split.split()
        if len(words) > 100:
            fallback_sentences = []
            for i in range(0, len(words), 100):
                chunk = " ".join(words[i : i + 100])
                if chunk.strip():
                    fallback_sentences.append(chunk)
            return fallback_sentences if fallback_sentences else [text_to_split]

        # Return the original text as single sentence if all else fails
        return [text_to_split] if text_to_split else []

    def chunk_text(
        self, text: str, max_tokens: int = 512, overlap: int = 50
    ) -> List[Chunk]:
        """Chunk `text` into a list of `Chunk` objects.

        - `max_tokens` is the approximate token budget per chunk.
        - `overlap` is the number of tokens to carry from one chunk to the next
          to preserve context.
        """
        sentences = self._split_sentences(text)
        chunks: List[Chunk] = []

        current_sentences: List[str] = []
        current_tokens = 0
        chunk_index = 0

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)

            if current_tokens + sent_tokens <= max_tokens or not current_sentences:
                current_sentences.append(sent)
                current_tokens += sent_tokens
            else:
                # finalize current chunk
                chunk_content = " ".join(current_sentences).strip()
                chunk_id = str(uuid.uuid4())
                metadata = ChunkMetadata(
                    source=None, page=None, chunk_index=chunk_index
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        content=chunk_content,
                        token_count=current_tokens,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

                # prepare overlap: take last `overlap` tokens of the finished chunk
                overlap_text = ""
                if overlap > 0:
                    tokens = chunk_content.split()
                    if len(tokens) > overlap:
                        overlap_text = " ".join(tokens[-overlap:])
                    else:
                        overlap_text = chunk_content

                # start next chunk with overlap (if any) + current sentence
                current_sentences = [overlap_text] if overlap_text else []
                current_sentences.append(sent)
                current_sentences = [s for s in current_sentences if s]
                current_tokens = self.count_tokens(" ".join(current_sentences))

        # finalize remaining
        if current_sentences:
            chunk_content = " ".join(current_sentences).strip()
            chunk_id = str(uuid.uuid4())
            metadata = ChunkMetadata(source=None, page=None, chunk_index=chunk_index)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    token_count=current_tokens,
                    metadata=metadata,
                )
            )

        return chunks


__all__ = ["SemanticChunker"]
