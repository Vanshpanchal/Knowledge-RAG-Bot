"""Hybrid semantic chunking."""

from __future__ import annotations

import re
from uuid import uuid4

from app.models.chunk import ChunkRecord
from app.utils.tokenizer import TokenCounter


class SemanticChunkerService:
    def __init__(self, min_tokens: int = 450, max_tokens: int = 700, overlap_tokens: int = 120):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.counter = TokenCounter()

    def chunk(self, document_id: str, text: str, source: str, page: int | None = None) -> list[ChunkRecord]:
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        chunk_texts: list[str] = []
        current: list[str] = []

        for paragraph in paragraphs:
            paragraph_tokens = self.counter.count(paragraph)
            current_text = "\n\n".join(current)
            current_tokens = self.counter.count(current_text)

            if current and current_tokens + paragraph_tokens > self.max_tokens:
                chunk_texts.append(current_text)
                overlap_text = self._overlap_tail(current_text)
                current = [overlap_text, paragraph] if overlap_text else [paragraph]
            else:
                current.append(paragraph)

        if current:
            chunk_texts.append("\n\n".join(current))

        merged_chunks = self._merge_small_chunks(chunk_texts)
        records: list[ChunkRecord] = []
        for index, chunk_text in enumerate(merged_chunks):
            records.append(
                ChunkRecord(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    chunk_index=index,
                    text=chunk_text,
                    token_count=self.counter.count(chunk_text),
                    page=page,
                    metadata={"source": source},
                )
            )
        return records

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        lines = [line.strip() for line in text.split("\n")]
        blocks = []
        bucket: list[str] = []
        for line in lines:
            if not line:
                if bucket:
                    blocks.append(" ".join(bucket).strip())
                    bucket = []
                continue
            bucket.append(line)
        if bucket:
            blocks.append(" ".join(bucket).strip())

        normalized_blocks: list[str] = []
        for block in blocks:
            if len(block) > 1400:
                normalized_blocks.extend(
                    [part.strip() for part in re.split(r"(?<=[.!?])\s+", block) if part.strip()]
                )
            else:
                normalized_blocks.append(block)
        return normalized_blocks

    def _overlap_tail(self, text: str) -> str:
        words = text.split()
        if not words:
            return ""
        return " ".join(words[-self.overlap_tokens :])

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return []

        merged: list[str] = []
        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue
            if self.counter.count(chunk) < self.min_tokens:
                merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
            else:
                merged.append(chunk)
        return merged
