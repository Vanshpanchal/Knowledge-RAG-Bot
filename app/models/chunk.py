"""Chunk models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    page: int | None = None
    embedding: list[float] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
