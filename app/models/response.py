"""API request and response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


class TextEntryRequest(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    source_type: str = Field(default="text", min_length=1)
    source_url: str | None = None
    sensitivity: str = Field(default="normal", min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    filters: dict | None = None
    tags: list[str] = Field(default_factory=list)
    sensitivity_filter: str = Field(
        default="normal",
        description="Filter by sensitivity level: 'normal', 'high', or 'classified'",
    )


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    source: str | None = None
    page: int | None = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    status: str = Field(default="success")
    strategy: str | None = None


class AudioQueryResponse(QueryResponse):
    question: str
    transcript: str
    audio_mime_type: str | None = None
    audio_base64: str | None = None
    audio_error: str | None = None
    audio_storage_url: str | None = None
    audio_storage_path: str | None = None
