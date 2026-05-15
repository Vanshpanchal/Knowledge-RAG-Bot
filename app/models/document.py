"""Document metadata models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    uploaded = "uploaded"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    storage_url: str
    storage_path: str
    mime_type: str
    sha256: str
    title: str | None = None
    source_type: str = "file"
    source_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    sensitivity: str = "normal"
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: DocumentStatus = DocumentStatus.uploaded
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict)
