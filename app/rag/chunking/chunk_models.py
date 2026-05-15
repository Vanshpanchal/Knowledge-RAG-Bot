from pydantic import BaseModel
from typing import Optional


class ChunkMetadata(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_index: int


class Chunk(BaseModel):
    chunk_id: str
    content: str
    token_count: int
    metadata: ChunkMetadata
