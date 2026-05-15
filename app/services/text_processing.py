"""Text cleaning and semantic chunking utilities."""
import re
from typing import List
from app.core.logging import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Cleans and normalizes extracted text."""

    @classmethod
    def clean(cls, text: str, show_logs: bool = False) -> str:
        """Remove noise, extra whitespace, and normalize Unicode."""
        if show_logs:
            logger.info(f"Cleaning text of length {len(text)} characters")
        
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Remove control characters
        text = re.sub(r"[\r\n\t]", " ", text)
        
        # Normalize Unicode (convert smart quotes to regular quotes)
        text = text.encode("ascii", "ignore").decode("ascii")
        
        cleaned = text.strip()
        if show_logs:
            logger.info(f"Cleaned text down to {len(cleaned)} characters")
        
        return cleaned


class SemanticChunker:
    """Splits text into semantically meaningful chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, show_logs: bool = False) -> List[str]:
        """Split text into overlapping chunks."""
        if show_logs:
            logger.info(f"Chunking text of length {len(text)} characters")
        
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            if show_logs and len(chunks) % 10 == 0:
                logger.info(f"Generated {len(chunks)} chunks so far...")
            
            start = end - self.overlap
            if end - self.overlap <= start:
                start = end
        
        if show_logs:
            logger.info(f"Generated {len(chunks)} total chunks")
        
        return chunks