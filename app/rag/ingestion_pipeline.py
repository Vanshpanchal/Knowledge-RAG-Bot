"""
Ingestion pipeline: orchestrates PDF parsing, text cleaning, and semantic chunking.
"""

import logging
from typing import List, Dict, Any

from app.rag.ingestion.pdf_parser import PDFParser
from app.rag.ingestion.text_cleaner import TextCleaner
from app.rag.chunking.semantic_chunker import SemanticChunker
from app.rag.chunking.chunk_models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end pipeline: upload file → parse PDF → clean text → generate chunks."""

    def __init__(self, max_tokens: int = 512, overlap: int = 50):
        """Initialize pipeline with chunk configuration.

        Args:
            max_tokens: Target token count per chunk.
            overlap: Token overlap between consecutive chunks for context preservation.
        """
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.chunker = SemanticChunker()

    def process_pdf(self, file_path: str, source_name: str = "") -> Dict[str, Any]:
        """Process a PDF file through the full ingestion pipeline.

        Args:
            file_path: Path to the PDF file.
            source_name: Optional name/identifier for the source file.

        Returns:
            A dictionary with:
            - chunks: List[Chunk] - the generated semantic chunks
            - page_count: int - number of pages in the PDF
            - total_tokens: int - total token count across all chunks
            - source: str - the source filename
        """
        # Step 1: Parse PDF
        print(f"[Pipeline] Parsing PDF: {file_path}")
        parsed_data = PDFParser.extract_text(file_path)
        full_text = parsed_data["full_text"]
        print(
            f"[Pipeline] Extracted {parsed_data['page_count']} pages, {len(full_text)} characters"
        )

        # Step 2: Clean text
        print("[Pipeline] Cleaning text...")
        cleaned_text = TextCleaner.clean(full_text)
        print(f"[Pipeline] Cleaned text: {len(cleaned_text)} characters")

        # Check if text is empty after cleaning
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            error_msg = (
                "No extractable text found in PDF. "
                "This is likely a scanned image-based PDF. "
                "Consider using OCR (e.g., Tesseract or pytesseract) to extract text."
            )
            print(f"[Pipeline] WARNING: {error_msg}")
            return {
                "chunks": [],
                "page_count": parsed_data["page_count"],
                "total_tokens": 0,
                "source": source_name or file_path,
                "chunk_count": 0,
                "error": error_msg,
            }

        # Step 3: Generate chunks with metadata
        print(
            f"[Pipeline] Generating chunks (max_tokens={self.max_tokens}, overlap={self.overlap})..."
        )
        chunks = self.chunker.chunk_text(
            cleaned_text, max_tokens=self.max_tokens, overlap=self.overlap
        )
        print(f"[Pipeline] Generated {len(chunks)} chunks")

        # Step 4: Enrich chunks with source and page metadata
        if source_name is None:
            source_name = file_path

        for chunk in chunks:
            chunk.metadata.source = source_name
            # Optionally set page if we can infer it from the document structure
            # For now, leaving page=None; could be enhanced with better document navigation

        # Aggregate stats
        total_tokens = sum(c.token_count for c in chunks)
        print(f"[Pipeline] Total tokens across all chunks: {total_tokens}")

        return {
            "chunks": chunks,
            "page_count": parsed_data["page_count"],
            "total_tokens": total_tokens,
            "source": source_name,
            "chunk_count": len(chunks),
        }


__all__ = ["IngestionPipeline"]
