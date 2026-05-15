"""Document text extraction strategy."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.services.audio.service import AudioService
from app.services.ocr.base import OcrProvider
from app.services.extraction.ocr_cleaner import OcrCleaner


@dataclass
class ExtractionResult:
    text: str
    page_count: int
    method: str


class DocumentExtractor:
    def __init__(
        self,
        doc_ocr: OcrProvider,
        vision_ocr: OcrProvider,
        audio_service: AudioService | None = None,
    ):
        self.doc_ocr = doc_ocr
        self.vision_ocr = vision_ocr
        self.audio_service = audio_service

    def extract(
        self, payload: bytes, filename: str, mime_type: str
    ) -> ExtractionResult:
        extension = Path(filename).suffix.lower()

        if extension == ".txt":
            text = payload.decode("utf-8", errors="ignore")
            return ExtractionResult(
                text=OcrCleaner.filter_quality(text), page_count=1, method="txt"
            )

        if extension == ".docx":
            docx_module = import_module("docx")
            doc = docx_module.Document(BytesIO(payload))
            lines = [para.text for para in doc.paragraphs if para.text.strip()]
            text = "\n".join(lines)
            return ExtractionResult(
                text=OcrCleaner.filter_quality(text), page_count=1, method="docx"
            )

        if extension == ".pdf":
            return self._extract_pdf(payload, mime_type)

        if extension in {".png", ".jpg", ".jpeg"}:
            text = self.vision_ocr.extract_text(payload, mime_type)
            return ExtractionResult(
                text=self._clean_ocr_text(text), page_count=1, method="vision_ocr"
            )

        if extension in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
            if self.audio_service is None:
                return ExtractionResult(
                    text="", page_count=1, method="audio_unsupported"
                )
            transcript = self.audio_service.transcribe(payload, filename, mime_type)
            return ExtractionResult(
                text=OcrCleaner.filter_quality(transcript),
                page_count=1,
                method="audio_transcription",
            )

        return ExtractionResult(text="", page_count=0, method="unsupported")

    def _extract_pdf(self, payload: bytes, mime_type: str) -> ExtractionResult:
        reader = PdfReader(BytesIO(payload))
        pages_text: list[str] = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        merged = "\n\n".join(pages_text)
        normalized = OcrCleaner.normalize(merged)

        # OCR fallback for scanned PDFs.
        if OcrCleaner.report(normalized).is_low_quality:
            ocr_text = self._clean_ocr_text(
                self.doc_ocr.extract_text(payload, mime_type)
            )
            if ocr_text:
                return ExtractionResult(
                    text=ocr_text,
                    page_count=len(reader.pages),
                    method="doc_intelligence_ocr",
                )
            return ExtractionResult(
                text=normalized,
                page_count=len(reader.pages),
                method="pdf_parser_low_text",
            )
        return ExtractionResult(
            text=normalized, page_count=len(reader.pages), method="pdf_parser"
        )

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        normalized = OcrCleaner.normalize(text)
        return OcrCleaner.filter_quality(normalized)
