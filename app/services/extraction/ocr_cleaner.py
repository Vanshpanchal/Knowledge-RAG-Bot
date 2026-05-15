"""OCR normalization and quality filtering helpers.

This module turns raw OCR into cleaner, more searchable text before chunking
and embeddings. It is intentionally lightweight so it can run inline during
ingestion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class OcrQualityReport:
    character_count: int
    alpha_ratio: float
    digit_ratio: float
    whitespace_ratio: float
    repeated_char_ratio: float
    line_count: int
    is_low_quality: bool


class OcrCleaner:
    """Normalize OCR text and filter low-quality noise."""

    PAGE_NOISE_PATTERNS = (
        re.compile(r"^page\s*\d+(\s*of\s*\d+)?$", re.IGNORECASE),
        re.compile(r"^\d+$"),
        re.compile(r"^(?:[|\-_=]{3,})$"),
    )

    @classmethod
    def normalize(cls, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\t\x0b\x0c]+", " ", text)
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        cleaned_lines: list[str] = []
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                cleaned_lines.append("")
                continue
            if cls._is_noise_line(line):
                continue
            line = cls._fix_spaced_tokens(line)
            line = cls._collapse_repeated_chars(line)
            cleaned_lines.append(line)

        normalized = "\n".join(cleaned_lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"[ ]{2,}", " ", normalized)
        return normalized.strip()

    @classmethod
    def filter_quality(cls, text: str, min_length: int = 20) -> str:
        if not text:
            return ""

        normalized = cls.normalize(text)
        if len(normalized) < min_length:
            return normalized

        lines = [line for line in normalized.split("\n") if line.strip()]
        filtered_lines = [line for line in lines if not cls._looks_like_garbage(line)]

        if not filtered_lines:
            return ""

        return "\n".join(filtered_lines).strip()

    @classmethod
    def report(cls, text: str) -> OcrQualityReport:
        text = text or ""
        character_count = len(text)
        if not character_count:
            return OcrQualityReport(
                character_count=0,
                alpha_ratio=0.0,
                digit_ratio=0.0,
                whitespace_ratio=0.0,
                repeated_char_ratio=0.0,
                line_count=0,
                is_low_quality=True,
            )

        alpha_count = sum(1 for char in text if char.isalpha())
        digit_count = sum(1 for char in text if char.isdigit())
        whitespace_count = sum(1 for char in text if char.isspace())
        repeated_char_count = sum(
            len(match.group(0)) for match in re.finditer(r"(.)\1{3,}", text)
        )

        alpha_ratio = alpha_count / character_count
        digit_ratio = digit_count / character_count
        whitespace_ratio = whitespace_count / character_count
        repeated_char_ratio = repeated_char_count / character_count
        line_count = len([line for line in text.splitlines() if line.strip()])

        is_low_quality = (
            character_count < min(40, line_count * 10 if line_count else 40)
            or alpha_ratio < 0.15
            or repeated_char_ratio > 0.2
        )

        return OcrQualityReport(
            character_count=character_count,
            alpha_ratio=alpha_ratio,
            digit_ratio=digit_ratio,
            whitespace_ratio=whitespace_ratio,
            repeated_char_ratio=repeated_char_ratio,
            line_count=line_count,
            is_low_quality=is_low_quality,
        )

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        lowered = line.lower().strip()
        return any(
            pattern.fullmatch(lowered) for pattern in OcrCleaner.PAGE_NOISE_PATTERNS
        )

    @staticmethod
    def _fix_spaced_tokens(line: str) -> str:
        # Turn letter-by-letter OCR like "A B C" into "ABC" when it looks intentional.
        if re.fullmatch(r"(?:[A-Za-z]\s+){2,}[A-Za-z]", line):
            return re.sub(r"\s+", "", line)
        return line

    @staticmethod
    def _collapse_repeated_chars(line: str) -> str:
        return re.sub(r"(.)\1{4,}", r"\1\1\1", line)

    @staticmethod
    def _looks_like_garbage(line: str) -> bool:
        if len(line) < 3:
            return True
        alpha = sum(1 for char in line if char.isalpha())
        digit = sum(1 for char in line if char.isdigit())
        other = len(line) - alpha - digit - sum(1 for char in line if char.isspace())
        if alpha == 0 and digit > 0 and other > 0:
            return True
        if alpha / max(1, len(line)) < 0.1 and digit / max(1, len(line)) > 0.4:
            return True
        return False


__all__ = ["OcrCleaner", "OcrQualityReport"]
