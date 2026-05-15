"""Text normalization helpers."""

from __future__ import annotations

import json
import re
from typing import Any


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)

    lines: list[str] = []
    for line in text.split("\n"):
        candidate = line.strip()
        if not candidate:
            lines.append("")
            continue
        if re.fullmatch(r"(page\s*)?\d{1,4}", candidate.lower()):
            continue
        lines.append(candidate)

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def build_metadata_search_text(metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    structured_fields = metadata.get("structured_fields") or {}

    parts: list[str] = []
    for key in (
        "title",
        "source_type",
        "document_type",
        "domain",
        "structure",
        "sensitivity",
        "source",
        "date",
        "document_date",
        "filename_date",
    ):
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            parts.append(str(value))

    tags = metadata.get("tags") or []
    if isinstance(tags, list):
        parts.extend(str(tag) for tag in tags if str(tag).strip())

    if isinstance(structured_fields, dict) and structured_fields:
        parts.append(json.dumps(structured_fields, default=str, ensure_ascii=False))

    return normalize_text(" ".join(parts))
