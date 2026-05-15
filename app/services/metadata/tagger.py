"""Document tagging service using Gemini LLM.

Produces a small list of short tags as JSON using a compact Gemini model
to minimise token usage. Falls back to simple heuristics if LLM fails.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, List, cast

from google.genai import client as genai_client

from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiTagger:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        # Use the requested preview Flash-lite model by default
        self.model = model or "gemini-3.1-flash-lite-preview"
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiTagger")
        self.client = genai_client.Client(api_key=self.api_key)

    def generate_tags(
        self, metadata: dict[str, Any], max_tags: int | None = 12
    ) -> List[str]:
        """Return a list of tags for the provided metadata.

        The LLM is allowed to decide how many tags to return. We still cap the
        number returned locally by `max_tags` to avoid exploding lists.
        """
        payload = {
            "title": metadata.get("title"),
            "source_type": metadata.get("source_type"),
            "document_type": metadata.get("document_type"),
            "domain": metadata.get("domain"),
            "structure": metadata.get("structure"),
            "sensitivity": metadata.get("sensitivity"),
            "tags": metadata.get("tags") or [],
            "structured_fields": metadata.get("structured_fields") or {},
            "date": metadata.get("date"),
            "document_date": metadata.get("document_date"),
            "filename_date": metadata.get("filename_date"),
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
        }
        prompt = (
            "Extract concise tags (single words or short phrases) from the metadata. "
            "Use the metadata fields to infer document topics, entities and structured fields.\n"
            'Return ONLY a JSON object with a single key "tags" whose value is an array of strings. '
            'Do not include any additional text. Example: {"tags":["driver_license","expiry_date"]}\n\n'
            "Metadata JSON:\n" + json.dumps(payload, default=str, ensure_ascii=False)
        )

        logger.info(
            "Generating tags with Gemini model=%s (cap=%s)", self.model, str(max_tags)
        )

        try:
            config = cast(
                Any,
                {
                    "temperature": 0.0,
                    "top_p": 0.0,
                    "top_k": 4,
                    "max_output_tokens": 200,
                },
            )
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            raw = resp.text or ""
            # Try to extract JSON object from model output
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                    tags = (
                        parsed.get("tags")
                        or parsed.get("Tags")
                        or parsed.get("tags_list")
                    )
                    if isinstance(tags, list):
                        cleaned = [
                            str(t).strip().lower() for t in tags if str(t).strip()
                        ]
                        # dedupe while preserving order
                        seen = set()
                        out = []
                        for t in cleaned:
                            if t not in seen:
                                seen.add(t)
                                out.append(t)
                        if max_tags:
                            return out[:max_tags]
                        return out
                except Exception:
                    logger.debug("Failed to parse tagger JSON, falling back")

            # Fallback: derive tags from metadata values only.
            tokens: list[str] = []
            for key, value in payload.items():
                if value in (None, "", [], {}):
                    continue
                tokens.extend(re.findall(r"\w+", str(key)))
                tokens.extend(re.findall(r"\w+", str(value)))

            counts = Counter(token.lower() for token in tokens if len(token) > 2)
            common = [t for t, _ in counts.most_common(max_tags or 12)]
            return common
        except Exception as exc:
            logger.exception("Gemini tag generation failed: %s", exc)
            return []


__all__ = ["GeminiTagger"]
