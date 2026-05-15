"""Token counting utilities."""

from __future__ import annotations

import tiktoken


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is None:
            return len(text.split())
        try:
            return len(self._encoding.encode(text))
        except Exception:
            return len(text.split())
