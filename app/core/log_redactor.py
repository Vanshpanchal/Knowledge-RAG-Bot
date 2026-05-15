"""Utilities for redacting sensitive data from logs."""

from __future__ import annotations

import re
from typing import Any


class LogRedactor:
    """Redacts sensitive information from log messages."""

    # Patterns for sensitive data
    PATTERNS = {
        "api_key": r"[a-zA-Z0-9\-_]{32,}",
        "password": r"password['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
        "token": r"token['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    }

    @staticmethod
    def redact_text(text: str, max_length: int = 200) -> str:
        """Redact sensitive data from text."""
        if not isinstance(text, str):
            return str(text)[:max_length]

        # Truncate to max length
        text = text[:max_length]

        # Redact API keys (32+ char alphanumeric strings)
        text = re.sub(
            r"(['\"]?[a-zA-Z0-9\-_]{32,}['\"]?)",
            "[REDACTED_KEY]",
            text,
        )

        # Redact passwords
        text = re.sub(
            r"password['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
            'password: "[REDACTED]"',
            text,
            flags=re.IGNORECASE,
        )

        # Redact tokens
        text = re.sub(
            r"token['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
            'token: "[REDACTED]"',
            text,
            flags=re.IGNORECASE,
        )

        # Redact email addresses
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[REDACTED_EMAIL]",
            text,
        )

        return text

    @staticmethod
    def redact_dict(
        data: dict[str, Any], max_value_length: int = 200
    ) -> dict[str, Any]:
        """Redact sensitive fields in a dictionary."""
        sensitive_keys = {"password", "token", "api_key", "secret", "key", "auth"}
        redacted: dict[str, Any] = {}

        for key, value in data.items():
            if key.lower() in sensitive_keys:
                redacted[key] = "[REDACTED]"
            elif isinstance(value, str):
                redacted[key] = LogRedactor.redact_text(value, max_value_length)
            elif isinstance(value, dict):
                nested_redacted: dict[str, Any] = LogRedactor.redact_dict(
                    value, max_value_length
                )
                redacted[key] = nested_redacted
            else:
                redacted[key] = value

        return redacted
