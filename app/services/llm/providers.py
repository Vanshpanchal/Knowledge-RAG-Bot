"""LLM providers."""

from __future__ import annotations

import json
import logging
from typing import Protocol

import requests

logger = logging.getLogger(__name__)


class LlmProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class MockLlmProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "Mock response. Configure `LLM_PROVIDER` to `gemini` or `openai` with valid keys. "
            "Question and context were received successfully."
        )


class OpenAILlmProvider:
    def __init__(self, api_key: str, model: str, timeout: float = 30.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing.")
        logger.info(
            "OpenAI endpoint called: https://api.openai.com/v1/chat/completions model=%s",
            self.model,
        )
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
            },
            timeout=self.timeout,
        )
        logger.info("OpenAI response received: status=%s", response.status_code)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class GeminiLlmProvider:
    def __init__(self, api_key: str, model: str, timeout: float = 30.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        logger.info(
            "Gemini endpoint called: https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent model=%s",
            self.model,
            self.model,
        )
        response = requests.post(url, json=payload, timeout=self.timeout)
        logger.info("Gemini response received: status=%s", response.status_code)
        response.raise_for_status()
        body = response.json()
        candidates = body.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()
