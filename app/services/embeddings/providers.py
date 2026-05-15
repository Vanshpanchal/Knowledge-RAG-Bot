"""Embedding providers."""

from __future__ import annotations

import hashlib
import logging
import random
import time
from typing import Protocol

import requests

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def _with_retry(callable_fn, attempts: int = 3, backoff_seconds: float = 0.75):
    last_error = None
    for attempt in range(attempts):
        try:
            return callable_fn()
        except Exception as exc:  # pragma: no cover - defensive retries
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
    if last_error is None:
        raise RuntimeError("Retry failed without capturing an exception.")
    raise last_error


class MockEmbeddingProvider:
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            vectors.append([rng.uniform(-1.0, 1.0) for _ in range(self.dimensions)])
        return vectors


class OpenAIEmbeddingProvider:
    def __init__(
        self, api_key: str, model: str, dimensions: int, timeout: float = 30.0
    ):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing.")

        def _call():
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()["data"]
            return [
                item["embedding"] for item in sorted(data, key=lambda x: x["index"])
            ]

        vectors = _with_retry(_call)
        if vectors:
            self.dimensions = len(vectors[0])
        return vectors


class GeminiEmbeddingProvider:
    def __init__(
        self, api_key: str, model: str, dimensions: int, timeout: float = 30.0
    ):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing.")

        model_name = self.model
        if model_name.startswith("models/"):
            model_name = model_name[len("models/") :]

        vectors: list[list[float]] = []

        # Gemini embedContent returns a single embedding for one content payload.
        # Send one text per call to preserve 1:1 chunk-to-vector mapping.
        for text in texts:

            def _call_single():
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent"
                logger.info(
                    "Hitting Gemini embedding endpoint: %s (model=models/%s)",
                    url,
                    model_name,
                )
                response = requests.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                    },
                    json={
                        "model": f"models/{model_name}",
                        "content": {"parts": [{"text": text}]},
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()

            body = _with_retry(_call_single)

            if (
                isinstance(body, dict)
                and "embedding" in body
                and isinstance(body["embedding"], dict)
                and "values" in body["embedding"]
            ):
                vectors.append(body["embedding"]["values"])
                continue

            raise ValueError("Unexpected Gemini embed response shape: %r" % (body,))

        if vectors:
            self.dimensions = len(vectors[0])
        return vectors
