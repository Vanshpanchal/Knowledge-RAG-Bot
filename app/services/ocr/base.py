"""OCR abstractions and implementations."""

from __future__ import annotations

import time
from typing import Protocol

import requests
import logging
from app.services.ocr.poller import poll_operation_result_sync

logger = logging.getLogger(__name__)


class OcrProvider(Protocol):
    def extract_text(self, payload: bytes, mime_type: str) -> str: ...


class NullOcrProvider:
    def extract_text(self, payload: bytes, mime_type: str) -> str:
        return ""


class AzureDocumentIntelligenceProvider:
    def __init__(self, endpoint: str, api_key: str, timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def extract_text(self, payload: bytes, mime_type: str) -> str:
        if not self.endpoint or not self.api_key:
            return ""

        start_url = (
            f"{self.endpoint}/documentintelligence/documentModels/prebuilt-read:analyze"
            "?api-version=2024-02-29-preview"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": mime_type,
        }
        logger.info(
            "Azure Document Intelligence endpoint called: %s payload_bytes=%d mime_type=%s",
            start_url,
            len(payload),
            mime_type,
        )
        logger.info(
            "Azure DocumentIntelligence POST %s headers=[Ocp-Apim-Subscription-Key=REDACTED] payload_bytes=%d",
            start_url,
            len(payload),
        )
        response = requests.post(
            start_url, data=payload, headers=headers, timeout=self.timeout
        )
        if not response.ok:
            logger.error(
                "Azure DocumentIntelligence error %s %s",
                response.status_code,
                response.text[:2048],
            )
        response.raise_for_status()
        operation_location = response.headers.get(
            "operation-location"
        ) or response.headers.get("Operation-Location")
        if not operation_location:
            return ""

        poll_body = poll_operation_result_sync(
            operation_location,
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            timeout=self.timeout,
        )
        if poll_body.get("status", "").lower() != "succeeded":
            logger.error("Document Intelligence analysis failed: %s", poll_body)
            return ""
        logger.info("Azure Document Intelligence analysis succeeded")
        return poll_body.get("analyzeResult", {}).get("content", "")


class AzureVisionOcrProvider:
    def __init__(self, endpoint: str, api_key: str, timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def extract_text(self, payload: bytes, mime_type: str) -> str:
        if not self.endpoint or not self.api_key:
            return ""
        url = (
            f"{self.endpoint}/computervision/imageanalysis:analyze"
            "?features=read&model-version=latest&language=en&api-version=2024-02-01"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/octet-stream",
        }
        logger.info(
            "Azure Vision endpoint called: %s payload_bytes=%d mime_type=%s",
            url,
            len(payload),
            mime_type,
        )
        logger.info(
            "Azure Vision POST %s headers=[Ocp-Apim-Subscription-Key=REDACTED] payload_bytes=%d",
            url,
            len(payload),
        )
        response = requests.post(
            url, headers=headers, data=payload, timeout=self.timeout
        )
        if not response.ok:
            logger.error(
                "Azure Vision error %s %s", response.status_code, response.text[:2048]
            )
        response.raise_for_status()
        body = response.json()
        lines: list[str] = []

        # New Computer Vision Read response shape (api-version=2024-02-01):
        # readResult.blocks[].lines[].text
        read_result = body.get("readResult", {})
        for block in read_result.get("blocks", []):
            for line in block.get("lines", []):
                text = line.get("text", "").strip()
                if text:
                    lines.append(text)

        # Backward compatible fallback for legacy v3.2 OCR schema.
        if not lines:
            for region in body.get("regions", []):
                for line in region.get("lines", []):
                    words = [word.get("text", "") for word in line.get("words", [])]
                    if words:
                        lines.append(" ".join(words))
        logger.info(
            "Azure Vision OCR extraction succeeded characters=%d", len("\n".join(lines))
        )
        return "\n".join(lines)
