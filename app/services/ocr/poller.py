"""Helpers to poll Azure operation URLs (sync and async).

Provides small utilities to poll operation-location URLs until completion
and return the final JSON body. This centralizes retry/poll logic used by
both async `httpx` callers and sync `requests` callers.
"""

from __future__ import annotations

import time
import requests
import logging
import httpx
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)


def poll_operation_result_sync(
    operation_url: str,
    headers: Dict[str, str],
    timeout: float = 30.0,
    max_attempts: int = 20,
    backoff: float = 0.7,
) -> Dict[str, Any]:
    """Poll an Azure operation URL synchronously until it completes.

    Returns the JSON body of the final poll response when status == 'succeeded',
    or the last body on failure.
    """
    for attempt in range(max_attempts):
        resp = requests.get(operation_url, headers=headers, timeout=timeout)
        if not resp.ok:
            logger.warning("Poll attempt %d returned %s", attempt + 1, resp.status_code)
        resp.raise_for_status()
        body = resp.json()
        status = body.get("status", "").lower()
        if status == "succeeded":
            return body
        if status == "failed":
            logger.error("Operation failed: %s", body)
            return body
        time.sleep(backoff)
    logger.warning("Exceeded max attempts polling %s", operation_url)
    return body


async def poll_operation_result_async(
    client: httpx.AsyncClient,
    operation_url: str,
    headers: Dict[str, str],
    poll_interval: float = 2.0,
    max_attempts: int = 30,
) -> Dict[str, Any]:
    """Async poll using an httpx.AsyncClient until operation completes.

    Returns the JSON body with final status.
    """
    for attempt in range(max_attempts):
        resp = await client.get(operation_url, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        status = body.get("status", "").lower()
        if status == "succeeded":
            return body
        if status == "failed":
            logger.error("Async operation failed: %s", body)
            return body
        await asyncio.sleep(poll_interval)
    logger.warning("Async polling exceeded max attempts for %s", operation_url)
    return body
