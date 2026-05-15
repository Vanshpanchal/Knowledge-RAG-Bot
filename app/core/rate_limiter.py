"""In-memory rate limiting without Redis."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from threading import Lock

from fastapi import Depends, HTTPException, status

from app.core.auth import verify_api_key

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """Simple in-memory rate limiter using time windows."""

    def __init__(self, requests_per_hour: int = 1000, requests_per_day: int = 10000):
        """Initialize rate limiter.

        Args:
            requests_per_hour: Max requests per hour per API key
            requests_per_day: Max requests per day per API key
        """
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.request_log: dict[str, list[float]] = defaultdict(
            list
        )  # {api_key: [timestamp, timestamp, ...]}
        self.lock = Lock()

    def check_rate_limit(self, api_key: str) -> None:
        """Check if request should be allowed."""
        with self.lock:
            now = time.time()
            hour_ago = now - 3600
            day_ago = now - 86400

            # Clean old entries
            if api_key in self.request_log:
                self.request_log[api_key] = [
                    t for t in self.request_log[api_key] if t > day_ago
                ]

            recent = self.request_log[api_key]
            requests_this_hour = sum(1 for t in recent if t > hour_ago)
            requests_this_day = len(recent)

            if requests_this_hour >= self.requests_per_hour:
                logger.warning(f"Rate limit exceeded (hourly) for {api_key[:8]}...")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {self.requests_per_hour} requests per hour",
                    headers={"Retry-After": "3600"},
                )

            if requests_this_day >= self.requests_per_day:
                logger.warning(f"Rate limit exceeded (daily) for {api_key[:8]}...")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {self.requests_per_day} requests per day",
                    headers={"Retry-After": "86400"},
                )

            self.request_log[api_key].append(now)
            logger.debug(
                f"Request logged for {api_key[:8]}... ({requests_this_hour}/hour, {requests_this_day}/day)"
            )


# Global rate limiter instance
_rate_limiter = InMemoryRateLimiter(requests_per_hour=1000, requests_per_day=10000)


async def check_rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    """Dependency to check rate limit on protected endpoints."""
    _rate_limiter.check_rate_limit(api_key)
    return api_key
