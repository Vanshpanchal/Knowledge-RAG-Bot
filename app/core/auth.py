"""API authentication and authorization."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


class APIKeyAuthenticator:
    """Manages API key authentication."""

    def __init__(self, valid_keys: Optional[list[str]] = None):
        """Initialize with list of valid API keys."""
        # Default: use settings; can be extended
        self.valid_keys = valid_keys or self._load_keys()

    @staticmethod
    def _load_keys() -> list[str]:
        """Load API keys from settings."""
        keys_env = settings.API_KEYS

        if not keys_env:
            logger.warning(
                "No API_KEYS environment variable set; using default dev key"
            )
            return ["dev-key-12345"]  # Default for development
        return [k.strip() for k in keys_env.split(",") if k.strip()]

    def verify_key(self, credentials: Optional[HTTPAuthorizationCredentials]) -> str:
        """Verify API key from Bearer token."""
        if not credentials:
            logger.warning("API request received without credentials")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = credentials.credentials
        if token not in self.valid_keys:
            logger.warning(f"API request with invalid key: {token[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

        logger.info(f"API request authenticated: {token[:8]}...")
        return token


# Global authenticator instance
_authenticator = APIKeyAuthenticator()


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """Dependency to verify API key on protected endpoints."""
    return _authenticator.verify_key(credentials)
