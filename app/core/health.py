"""Health and readiness checks."""

from __future__ import annotations

import logging
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class HealthStatus(BaseModel):
    """Health check response."""

    status: str  # "healthy" or "degraded"
    version: str
    environment: str
    services: dict[str, str]  # {service_name: "ok" | "error"}


async def check_mongodb_health() -> str:
    """Check MongoDB connectivity."""
    try:
        from pymongo import MongoClient

        MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000).admin.command(
            "ping"
        )
        return "ok"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        return "error"


async def check_gemini_health() -> str:
    """Check Gemini API connectivity."""
    try:
        if not settings.GEMINI_API_KEY:
            return "skipped"

        import requests

        # Test Gemini API with a simple request
        url = f"https://generativelanguage.googleapis.com/v1/models?key={settings.GEMINI_API_KEY}"
        resp = requests.get(url, timeout=5)
        if resp.status_code >= 500:
            return "error"
        return "ok"
    except Exception as e:
        logger.error(f"Gemini health check failed: {e}")
        return "error"


async def check_azure_health() -> str:
    """Check Azure services connectivity."""
    try:
        import requests

        # Check Document Intelligence
        headers = {"Ocp-Apim-Subscription-Key": settings.AZURE_DOC_INTELLIGENCE_KEY}
        resp = requests.get(
            f"{settings.AZURE_DOC_INTELLIGENCE_ENDPOINT.rstrip('/')}/formrecognizer/v3.0/info",
            headers=headers,
            timeout=5,
        )
        if resp.status_code >= 500:
            return "error"
        # Check Computer Vision
        headers = {"Ocp-Apim-Subscription-Key": settings.AZURE_VISION_KEY}
        resp = requests.get(
            f"{settings.AZURE_VISION_ENDPOINT.rstrip('/')}/vision/v3.2/read/analyzeresults/0",
            headers=headers,
            timeout=5,
        )
        if resp.status_code >= 500:
            return "error"
        return "ok"
    except Exception as e:
        logger.error(f"Azure health check failed: {e}")
        return "error"


async def get_health_status() -> HealthStatus:
    """Get full health status."""
    services = {
        "mongodb": await check_mongodb_health(),
        "gemini": await check_gemini_health(),
        "azure": await check_azure_health(),
    }

    # Determine overall status
    status = "healthy" if all(v == "ok" for v in services.values()) else "degraded"

    return HealthStatus(
        status=status,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        services=services,
    )
