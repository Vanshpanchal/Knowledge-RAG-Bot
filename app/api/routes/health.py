"""Health and readiness check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.health import HealthStatus, get_health_status

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/ping", response_model=dict)
async def ping() -> dict:
    """Quick ping to check if API is alive."""
    return {"status": "alive", "message": "pong"}


@router.get("/ready", response_model=HealthStatus)
async def ready() -> HealthStatus:
    """Full readiness check (all dependencies)."""
    return await get_health_status()


@router.get("/status", response_model=HealthStatus)
async def status() -> HealthStatus:
    """Get current system status."""
    return await get_health_status()
