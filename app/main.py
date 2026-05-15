"""Knowledge RAG API entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
import requests
from starlette.responses import Response as StarletteResponse
import time
from typing import MutableMapping
import json

SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-goog-api-key",
    "x-api-key",
}
MAX_LOG_BODY = 10 * 1024  # 10 KB


def _mask_headers(headers: MutableMapping[str, str]) -> dict:
    out: dict = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in SENSITIVE_HEADERS:
            out[k] = "[REDACTED]"
        else:
            out[k] = v
    return out


from pymongo.errors import PyMongoError

from app.api.dependencies.container import initialize_container, get_container
from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.health import router as health_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.database import ensure_indexes, get_mongo_client, lifespan as mongo_lifespan

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    async with mongo_lifespan(app):
        if app.state.mongo_db is not None:
            ensure_indexes(app)
        app.state.container = initialize_container()
        yield


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=app_lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    req_logger = logging.getLogger("app.requests")

    # Read and preserve request body
    try:
        body_bytes = await request.body()
    except Exception:
        body_bytes = b""

    # Reconstruct request for downstream handlers
    async def _receive():
        return {"type": "http.request", "body": body_bytes}

    new_request = Request(request.scope, _receive)

    # Mask headers before logging
    headers = {k: v for k, v in request.headers.items()}
    masked = _mask_headers(headers)

    # Prepare small-body-safe representation
    req_body_text = None
    content_type = request.headers.get("content-type", "")
    if body_bytes and ("application/json" in content_type or "text/" in content_type):
        if len(body_bytes) <= MAX_LOG_BODY:
            try:
                req_body_text = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                try:
                    req_body_text = body_bytes.decode("utf-8", errors="replace")
                except Exception:
                    req_body_text = "[unreadable]"
        else:
            req_body_text = f"[truncated {len(body_bytes)} bytes]"

    req_logger.info(
        "--> %s %s headers=%s body=%s",
        request.method,
        request.url.path,
        masked,
        req_body_text,
    )

    try:
        response = await call_next(new_request)
    except Exception:
        req_logger.exception(
            "Request handling failed: %s %s", request.method, request.url.path
        )
        raise

    # Capture response body (may consume iterator)
    resp_body = b""
    try:
        async for chunk in response.body_iterator:
            resp_body += chunk
    except Exception:
        resp_body = b""

    # Recreate response to return to client
    new_response = StarletteResponse(
        content=resp_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

    # Prepare response body text for logging
    resp_body_text = None
    resp_ct = response.media_type or response.headers.get("content-type", "")
    if resp_body and (
        resp_ct and ("application/json" in resp_ct or "text/" in resp_ct)
    ):
        if len(resp_body) <= MAX_LOG_BODY:
            try:
                resp_body_text = json.loads(resp_body.decode("utf-8"))
            except Exception:
                try:
                    resp_body_text = resp_body.decode("utf-8", errors="replace")
                except Exception:
                    resp_body_text = "[unreadable]"
        else:
            resp_body_text = f"[truncated {len(resp_body)} bytes]"

    duration = (time.time() - start) * 1000
    req_logger.info(
        "<-- %s %s %s %.2fms headers=%s response_body=%s",
        request.method,
        request.url.path,
        new_response.status_code,
        duration,
        _mask_headers(dict(new_response.headers)),
        resp_body_text,
    )
    return new_response


metrics_enabled = False
if settings.PROMETHEUS_ENABLED:
    try:
        from app.api.middleware.metrics import metrics_middleware

        app.middleware("http")(metrics_middleware)
        metrics_enabled = True
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("Prometheus middleware disabled: %s", exc)


@app.get("/")
def root():
    return {
        "service": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health")
def health_check():
    mongo_status = "unknown"
    mongo_client = get_mongo_client(app)
    try:
        if mongo_client is not None:
            mongo_client.admin.command("ping")
            mongo_status = "connected"
        else:
            mongo_status = "unavailable"
    except (AttributeError, PyMongoError):
        mongo_status = "disconnected"
    return {"status": "healthy", "mongo": mongo_status}


@app.get("/health/appwrite")
def appwrite_health_check():
    missing: list[str] = []
    if not settings.APPWRITE_ENDPOINT:
        missing.append("APPWRITE_ENDPOINT")
    if not settings.APPWRITE_PROJECT_ID:
        missing.append("APPWRITE_PROJECT_ID")
    if not settings.APPWRITE_API_KEY:
        missing.append("APPWRITE_API_KEY")
    if missing:
        return {"status": "unconfigured", "missing": missing}

    url = f"{settings.APPWRITE_ENDPOINT.rstrip('/')}/health"
    try:
        headers = {
            "X-Appwrite-Project": settings.APPWRITE_PROJECT_ID,
            "X-Appwrite-Key": settings.APPWRITE_API_KEY,
        }
        response = requests.get(
            url,
            headers=headers,
            timeout=settings.HTTP_TIMEOUT_SECONDS,
        )
        if response.ok:
            return {"status": "ok", "http_status": response.status_code}
        return {
            "status": "error",
            "http_status": response.status_code,
            "body": response.text[:500],
        }
    except requests.RequestException as exc:
        return {"status": "error", "error": str(exc)}


@app.get("/ready")
def readiness():
    if not app.state.container:
        return {"ready": False, "reason": "database_unavailable"}
    return {"ready": True}


@app.get("/metrics", include_in_schema=False)
def metrics():
    if not metrics_enabled:
        return {"enabled": False}
    from app.api.middleware.metrics import metrics_response

    return metrics_response()


app.include_router(documents_router, prefix=settings.API_PREFIX)
app.include_router(query_router, prefix=settings.API_PREFIX)
app.include_router(
    knowledge_router, prefix=settings.API_PREFIX + "/knowledge", tags=["Knowledge"]
)
app.include_router(health_router, prefix=settings.API_PREFIX)
