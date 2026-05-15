"""Prometheus metrics middleware."""

from __future__ import annotations

import time

from fastapi import Request
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

REQUEST_COUNT = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "rag_http_request_latency_seconds",
    "HTTP request latency",
    ["method", "path"],
)


async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    method = request.method
    path = request.url.path
    REQUEST_COUNT.labels(method=method, path=path, status=response.status_code).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
    return response


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
