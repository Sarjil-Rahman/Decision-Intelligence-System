from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQ_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Request latency in seconds",
    ["method", "path", "status"],
)

REQ_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

ERROR_COUNT = Counter(
    "http_errors_total",
    "Total HTTP errors (status >= 500)",
    ["method", "path", "status"],
)


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        dur = time.perf_counter() - start
        path = request.url.path
        REQ_LATENCY.labels(request.method, path, status).observe(dur)
        REQ_COUNT.labels(request.method, path, status).inc()
        if int(status) >= 500:
            ERROR_COUNT.labels(request.method, path, status).inc()


def metrics_endpoint() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
