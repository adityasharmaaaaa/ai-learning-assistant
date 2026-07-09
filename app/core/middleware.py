"""Middleware that stamps every request with a request_id and logs timing."""
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id
        start = time.perf_counter()

        adapter = logging.LoggerAdapter(logger, {"request_id": request_id})
        adapter.info("--> %s %s", request.method, request.url.path)

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            adapter.exception("<-- %s %s failed after %.1fms", request.method, request.url.path, duration_ms)
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        adapter.info(
            "<-- %s %s %d (%.1fms)", request.method, request.url.path, response.status_code, duration_ms
        )
        response.headers["X-Request-ID"] = request_id
        return response
