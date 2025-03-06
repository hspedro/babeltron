import base64
import binascii
from typing import Awaitable, Callable, Optional, Tuple

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, username: str, password: str, exclude_paths: list = None):
        super().__init__(app)
        self.username = username
        self.password = password
        self.exclude_paths = exclude_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/healthz",
            "/readyz",
        ]

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ):
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return self._unauthorized_response()

        credentials = decode_basic_auth(auth_header)
        if credentials is None:
            return self._unauthorized_response()

        username, password = credentials
        if username != self.username or password != self.password:
            return self._unauthorized_response()

        return await call_next(request)

    def _unauthorized_response(self):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid authentication credentials"},
            headers={"WWW-Authenticate": "Basic"},
        )


def decode_basic_auth(auth_header: str) -> Optional[Tuple[str, str]]:
    """
    Decode a Basic Authentication header value.

    Args:
        auth_header: The value of the Authorization header

    Returns:
        A tuple of (username, password) if successful, None otherwise
    """
    if not auth_header.startswith("Basic "):
        return None

    try:
        auth_decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
        username, password = auth_decoded.split(":", 1)
        return username, password
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
