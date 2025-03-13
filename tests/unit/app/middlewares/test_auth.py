import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import Request
from starlette.datastructures import Headers
import base64

from babeltron.app.middlewares.auth import (
    BasicAuthMiddleware,
    decode_basic_auth
)


class TestBasicAuthMiddleware:
    @pytest.fixture
    def middleware(self):
        return BasicAuthMiddleware(
            app=MagicMock(),
            username="test_user",
            password="test_password",
            exclude_paths=["/docs", "/redoc", "/openapi.json", "/healthz", "/readyz"],
        )

    @pytest.fixture
    def mock_request(self):
        credentials = base64.b64encode(b"test_user:test_password").decode()
        request = MagicMock(spec=Request)
        request.headers = Headers({"Authorization": f"Basic {credentials}"})
        request.url.path = "/api/v1/translate"
        return request

    @pytest.fixture
    def mock_public_request(self):
        request = MagicMock(spec=Request)
        request.url.path = "/docs"
        return request

    async def test_dispatch_with_valid_credentials(self, middleware, mock_request):
        mock_call_next = AsyncMock()
        mock_call_next.return_value = "response"

        result = await middleware.dispatch(mock_request, mock_call_next)

        assert result == "response"
        mock_call_next.assert_called_once_with(mock_request)

    async def test_dispatch_with_invalid_credentials(self, middleware):
        credentials = base64.b64encode(b"wrong_user:wrong_password").decode()
        request = MagicMock(spec=Request)
        request.headers = Headers({"Authorization": f"Basic {credentials}"})
        request.url.path = "/api/v1/translate"

        mock_call_next = AsyncMock()

        result = await middleware.dispatch(request, mock_call_next)

        assert result.status_code == 401
        assert "Invalid authentication credentials" in result.body.decode()
        mock_call_next.assert_not_called()

    async def test_dispatch_with_public_path(self, middleware, mock_public_request):
        mock_call_next = AsyncMock()
        mock_call_next.return_value = "response"

        result = await middleware.dispatch(mock_public_request, mock_call_next)

        assert result == "response"
        mock_call_next.assert_called_once_with(mock_public_request)

    async def test_dispatch_with_missing_auth_header(self, middleware):
        request = MagicMock(spec=Request)
        request.headers = Headers({})
        request.url.path = "/api/v1/translate"

        mock_call_next = AsyncMock()

        result = await middleware.dispatch(request, mock_call_next)

        assert result.status_code == 401
        assert "Invalid authentication credentials" in result.body.decode()
        mock_call_next.assert_not_called()


class TestDecodeBasicAuth:
    def test_decode_basic_auth_valid(self):
        credentials = base64.b64encode(b"user:password").decode()
        auth_header = f"Basic {credentials}"

        result = decode_basic_auth(auth_header)

        assert result == ("user", "password")

    def test_decode_basic_auth_invalid_scheme(self):
        auth_header = "Bearer token"

        result = decode_basic_auth(auth_header)

        assert result is None

    def test_decode_basic_auth_invalid_format(self):
        credentials = base64.b64encode(b"userpassword").decode()
        auth_header = f"Basic {credentials}"

        result = decode_basic_auth(auth_header)

        assert result is None

    def test_decode_basic_auth_invalid_base64(self):
        auth_header = "Basic not-base64"

        result = decode_basic_auth(auth_header)

        assert result is None

    def test_decode_basic_auth_empty_credentials(self):
        auth_header = "Basic "

        result = decode_basic_auth(auth_header)

        assert result is None
