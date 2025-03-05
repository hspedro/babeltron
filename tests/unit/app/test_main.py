from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.backends.redis import RedisBackend

from babeltron.app.main import app, __version__, lifespan


class TestMainApp:
    """Tests for the main app module."""

    def test_app_instance(self):
        """Test that the app is a FastAPI instance."""
        assert isinstance(app, FastAPI)
        assert app.title == "Babeltron Translation API"
        assert app.version == __version__

    @patch("babeltron.app.main.include_routers")
    def test_app_initialization(self, mock_include_routers):
        """Test that the app is initialized correctly."""
        assert app.title == "Babeltron Translation API"
        assert app.version == __version__

        # We can't directly verify that include_routers was called since it happens
        # during import, but we can check that the app has routes
        assert len(app.routes) > 0

    @pytest.mark.asyncio
    @patch.object(FastAPICache, "init")
    @patch.dict("os.environ", {"CACHE_URL": ""})
    async def test_lifespan_with_no_cache_url(self, mock_init):
        mock_app = MagicMock()

        async with lifespan(mock_app):
            pass

        # No cache URL should not initialize any cache
        mock_init.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(FastAPICache, "init")
    @patch.dict("os.environ", {"CACHE_URL": "in-memory"})
    async def test_lifespan_with_inmemory_cache_url(self, mock_init):
        mock_app = MagicMock()

        async with lifespan(mock_app):
            pass

        mock_init.assert_called_once()
        args, kwargs = mock_init.call_args
        assert isinstance(args[0], InMemoryBackend)
        assert kwargs.get("prefix") == "babeltron"

    @pytest.mark.asyncio
    @patch.object(FastAPICache, "init")
    @patch("babeltron.app.main.aioredis")
    @patch.dict("os.environ", {"CACHE_URL": "redis://localhost:6379"})
    async def test_lifespan_with_redis_cache_url(self, mock_aioredis, mock_init):
        mock_redis = AsyncMock()
        mock_aioredis.from_url.return_value = mock_redis

        mock_app = MagicMock()

        async with lifespan(mock_app):
            pass

        mock_aioredis.from_url.assert_called_once_with("redis://localhost:6379")

        mock_init.assert_called_once()
        args, kwargs = mock_init.call_args
        assert isinstance(args[0], RedisBackend)
        assert kwargs.get("prefix") == "babeltron"

    @pytest.mark.asyncio
    @patch.object(FastAPICache, "init")
    @patch.dict("os.environ", {"CACHE_URL": "other://localhost:1234"})
    async def test_lifespan_with_unsupported_cache_url(self, mock_init):
        mock_app = MagicMock()

        async with lifespan(mock_app):
            pass

        mock_init.assert_not_called()

    @patch("babeltron.app.main.aioredis")
    def test_redis_import(self, mock_aioredis):
        assert mock_aioredis is not None
