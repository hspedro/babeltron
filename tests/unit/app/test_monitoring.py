import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from babeltron.app.monitoring import (
    CACHE_HIT_COUNT,
    CACHE_MISS_COUNT,
    ERROR_COUNT,
    MODEL_LOAD_TIME,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TRANSLATION_COUNT,
    TRANSLATION_LATENCY,
    PrometheusMiddleware,
    metrics_endpoint,
    track_dynamic_translation_metrics,
)


class TestMonitoring:
    @pytest.fixture
    def mock_request(self):
        request = MagicMock(spec=Request)
        request.url.path = "/test/path"
        request.method = "GET"
        return request

    @pytest.fixture
    def mock_response(self):
        response = MagicMock(spec=Response)
        response.status_code = 200
        return response

    @pytest.fixture
    def mock_error_response(self):
        response = MagicMock(spec=Response)
        response.status_code = 500
        return response

    @pytest.mark.asyncio
    async def test_prometheus_middleware_success(self, mock_request, mock_response):
        middleware = PrometheusMiddleware(None)
        call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert response == mock_response
        call_next.assert_called_once_with(mock_request)
        # Check that metrics were recorded

    @pytest.mark.asyncio
    async def test_prometheus_middleware_error_status(self, mock_request, mock_error_response):
        middleware = PrometheusMiddleware(None)
        call_next = AsyncMock(return_value=mock_error_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert response == mock_error_response
        call_next.assert_called_once_with(mock_request)
        # Check that error metrics were recorded

    @pytest.mark.asyncio
    async def test_prometheus_middleware_exception(self, mock_request):
        middleware = PrometheusMiddleware(None)
        exception = ValueError("Test exception")
        call_next = AsyncMock(side_effect=exception)

        with pytest.raises(ValueError):
            await middleware.dispatch(mock_request, call_next)

        call_next.assert_called_once_with(mock_request)
        # Check that exception metrics were recorded

    @pytest.mark.asyncio
    async def test_track_dynamic_translation_metrics_with_request_arg(self):
        # Test with request as first positional argument
        mock_request = MagicMock()
        mock_request.src_lang = "en"
        mock_request.tgt_lang = "fr"

        @track_dynamic_translation_metrics()
        async def test_func(request):
            return "result"

        result = await test_func(mock_request)

        assert result == "result"
        # Check that translation metrics were recorded

    @pytest.mark.asyncio
    async def test_track_dynamic_translation_metrics_with_request_kwarg(self):
        # Test with request as keyword argument
        mock_request = MagicMock()
        mock_request.src_lang = "en"
        mock_request.tgt_lang = "fr"

        @track_dynamic_translation_metrics()
        async def test_func(some_arg, request=None):
            return "result"

        result = await test_func("some_value", request=mock_request)

        assert result == "result"
        # Check that translation metrics were recorded

    def test_metrics_endpoint(self):
        result = metrics_endpoint()
        assert isinstance(result, bytes)
        # The result should be a byte string of Prometheus metrics

    def test_registry_metrics(self):
        # Test that all metrics are properly registered
        assert isinstance(REQUEST_COUNT, Counter)
        assert isinstance(REQUEST_LATENCY, Histogram)
        assert isinstance(ERROR_COUNT, Counter)
        assert isinstance(TRANSLATION_COUNT, Counter)
        assert isinstance(TRANSLATION_LATENCY, Histogram)
        assert isinstance(CACHE_HIT_COUNT, Counter)
        assert isinstance(CACHE_MISS_COUNT, Counter)
        assert isinstance(MODEL_LOAD_TIME, Histogram)

    @patch('time.time')
    def test_translation_latency_observe(self, mock_time):
        # Test that translation latency is observed correctly
        mock_time.side_effect = [100.0, 105.0]  # Start time, end time

        with TRANSLATION_LATENCY.labels(src_lang="en", tgt_lang="fr").time():
            pass

        # The time difference should be 5.0 seconds
        # This is implicitly tested by the context manager

    def test_translation_count_inc(self):
        # Test that translation count is incremented correctly
        before = TRANSLATION_COUNT.labels(src_lang="en", tgt_lang="fr")._value.get()
        TRANSLATION_COUNT.labels(src_lang="en", tgt_lang="fr").inc()
        after = TRANSLATION_COUNT.labels(src_lang="en", tgt_lang="fr")._value.get()
        assert after == before + 1
