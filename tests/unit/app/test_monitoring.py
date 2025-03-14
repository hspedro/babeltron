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

    @pytest.mark.asyncio
    async def test_prometheus_middleware_error_status(self, mock_request, mock_error_response):
        middleware = PrometheusMiddleware(None)
        call_next = AsyncMock(return_value=mock_error_response)

        response = await middleware.dispatch(mock_request, call_next)

        assert response == mock_error_response
        call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_prometheus_middleware_exception(self, mock_request):
        middleware = PrometheusMiddleware(None)
        exception = ValueError("Test exception")
        call_next = AsyncMock(side_effect=exception)

        with pytest.raises(ValueError):
            await middleware.dispatch(mock_request, call_next)

        call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_track_dynamic_translation_metrics_with_request_arg(self):
        mock_request = MagicMock()
        mock_request.src_lang = "en"
        mock_request.tgt_lang = "fr"

        @track_dynamic_translation_metrics()
        async def test_func(request):
            return "result"

        result = await test_func(mock_request)

        assert result == "result"

    @pytest.mark.asyncio
    async def test_track_dynamic_translation_metrics_with_request_kwarg(self):
        mock_request = MagicMock()
        mock_request.src_lang = "en"
        mock_request.tgt_lang = "fr"

        @track_dynamic_translation_metrics()
        async def test_func(some_arg, request=None):
            return "result"

        result = await test_func("some_value", request=mock_request)

        assert result == "result"

    def test_metrics_endpoint(self):
        result = metrics_endpoint()
        assert isinstance(result, bytes)

    def test_registry_metrics(self):
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
        mock_time.side_effect = [100.0, 105.0]

        with TRANSLATION_LATENCY.labels(src_lang="en", tgt_lang="fr", detection_used="false").time():
            pass

    def test_translation_count_inc(self):
        before = TRANSLATION_COUNT.labels(src_lang="en", tgt_lang="fr", detection_used="false")._value.get()
        TRANSLATION_COUNT.labels(src_lang="en", tgt_lang="fr", detection_used="false").inc()
        after = TRANSLATION_COUNT.labels(src_lang="en", tgt_lang="fr", detection_used="false")._value.get()
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_track_metrics_with_auto_detection(self):
        """Test that detection_used is set to 'true' when src_lang is 'auto'"""
        mock_request = MagicMock()
        mock_request.src_lang = "auto"
        mock_request.tgt_lang = "fr"
        mock_request.text = "Hello world"

        # Mock the metrics to check they're called with the right labels
        with patch('babeltron.app.monitoring.TRANSLATION_COUNT.labels') as mock_count, \
             patch('babeltron.app.monitoring.TRANSLATION_LATENCY.labels') as mock_latency:

            # Set up the mocks to return objects with the necessary methods
            mock_count.return_value = MagicMock()
            mock_latency.return_value = MagicMock()

            @track_dynamic_translation_metrics()
            async def test_func(request):
                return "result"

            result = await test_func(mock_request)

            # Verify the metrics were called with detection_used="true"
            mock_count.assert_called_with(src_lang="auto", tgt_lang="fr", detection_used="true")
            mock_latency.assert_called_with(src_lang="auto", tgt_lang="fr", detection_used="true")

            assert result == "result"

    @pytest.mark.asyncio
    async def test_track_metrics_with_empty_src_lang(self):
        """Test that detection_used is set to 'true' when src_lang is empty"""
        mock_request = MagicMock()
        mock_request.src_lang = ""
        mock_request.tgt_lang = "fr"
        mock_request.text = "Hello world"

        # Mock the metrics to check they're called with the right labels
        with patch('babeltron.app.monitoring.TRANSLATION_COUNT.labels') as mock_count, \
             patch('babeltron.app.monitoring.TRANSLATION_LATENCY.labels') as mock_latency:

            # Set up the mocks to return objects with the necessary methods
            mock_count.return_value = MagicMock()
            mock_latency.return_value = MagicMock()

            @track_dynamic_translation_metrics()
            async def test_func(request):
                return "result"

            result = await test_func(mock_request)

            # Verify the metrics were called with detection_used="true"
            mock_count.assert_called_with(src_lang="", tgt_lang="fr", detection_used="true")
            mock_latency.assert_called_with(src_lang="", tgt_lang="fr", detection_used="true")

            assert result == "result"

    @pytest.mark.asyncio
    async def test_track_metrics_with_none_src_lang(self):
        """Test that detection_used is set to 'true' when src_lang is None"""
        mock_request = MagicMock()
        mock_request.src_lang = None
        mock_request.tgt_lang = "fr"
        mock_request.text = "Hello world"

        # Mock the metrics to check they're called with the right labels
        with patch('babeltron.app.monitoring.TRANSLATION_COUNT.labels') as mock_count, \
             patch('babeltron.app.monitoring.TRANSLATION_LATENCY.labels') as mock_latency:

            # Set up the mocks to return objects with the necessary methods
            mock_count.return_value = MagicMock()
            mock_latency.return_value = MagicMock()

            @track_dynamic_translation_metrics()
            async def test_func(request):
                return "result"

            result = await test_func(mock_request)

            # Verify the metrics were called with detection_used="true"
            mock_count.assert_called_with(src_lang=None, tgt_lang="fr", detection_used="true")
            mock_latency.assert_called_with(src_lang=None, tgt_lang="fr", detection_used="true")

            assert result == "result"

    @pytest.mark.asyncio
    async def test_track_metrics_with_detection_only(self):
        """Test that detection_used is set to 'true' for detection-only endpoints"""
        mock_request = MagicMock()
        # For detection-only endpoints, we need text but not src_lang
        delattr(mock_request, 'src_lang')  # Ensure src_lang doesn't exist
        mock_request.text = "Hello world"
        mock_request.tgt_lang = "detect"

        # Mock the metrics to check they're called with the right labels
        with patch('babeltron.app.monitoring.TRANSLATION_COUNT.labels') as mock_count, \
             patch('babeltron.app.monitoring.TRANSLATION_LATENCY.labels') as mock_latency:

            # Set up the mocks to return objects with the necessary methods
            mock_count.return_value = MagicMock()
            mock_latency.return_value = MagicMock()

            @track_dynamic_translation_metrics()
            async def test_func(request):
                return "result"

            result = await test_func(mock_request)

            # Verify the metrics were called with detection_used="true"
            mock_count.assert_called_with(src_lang="detect", tgt_lang="detect", detection_used="true")
            mock_latency.assert_called_with(src_lang="detect", tgt_lang="detect", detection_used="true")

            assert result == "result"
