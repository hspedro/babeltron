import pytest
from unittest.mock import patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from babeltron.app.routers.translate import router as translate_router
from babeltron.app.routers.detect import router as detect_router


class TestCacheIntegration:
    """Tests for cache integration in the endpoints"""

    @pytest.fixture
    def mock_translation_model(self):
        """Mock for the translation model"""
        mock = MagicMock()
        mock.model_type = "m2m100"
        mock.architecture = "M2M100ForConditionalGeneration"
        mock.is_loaded = True
        mock.translate.return_value = "Bonjour le monde"
        return mock

    @pytest.fixture
    def mock_detection_model(self):
        """Mock for the detection model"""
        mock = MagicMock()
        mock.architecture = "Lingua"
        mock.is_loaded = True
        mock.detect.return_value = ("en", 0.98)
        return mock

    @pytest.fixture
    def app(self, mock_translation_model, mock_detection_model):
        """Create a test app with the translate and detect routers"""
        app = FastAPI()
        app.include_router(translate_router)
        app.include_router(detect_router)

        # Patch the models
        with patch("babeltron.app.routers.translate.translation_model", mock_translation_model), \
             patch("babeltron.app.routers.translate.detection_model", mock_detection_model), \
             patch("babeltron.app.routers.detect.detection_model", mock_detection_model):
            yield app

    @pytest.fixture
    def client(self, app):
        """Test client for the FastAPI app"""
        return TestClient(app)

    @pytest.fixture
    def mock_cache_service(self):
        """Mock for the CacheService"""
        with patch("babeltron.app.routers.translate.cache_service") as mock_translate, \
             patch("babeltron.app.routers.detect.cache_service") as mock_detect:
            # Configure both mocks to have the same behavior
            for mock in [mock_translate, mock_detect]:
                mock.get_translation.return_value = None
                mock.get_detection.return_value = None

            yield {
                "translate": mock_translate,
                "detect": mock_detect
            }

    def test_translate_cache_miss(self, client, mock_cache_service):
        """Test translation endpoint with cache miss"""
        # Setup
        mock_cache_service["translate"].get_translation.return_value = None

        # Execute
        response = client.post(
            "/translate",
            json={"text": "Hello world", "src_lang": "en", "tgt_lang": "fr"}
        )

        # Verify
        assert response.status_code == 200
        assert response.json()["cached"] is False
        mock_cache_service["translate"].get_translation.assert_called_once()
        mock_cache_service["translate"].save_translation.assert_called_once()

    def test_translate_cache_hit(self, client, mock_cache_service):
        """Test translation endpoint with cache hit"""
        # Setup
        cached_result = {
            "translation": "Bonjour le monde",
            "model_type": "m2m100",
            "architecture": "M2M100ForConditionalGeneration",
            "detected_lang": None,
            "detection_confidence": None
        }
        mock_cache_service["translate"].get_translation.return_value = cached_result

        # Execute
        response = client.post(
            "/translate",
            json={"text": "Hello world", "src_lang": "en", "tgt_lang": "fr"}
        )

        # Verify
        assert response.status_code == 200
        assert response.json()["cached"] is True
        assert response.json()["translation"] == "Bonjour le monde"
        mock_cache_service["translate"].get_translation.assert_called_once()
        mock_cache_service["translate"].save_translation.assert_not_called()

    def test_detect_cache_miss(self, client, mock_cache_service):
        """Test detection endpoint with cache miss"""
        # Setup
        mock_cache_service["detect"].get_detection.return_value = None

        # Execute
        response = client.post(
            "/detect",
            json={"text": "Hello world"}
        )

        # Verify
        assert response.status_code == 200
        assert response.json()["cached"] is False
        mock_cache_service["detect"].get_detection.assert_called_once()
        mock_cache_service["detect"].save_detection.assert_called_once()

    def test_detect_cache_hit(self, client, mock_cache_service):
        """Test detection endpoint with cache hit"""
        # Setup
        cached_result = {
            "language": "en",
            "confidence": 0.98
        }
        mock_cache_service["detect"].get_detection.return_value = cached_result

        # Execute
        response = client.post(
            "/detect",
            json={"text": "Hello world"}
        )

        # Verify
        assert response.status_code == 200
        assert response.json()["cached"] is True
        assert response.json()["language"] == "en"
        assert response.json()["confidence"] == 0.98
        mock_cache_service["detect"].get_detection.assert_called_once()
        mock_cache_service["detect"].save_detection.assert_not_called()

    def test_translate_with_auto_detection_cache(self, client, mock_cache_service):
        """Test translation endpoint with auto detection and cache"""
        # Setup
        mock_cache_service["translate"].get_translation.return_value = None

        # Execute
        response = client.post(
            "/translate",
            json={"text": "Hello world", "src_lang": "auto", "tgt_lang": "fr"}
        )

        # Verify
        assert response.status_code == 200
        assert response.json()["cached"] is False
        # The src_lang should be the detected language in the cache key
        mock_cache_service["translate"].save_translation.assert_called_once()
