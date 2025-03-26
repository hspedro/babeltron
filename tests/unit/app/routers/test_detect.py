import pytest
from unittest.mock import patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient

from babeltron.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_detection_model():
    with patch("babeltron.app.routers.detect.detection_model") as mock_model:
        mock_model.is_loaded = True
        mock_model.architecture = "lingua"
        mock_model.detect.return_value = ("en", 0.98)
        yield mock_model


@pytest.fixture
def mock_cache_service():
    with patch("babeltron.app.routers.detect.cache_service") as mock_cache:
        mock_cache.get_detection.return_value = None
        yield mock_cache


def test_detect_success(client, mock_detection_model):
    # Mock the cache service to avoid cache-related issues
    with patch("babeltron.app.routers.detect.cache_service") as mock_cache:
        mock_cache.get_detection.return_value = None

        response = client.post(
            "/api/v1/detect",
            json={"text": "Hello, how are you?"},
        )
        assert response.status_code == status.HTTP_200_OK

        # Only check the specific fields we care about
        response_json = response.json()
        assert response_json["language"] == "en"
        assert response_json["confidence"] == 0.98

        mock_detection_model.detect.assert_called_once()


def test_detect_model_not_loaded(client):
    with patch("babeltron.app.routers.detect.detection_model") as mock_model:
        mock_model.is_loaded = False
        response = client.post(
            "/api/v1/detect",
            json={"text": "Hello, how are you?"},
        )
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "not loaded" in response.json()["detail"]


def test_detect_model_none(client):
    with patch("babeltron.app.routers.detect.detection_model", None):
        response = client.post(
            "/api/v1/detect",
            json={"text": "Hello, how are you?"},
        )
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "not loaded" in response.json()["detail"]


def test_detect_error(client, mock_detection_model):
    mock_detection_model.detect.side_effect = Exception("Test error")
    response = client.post(
        "/api/v1/detect",
        json={"text": "Hello, how are you?"},
    )
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Test error" in response.json()["detail"]


def test_detect_invalid_request(client):
    response = client.post(
        "/api/v1/detect",
        json={},  # Missing required field 'text'
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@patch("babeltron.app.models.detection.factory.get_detection_model")
@patch("babeltron.app.routers.detect.detection_model", new_callable=MagicMock)
def test_detect_with_cache_disabled(mock_detection_model, mock_get_model, client):
    # Create a mock model
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.architecture = "lingua"
    mock_model.detect.return_value = ("fr", 0.95)

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the detection_model mock
    mock_detection_model.is_loaded = True
    mock_detection_model.architecture = "lingua"
    mock_detection_model.detect.return_value = ("fr", 0.95)

    # Test data with cache disabled
    test_data = {
        "text": "Bonjour, comment ça va?",
        "cache": False
    }

    # Mock the cache service to return a cached result
    with patch("babeltron.app.routers.detect.cache_service") as mock_cache:
        mock_cache.get_detection.return_value = {
            "language": "en",
            "confidence": 0.98,
            "cached": True
        }

        response = client.post("/api/v1/detect", json=test_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["language"] == "fr"  # Should use fresh detection, not cached
        assert data["confidence"] == 0.95
        assert data["cached"] is False

        # Verify the model was called correctly
        mock_detection_model.detect.assert_called_once()
        args, kwargs = mock_detection_model.detect.call_args
        assert args[0] == "Bonjour, comment ça va?"

        # Verify cache was not used
        mock_cache.get_detection.assert_not_called()
        mock_cache.save_detection.assert_not_called()


@patch("babeltron.app.models.detection.factory.get_detection_model")
@patch("babeltron.app.routers.detect.detection_model", new_callable=MagicMock)
def test_detect_with_cache_enabled(mock_detection_model, mock_get_model, client):
    # Create a mock model
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.architecture = "lingua"
    mock_model.detect.return_value = ("fr", 0.95)

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the detection_model mock
    mock_detection_model.is_loaded = True
    mock_detection_model.architecture = "lingua"
    mock_detection_model.detect.return_value = ("fr", 0.95)

    # Test data with cache enabled (default)
    test_data = {
        "text": "Bonjour, comment ça va?"
    }

    # Mock the cache service to return a cached result
    with patch("babeltron.app.routers.detect.cache_service") as mock_cache:
        mock_cache.get_detection.return_value = {
            "language": "en",
            "confidence": 0.98,
            "cached": True
        }

        response = client.post("/api/v1/detect", json=test_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["language"] == "en"  # Should use cached result
        assert data["confidence"] == 0.98
        assert data["cached"] is True

        # Verify the model was not called (using cached result)
        mock_detection_model.detect.assert_not_called()

        # Verify cache was used
        mock_cache.get_detection.assert_called_once()
        mock_cache.save_detection.assert_not_called()  # Should not save since we used cached result
