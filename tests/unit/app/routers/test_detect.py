import pytest
from unittest.mock import patch
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
