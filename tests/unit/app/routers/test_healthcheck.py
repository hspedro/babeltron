import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from fastapi import status

from babeltron.app.main import app
from babeltron.app.models.m2m import ModelArchitecture


@pytest.fixture
def client():
    return TestClient(app)


@patch("babeltron.app.routers.healthcheck.translation_model")
def test_healthcheck_model_loaded(mock_model, client):
    # Mock the model as loaded
    mock_model.is_loaded = True
    mock_model.architecture = ModelArchitecture.CPU_STANDARD

    response = client.get("/healthz")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_architecture"] == ModelArchitecture.CPU_STANDARD
    assert "version" in data


@patch("babeltron.app.routers.healthcheck.translation_model")
def test_healthcheck_model_not_loaded(mock_model, client):
    # Mock the model as not loaded
    mock_model.is_loaded = False
    mock_model.architecture = None

    response = client.get("/healthz")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False
    assert data["model_architecture"] is None
    assert "version" in data


@patch("babeltron.app.routers.healthcheck.translation_model")
def test_readiness_model_loaded(mock_model, client):
    # Mock the model as loaded and working
    mock_model.is_loaded = True
    mock_model.architecture = ModelArchitecture.CPU_STANDARD
    mock_model.translate.return_value = "bonjour"

    response = client.get("/readyz")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_architecture"] == ModelArchitecture.CPU_STANDARD
    assert "version" in data


@patch("babeltron.app.routers.healthcheck.translation_model")
def test_readiness_model_not_loaded(mock_model, client):
    # Mock the model as not loaded
    mock_model.is_loaded = False

    response = client.get("/readyz")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert data["status"] == "not ready"
    assert "error" in data
    assert "version" in data


@patch("babeltron.app.routers.healthcheck.translation_model")
def test_readiness_model_error(mock_model, client):
    # Mock the model as loaded but failing on translate
    mock_model.is_loaded = True
    mock_model.translate.side_effect = Exception("Test error")

    response = client.get("/readyz")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert data["status"] == "not ready"
    assert "error" in data
    assert "Test error" in data["error"]
    assert "version" in data
