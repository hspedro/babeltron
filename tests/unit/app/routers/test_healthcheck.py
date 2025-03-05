from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from babeltron.app.routers.healthcheck import router, HealthResponse, ReadinessResponse


class TestHealthcheckRouter:
    """Tests for the healthcheck router."""

    @pytest.fixture
    def client(self):
        """Create a test client with the healthcheck router."""
        app = FastAPI()
        app.include_router(router)
        # Use the standard approach for this version of TestClient
        return TestClient(app)

    def test_healthcheck_endpoint(self, client):
        """Test the /healthcheck endpoint."""
        response = client.get("/healthcheck")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "version" in data

    def test_healthz_endpoint(self, client):
        """Test the /healthz endpoint."""
        response = client.get("/healthz")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "version" in data

    @patch("babeltron.app.routers.healthcheck.model", None)
    def test_healthcheck_with_no_model(self, client):
        """Test the healthcheck when model is not loaded."""
        response = client.get("/healthz")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False

    @patch("babeltron.app.routers.healthcheck.model", MagicMock())
    def test_healthcheck_with_model(self, client):
        """Test the healthcheck when model is loaded."""
        response = client.get("/healthz")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    @patch("babeltron.app.routers.healthcheck.model", None)
    def test_readiness_with_no_model(self, client):
        """Test the readiness probe when model is not loaded."""
        response = client.get("/readiness")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["status"] == "not ready"
        assert "error" in data
        assert "version" in data

    @patch("babeltron.app.routers.healthcheck.model", MagicMock())
    @patch("babeltron.app.routers.healthcheck.tokenizer", MagicMock())
    def test_readiness_with_model(self, client):
        """Test the readiness probe when model is loaded."""
        # Mock the model.generate method
        mock_model = MagicMock()
        mock_model.generate.return_value = "test output"

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}

        with patch("babeltron.app.routers.healthcheck.model", mock_model), \
             patch("babeltron.app.routers.healthcheck.tokenizer", mock_tokenizer):
            response = client.get("/readiness")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ready"
        assert "version" in data

    @patch("babeltron.app.routers.healthcheck.model", MagicMock())
    @patch("babeltron.app.routers.healthcheck.tokenizer", MagicMock())
    def test_readiness_with_model_error(self, client):
        """Test the readiness probe when model raises an error."""
        # Mock the model to raise an exception
        mock_model = MagicMock()
        mock_model.generate.side_effect = Exception("Test error")

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}

        with patch("babeltron.app.routers.healthcheck.model", mock_model), \
             patch("babeltron.app.routers.healthcheck.tokenizer", mock_tokenizer):
            response = client.get("/readiness")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        print("data: ", data)
        assert data["status"] == "not ready"
        assert "error" in data
        assert "Test error" in data["error"]
        assert "version" in data

    def test_health_response_model(self):
        """Test the HealthResponse model."""
        response = HealthResponse(status="ok", model_loaded=True, version="1.0.0")
        assert response.status == "ok"
        assert response.model_loaded is True
        assert response.version == "1.0.0"

    def test_readiness_response_model(self):
        """Test the ReadinessResponse model."""
        response = ReadinessResponse(status="ready", version="1.0.0")
        assert response.status == "ready"
        assert response.version == "1.0.0"
        assert response.error is None

        error_response = ReadinessResponse(status="not ready", version="1.0.0", error="Test error")
        assert error_response.status == "not ready"
        assert error_response.version == "1.0.0"
        assert error_response.error == "Test error"
