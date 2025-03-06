from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from babeltron.app.routers.healthcheck import router, HealthResponse, ReadinessResponse


class TestHealthcheckRouter:

    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_healthz_endpoint(self, client):
        response = client.get("/healthz")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "version" in data

    @patch("babeltron.app.routers.healthcheck.model", None)
    def test_healthcheck_with_no_model(self, client):
        response = client.get("/healthz")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False

    @patch("babeltron.app.routers.healthcheck.model", MagicMock())
    def test_healthcheck_with_model(self, client):
        response = client.get("/healthz")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

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
