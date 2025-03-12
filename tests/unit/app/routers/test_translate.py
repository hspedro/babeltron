import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status

from babeltron.app.main import app
from babeltron.app.models.m2m import ModelArchitecture


@pytest.fixture
def client():
    return TestClient(app)


# Patch both the factory method and the global translation_model
@patch("babeltron.app.models.factory.ModelFactory.get_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_success(mock_translation_model, mock_get_model, client):
    # Create a mock model
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.architecture = ModelArchitecture.CPU_COMPILED
    mock_model.translate.return_value = "Bonjour le monde"

    # Set up both mocks to return our mock model
    mock_get_model.return_value = mock_model

    # Configure the translation_model mock
    for attr_name in ["is_loaded", "architecture", "translate"]:
        setattr(mock_translation_model, attr_name, getattr(mock_model, attr_name))

    # Test data
    test_data = {
        "text": "Hello world",
        "src_lang": "en",
        "tgt_lang": "fr"
    }

    response = client.post("/api/v1/translate", json=test_data)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["translation"] == "Bonjour le monde"
    assert data["model_type"] == "m2m100"
    assert data["architecture"] == ModelArchitecture.CPU_COMPILED

    # Verify the model was called correctly
    mock_model.translate.assert_called_once()
    args, kwargs = mock_model.translate.call_args
    assert args[0] == "Hello world"
    assert args[1] == "en"
    assert args[2] == "fr"


@patch("babeltron.app.models.factory.ModelFactory.get_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_with_model_type(mock_translation_model, mock_get_model, client):
    # Create a mock model
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.architecture = ModelArchitecture.CPU_STANDARD
    mock_model.translate.return_value = "Bonjour le monde"

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the translation_model mock
    mock_translation_model.is_loaded = True

    # Test data with model_type
    test_data = {
        "text": "Hello world",
        "src_lang": "en",
        "tgt_lang": "fr",
        "model_type": "m2m100"
    }

    response = client.post("/api/v1/translate", json=test_data)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["translation"] == "Bonjour le monde"
    assert data["model_type"] == "m2m100"
    assert data["architecture"] == ModelArchitecture.CPU_STANDARD


@patch("babeltron.app.models.factory.ModelFactory.get_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_model_not_loaded(mock_translation_model, mock_get_model, client):
    # Create a mock model that's not loaded
    mock_model = MagicMock()
    mock_model.is_loaded = False

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the translation_model mock
    mock_translation_model.is_loaded = False

    # Test data
    test_data = {
        "text": "Hello world",
        "src_lang": "en",
        "tgt_lang": "fr"
    }

    response = client.post("/api/v1/translate", json=test_data)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert "detail" in data
    assert "not loaded" in data["detail"]


@patch("babeltron.app.models.factory.ModelFactory.get_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_model_error(mock_translation_model, mock_get_model, client):
    # Create a mock model that raises an error
    mock_model = MagicMock()
    mock_model.is_loaded = True

    # Make sure the exception is raised when translate is called
    mock_model.translate.side_effect = Exception("Test error")

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the translation_model mock
    mock_translation_model.is_loaded = True
    mock_translation_model.translate.side_effect = Exception("Test error")

    # Test data
    test_data = {
        "text": "Hello world",
        "src_lang": "en",
        "tgt_lang": "fr"
    }

    # Make the request
    response = client.post("/api/v1/translate", json=test_data)

    # Check that we got a 500 error
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert "detail" in data
    assert "Test error" in data["detail"]


@patch("babeltron.app.models.factory.ModelFactory.get_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_languages_endpoint(mock_translation_model, mock_get_model, client):
    # Create a mock model
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.get_languages.return_value = ["en", "fr", "es", "de"]

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the translation_model mock
    mock_translation_model.is_loaded = True
    mock_translation_model.get_languages.return_value = ["en", "fr", "es", "de"]

    response = client.get("/languages")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert "en" in data
    assert "fr" in data
    assert "es" in data
    assert "de" in data


@patch("babeltron.app.models.factory.ModelFactory.get_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_languages_with_model_type(mock_translation_model, mock_get_model, client):
    # Create a mock model
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.get_languages.return_value = ["en", "fr", "es", "de"]

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the translation_model mock
    mock_translation_model.is_loaded = True

    response = client.get("/languages?model_type=m2m100")
    assert response.status_code == status.HTTP_200_OK


@patch("babeltron.app.models.factory.ModelFactory.get_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_languages_model_not_loaded(mock_translation_model, mock_get_model, client):
    # Create a mock model that's not loaded
    mock_model = MagicMock()
    mock_model.is_loaded = False

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the translation_model mock
    mock_translation_model.is_loaded = False

    response = client.get("/languages")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert "detail" in data
    assert "not loaded" in data["detail"]
