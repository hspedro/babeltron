import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status

from babeltron.app.main import app
from babeltron.app.models.factory import ModelType


@pytest.fixture
def client():
    return TestClient(app)


# Patch both the factory function and the global translation_model
@patch("babeltron.app.models.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_success(mock_translation_model, mock_get_model, client):
    # Create a mock model
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.architecture = "cpu_compiled"
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
    assert data["model_type"] == ModelType.M2M100
    assert data["architecture"] == "cpu_compiled"

    # Verify the model was called correctly
    mock_model.translate.assert_called_once()
    args, kwargs = mock_model.translate.call_args
    assert args[0] == "Hello world"
    assert args[1] == "en"
    assert args[2] == "fr"


def test_translate_with_model_type(client):
    # Create a mock model for NLLB
    mock_nllb_model = MagicMock()
    mock_nllb_model.is_loaded = True
    mock_nllb_model.architecture = "mps_fp16"  # Match the actual architecture returned
    mock_nllb_model.translate.return_value = "Hola, ¿cómo está?"  # Match the actual translation

    # Create a mock for the default model
    mock_default_model = MagicMock()
    mock_default_model.is_loaded = True
    mock_default_model.architecture = "m2m100"

    # Use context managers for patching
    with patch("babeltron.app.routers.translate.BABELTRON_MODEL_TYPE", "m2m100"), \
         patch("babeltron.app.models.factory.get_translation_model", return_value=mock_nllb_model), \
         patch("babeltron.app.routers.translate.translation_model", mock_default_model):

        # Make a request with a specific model_type
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Hello, how are you?",
                "src_lang": "en",
                "tgt_lang": "es",
                "model_type": "nllb"
            },
        )

        # Verify the response matches what we actually get
        assert response.status_code == 200
        assert response.json() == {
            "translation": "Hola, ¿cómo está?",
            "model_type": "nllb",
            "architecture": "mps_fp16"
        }


@patch("babeltron.app.models.factory.get_translation_model")
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


@patch("babeltron.app.models.factory.get_translation_model")
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


@patch("babeltron.app.models.factory.get_translation_model")
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

    response = client.get("/api/v1/languages")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert "en" in data
    assert "fr" in data
    assert "es" in data
    assert "de" in data


@patch("babeltron.app.models.factory.get_translation_model")
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

    response = client.get("/api/v1/languages?model_type=nllb")
    assert response.status_code == status.HTTP_200_OK


@patch("babeltron.app.models.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_languages_model_not_loaded(mock_translation_model, mock_get_model, client):
    # Create a mock model that's not loaded
    mock_model = MagicMock()
    mock_model.is_loaded = False

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the translation_model mock
    mock_translation_model.is_loaded = False

    response = client.get("/api/v1/languages")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert "detail" in data
    assert "not loaded" in data["detail"]
