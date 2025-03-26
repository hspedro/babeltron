import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, status

from babeltron.app.routers import translate as translate_router

# Create a test app without authentication middleware
test_app = FastAPI()
test_app.include_router(translate_router.router, prefix="/api/v1")

# Common patches for model loading and initialization that should be applied across all tests
model_patches = [
    patch("transformers.AutoModelForSeq2SeqLM.from_pretrained"),
    patch("transformers.AutoTokenizer.from_pretrained"),
    patch("babeltron.app.models.translation.nllb.get_model_path", return_value="/mocked/path"),
    patch("babeltron.app.models.translation.m2m100.get_model_path", return_value="/mocked/path"),
]


@pytest.fixture
def client():
    # Create a client with our test app that has no auth middleware
    with TestClient(test_app) as client:
        yield client


@pytest.fixture(autouse=True)
def patch_model_loading():
    """This fixture patches model loading across all tests"""
    # Start all the patches
    started_patches = []
    for p in model_patches:
        started_patches.append(p.start())

    yield

    # Stop all patches after the test is done
    for p in model_patches:
        p.stop()


@pytest.fixture
def mock_nllb_model():
    """Create a mock NLLB model"""
    mock = MagicMock()
    mock.is_loaded = True
    mock.architecture = "mps_fp16"
    mock._model_path = "/mocked/path"
    mock.translate.return_value = "Hola, ¿cómo está?"

    with patch("babeltron.app.models.translation.nllb.NLLBTranslationModel.__new__", return_value=mock):
        yield mock


@pytest.fixture
def mock_m2m_model():
    """Create a mock M2M100 model"""
    mock = MagicMock()
    mock.is_loaded = True
    mock.architecture = "cpu_compiled"
    mock._model_path = "/mocked/path"
    mock.translate.return_value = "Bonjour le monde"

    with patch("babeltron.app.models.translation.m2m100.M2M100TranslationModel.__new__", return_value=mock):
        yield mock


@pytest.fixture
def mock_detection_model():
    """Create a mock detection model"""
    mock = MagicMock()
    mock.is_loaded = True
    mock.architecture = "lingua"
    mock.detect.return_value = ("fr", 0.95)

    with patch("babeltron.app.routers.translate.detection_model", mock):
        yield mock


# Patch both the factory function and the global translation_model
@patch("babeltron.app.models.translation.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_success(mock_translation_model, mock_get_model, mock_m2m_model, client):
    # Set up both mocks to return our mock model
    mock_get_model.return_value = mock_m2m_model

    # Configure the translation_model mock
    for attr_name in ["is_loaded", "architecture", "translate"]:
        setattr(mock_translation_model, attr_name, getattr(mock_m2m_model, attr_name))

    # Set the model_type attribute on the mock
    mock_translation_model.model_type = "m2m100"
    mock_m2m_model.model_type = "m2m100"

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
    assert data["architecture"] == "cpu_compiled"
    assert data["detected_lang"] is None
    assert data["detection_confidence"] is None

    # Verify the model was called correctly
    mock_m2m_model.translate.assert_called_once()
    args, kwargs = mock_m2m_model.translate.call_args
    assert args[0] == "Hello world"
    assert args[1] == "en"
    assert args[2] == "fr"


def test_translate_with_model_type(mock_nllb_model, client):
    # Create a mock for the default model
    default_mock = MagicMock()
    default_mock.is_loaded = True
    default_mock.architecture = "m2m100"
    default_mock.model_type = "m2m100"

    # Set the model_type attribute on the mock
    mock_nllb_model.model_type = "nllb"
    # Ensure translate returns a string, not a MagicMock
    mock_nllb_model.translate.return_value = "Hola, ¿cómo está?"

    # Create a mock tracer
    mock_tracer = MagicMock()

    # Patch the factory to return our NLLB mock
    with patch("babeltron.app.models.translation.factory.get_translation_model", return_value=mock_nllb_model), \
         patch("babeltron.app.routers.translate.translation_model", mock_nllb_model), \
         patch("opentelemetry.trace.get_tracer", return_value=mock_tracer), \
         patch("babeltron.app.routers.translate.cache_service") as mock_cache:

        # Configure the cache to return a miss
        mock_cache.get_translation.return_value = None

        # Make a request with a specific model_type
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Hello, how are you?",
                "src_lang": "en",
                "tgt_lang": "es"
            },
        )

        # Verify the response matches what we expect
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["translation"] == "Hola, ¿cómo está?"
        assert response_json["model_type"] == "nllb"
        assert response_json["architecture"] == "mps_fp16"
        assert response_json["detected_lang"] is None
        assert response_json["detection_confidence"] is None
        assert "cached" in response_json


@patch("babeltron.app.models.translation.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_model_not_loaded(mock_translation_model, mock_get_model, client):
    # Create a mock model that's not loaded
    mock_model = MagicMock()
    mock_model.is_loaded = False
    mock_model._model_path = "/mocked/path"

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


@patch("babeltron.app.models.translation.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_model_error(mock_translation_model, mock_get_model, client):
    # Create a mock model that raises an error
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model._model_path = "/mocked/path"

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


@patch("babeltron.app.models.translation.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_languages_endpoint(mock_translation_model, mock_get_model, client):
    # Create a mock model
    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model._model_path = "/mocked/path"
    mock_model.get_languages.return_value = ["en", "fr", "es", "de"]

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the translation_model mock
    mock_translation_model.is_loaded = True
    mock_translation_model.get_languages.return_value = ["en", "fr", "es", "de"]

    response = client.get("/api/v1/languages")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, dict)
    assert "languages" in data
    assert isinstance(data["languages"], list)
    assert "en" in data["languages"]
    assert "fr" in data["languages"]
    assert "es" in data["languages"]


@patch("babeltron.app.models.translation.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_languages_with_model_type(mock_translation_model, mock_get_model, mock_nllb_model, client):
    # Configure translation_model mock
    mock_translation_model.is_loaded = True

    # Make the factory return our mock model
    mock_get_model.return_value = mock_nllb_model

    # Set up mock languages
    mock_nllb_model.get_languages.return_value = ["en", "fr", "es", "de"]

    response = client.get("/api/v1/languages?model_type=nllb")
    assert response.status_code == status.HTTP_200_OK


@patch("babeltron.app.models.translation.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_languages_model_not_loaded(mock_translation_model, mock_get_model, client):
    # Create a mock model that's not loaded
    mock_model = MagicMock()
    mock_model.is_loaded = False
    mock_model._model_path = "/mocked/path"

    # Make the factory return our mock model
    mock_get_model.return_value = mock_model

    # Configure the translation_model mock
    mock_translation_model.is_loaded = False

    response = client.get("/api/v1/languages")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert "detail" in data
    assert "not loaded" in data["detail"]


# Tests for automatic language detection
def test_translate_with_auto_detection(mock_m2m_model, mock_detection_model, client):
    # Create a mock tracer
    mock_tracer = MagicMock()

    # Set the model_type attribute on the mock
    mock_m2m_model.model_type = "m2m100"

    # Patch the necessary components
    with patch("babeltron.app.models.translation.factory.get_translation_model", return_value=mock_m2m_model), \
         patch("babeltron.app.routers.translate.translation_model", mock_m2m_model), \
         patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):

        # Test with "auto" as source language
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Bonjour, comment ça va?",
                "src_lang": "auto",
                "tgt_lang": "en"
            },
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["translation"] == "Bonjour le monde"  # Mock response
        assert data["detected_lang"] == "fr"
        assert data["detection_confidence"] == 0.95
        assert data["model_type"] == "m2m100"
        assert data["architecture"] == "cpu_compiled"

        # Verify the detection model was called
        mock_detection_model.detect.assert_called_once()

        # Verify the translation model was called with the detected language
        mock_m2m_model.translate.assert_called_once()
        args, kwargs = mock_m2m_model.translate.call_args
        assert args[1] == "fr"  # Source language should be the detected one


def test_translate_with_empty_src_lang(mock_m2m_model, mock_detection_model, client):
    # Create a mock tracer
    mock_tracer = MagicMock()

    # Set the model_type attribute on the mock
    mock_m2m_model.model_type = "m2m100"

    # Patch the necessary components
    with patch("babeltron.app.models.translation.factory.get_translation_model", return_value=mock_m2m_model), \
         patch("babeltron.app.routers.translate.translation_model", mock_m2m_model), \
         patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):

        # Test with empty string as source language
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Bonjour, comment ça va?",
                "src_lang": "",
                "tgt_lang": "en"
            },
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["translation"] == "Bonjour le monde"  # Mock response
        assert data["detected_lang"] == "fr"
        assert data["detection_confidence"] == 0.95
        assert data["model_type"] == "m2m100"
        assert data["architecture"] == "cpu_compiled"

        # Verify the detection model was called
        mock_detection_model.detect.assert_called_once()


def test_translate_with_missing_src_lang(mock_m2m_model, mock_detection_model, client):
    # Create a mock tracer
    mock_tracer = MagicMock()

    # Set the model_type attribute on the mock
    mock_m2m_model.model_type = "m2m100"

    # Patch the necessary components
    with patch("babeltron.app.models.translation.factory.get_translation_model", return_value=mock_m2m_model), \
         patch("babeltron.app.routers.translate.translation_model", mock_m2m_model), \
         patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):

        # Test with no source language provided
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Bonjour, comment ça va?",
                "tgt_lang": "en"
            },
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["translation"] == "Bonjour le monde"  # Mock response
        assert data["detected_lang"] == "fr"
        assert data["detection_confidence"] == 0.95
        assert data["model_type"] == "m2m100"
        assert data["architecture"] == "cpu_compiled"

        # Verify the detection model was called
        mock_detection_model.detect.assert_called_once()


def test_translate_detection_model_not_loaded(mock_m2m_model, client):
    # Create a mock detection model that's not loaded
    mock_detection = MagicMock()
    mock_detection.is_loaded = False

    # Patch the necessary components
    with patch("babeltron.app.models.translation.factory.get_translation_model", return_value=mock_m2m_model), \
         patch("babeltron.app.routers.translate.translation_model", mock_m2m_model), \
         patch("babeltron.app.routers.translate.detection_model", mock_detection):

        # Test with "auto" as source language
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Bonjour, comment ça va?",
                "src_lang": "auto",
                "tgt_lang": "en"
            },
        )

        # Verify we get an error about the detection model
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "detection model not loaded" in data["detail"].lower()


def test_translate_detection_error(mock_m2m_model, client):
    # Create a mock detection model that raises an error
    mock_detection = MagicMock()
    mock_detection.is_loaded = True
    mock_detection.architecture = "lingua"
    mock_detection.detect.side_effect = Exception("Detection test error")

    # Patch the necessary components
    with patch("babeltron.app.models.translation.factory.get_translation_model", return_value=mock_m2m_model), \
         patch("babeltron.app.routers.translate.translation_model", mock_m2m_model), \
         patch("babeltron.app.routers.translate.detection_model", mock_detection):

        # Test with "auto" as source language
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Bonjour, comment ça va?",
                "src_lang": "auto",
                "tgt_lang": "en"
            },
        )

        # Verify we get an error about the detection
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "detection error" in data["detail"].lower()
        assert "test error" in data["detail"].lower()


@patch("babeltron.app.models.translation.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_with_cache_disabled(mock_translation_model, mock_get_model, mock_m2m_model, client):
    # Set up both mocks to return our mock model
    mock_get_model.return_value = mock_m2m_model

    # Configure the translation_model mock
    for attr_name in ["is_loaded", "architecture", "translate"]:
        setattr(mock_translation_model, attr_name, getattr(mock_m2m_model, attr_name))

    # Set the model_type attribute on the mock
    mock_translation_model.model_type = "m2m100"
    mock_m2m_model.model_type = "m2m100"

    # Test data with cache disabled
    test_data = {
        "text": "Hello world",
        "src_lang": "en",
        "tgt_lang": "fr",
        "cache": False
    }

    # Mock the cache service to return a cached result
    with patch("babeltron.app.routers.translate.cache_service") as mock_cache:
        mock_cache.get_translation.return_value = {
            "translation": "Cached translation",
            "model_type": "m2m100",
            "architecture": "cpu_compiled",
            "cached": True
        }

        response = client.post("/api/v1/translate", json=test_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["translation"] == "Bonjour le monde"  # Should use fresh translation, not cached
        assert data["model_type"] == "m2m100"
        assert data["architecture"] == "cpu_compiled"
        assert data["detected_lang"] is None
        assert data["detection_confidence"] is None
        assert data["cached"] is False

        # Verify the model was called correctly
        mock_m2m_model.translate.assert_called_once()
        args, kwargs = mock_m2m_model.translate.call_args
        assert args[0] == "Hello world"
        assert args[1] == "en"
        assert args[2] == "fr"

        # Verify cache was not used
        mock_cache.get_translation.assert_not_called()
        mock_cache.save_translation.assert_not_called()


@patch("babeltron.app.models.translation.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_with_cache_enabled(mock_translation_model, mock_get_model, mock_m2m_model, client):
    # Set up both mocks to return our mock model
    mock_get_model.return_value = mock_m2m_model

    # Configure the translation_model mock
    for attr_name in ["is_loaded", "architecture", "translate"]:
        setattr(mock_translation_model, attr_name, getattr(mock_m2m_model, attr_name))

    # Set the model_type attribute on the mock
    mock_translation_model.model_type = "m2m100"
    mock_m2m_model.model_type = "m2m100"

    # Test data with cache enabled (default)
    test_data = {
        "text": "Hello world",
        "src_lang": "en",
        "tgt_lang": "fr"
    }

    # Mock the cache service to return a cached result
    with patch("babeltron.app.routers.translate.cache_service") as mock_cache:
        mock_cache.get_translation.return_value = {
            "translation": "Cached translation",
            "model_type": "m2m100",
            "architecture": "cpu_compiled",
            "cached": True
        }

        response = client.post("/api/v1/translate", json=test_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["translation"] == "Cached translation"  # Should use cached translation
        assert data["model_type"] == "m2m100"
        assert data["architecture"] == "cpu_compiled"
        assert data["detected_lang"] is None
        assert data["detection_confidence"] is None
        assert data["cached"] is True

        # Verify the model was not called (using cached result)
        mock_m2m_model.translate.assert_not_called()

        # Verify cache was used
        mock_cache.get_translation.assert_called_once()
        mock_cache.save_translation.assert_not_called()  # Should not save since we used cached result
