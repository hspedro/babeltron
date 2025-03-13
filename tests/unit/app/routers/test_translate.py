import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, status

from babeltron.app.models.factory import ModelType
from babeltron.app.routers import translate as translate_router

# Create a test app without authentication middleware
test_app = FastAPI()
test_app.include_router(translate_router.router, prefix="/api/v1")

# Common patches for model loading and initialization that should be applied across all tests
model_patches = [
    patch("transformers.AutoModelForSeq2SeqLM.from_pretrained"),
    patch("transformers.AutoTokenizer.from_pretrained"),
    patch("babeltron.app.models.nllb.get_model_path", return_value="/mocked/path"),
    patch("babeltron.app.models.m2m100.get_model_path", return_value="/mocked/path"),
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

    with patch("babeltron.app.models.nllb.NLLBTranslationModel.__new__", return_value=mock):
        yield mock


@pytest.fixture
def mock_m2m_model():
    """Create a mock M2M100 model"""
    mock = MagicMock()
    mock.is_loaded = True
    mock.architecture = "cpu_compiled"
    mock._model_path = "/mocked/path"
    mock.translate.return_value = "Bonjour le monde"

    with patch("babeltron.app.models.m2m100.M2M100TranslationModel.__new__", return_value=mock):
        yield mock


# Patch both the factory function and the global translation_model
@patch("babeltron.app.models.factory.get_translation_model")
@patch("babeltron.app.routers.translate.translation_model", new_callable=MagicMock)
def test_translate_success(mock_translation_model, mock_get_model, mock_m2m_model, client):
    # Set up both mocks to return our mock model
    mock_get_model.return_value = mock_m2m_model

    # Configure the translation_model mock
    for attr_name in ["is_loaded", "architecture", "translate"]:
        setattr(mock_translation_model, attr_name, getattr(mock_m2m_model, attr_name))

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

    # Create a mock tracer
    mock_tracer = MagicMock()

    # Patch the factory to return our NLLB mock
    with patch("babeltron.app.models.factory.get_translation_model", return_value=mock_nllb_model), \
         patch("babeltron.app.routers.translate.translation_model", default_mock), \
         patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):

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

        # Verify the response matches what we expect
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


@patch("babeltron.app.models.factory.get_translation_model")
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


@patch("babeltron.app.models.factory.get_translation_model")
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
    assert isinstance(data, list)
    assert "en" in data
    assert "fr" in data
    assert "es" in data
    assert "de" in data


@patch("babeltron.app.models.factory.get_translation_model")
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


@patch("babeltron.app.models.factory.get_translation_model")
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
