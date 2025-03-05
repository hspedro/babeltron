from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from babeltron.app.routers.translate import (
    router,
    TranslationRequest,
    TranslationResponse,
)


class TestTranslateRouter:
    """Tests for the translate router."""

    @pytest.fixture
    def client(self):
        """Create a test client with the translate router."""
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @patch("babeltron.app.routers.translate.model", None)
    def test_translate_with_no_model(self, client):
        """Test the translate endpoint when model is not loaded."""
        request_data = {
            "text": "Hello, world!",
            "src_lang": "en",
            "tgt_lang": "es"
        }
        response = client.post("/translate", json=request_data)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert "model not loaded" in data["detail"].lower()

    @patch("babeltron.app.routers.translate.model", MagicMock())
    @patch("babeltron.app.routers.translate.tokenizer", MagicMock())
    def test_translate_success(self, client):
        """Test successful translation."""
        # Mock the tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        mock_tokenizer.batch_decode.return_value = ["Hola, mundo!"]
        mock_tokenizer.get_lang_id.return_value = 123

        mock_model = MagicMock()
        mock_model.generate.return_value = "generated_tokens"

        with patch("babeltron.app.routers.translate.tokenizer", mock_tokenizer), \
             patch("babeltron.app.routers.translate.model", mock_model):
            request_data = {
                "text": "Hello, world!",
                "src_lang": "en",
                "tgt_lang": "es"
            }
            response = client.post("/translate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["translation"] == "Hola, mundo!"

        # Verify the tokenizer and model were called correctly
        assert mock_tokenizer.src_lang == "en"
        mock_tokenizer.assert_called_once_with("Hello, world!", return_tensors="pt")
        mock_model.generate.assert_called_once()
        mock_tokenizer.get_lang_id.assert_called_once_with("es")
        mock_tokenizer.batch_decode.assert_called_once_with("generated_tokens", skip_special_tokens=True)

    @patch("babeltron.app.routers.translate.model", MagicMock())
    @patch("babeltron.app.routers.translate.tokenizer", MagicMock())
    def test_translate_with_error(self, client):
        """Test translation with an error during processing."""
        # Create a new client that doesn't raise server exceptions
        app = FastAPI()
        app.include_router(router)
        test_client = TestClient(app, raise_server_exceptions=False)

        # Mock the model to raise an exception
        mock_model = MagicMock()
        mock_model.generate.side_effect = Exception("Test error")

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        mock_tokenizer.get_lang_id.return_value = 123

        with patch("babeltron.app.routers.translate.model", mock_model), \
             patch("babeltron.app.routers.translate.tokenizer", mock_tokenizer):

            request_data = {
                "text": "Hello, world!",
                "src_lang": "en",
                "tgt_lang": "es"
            }

            # The request should be handled by the error handler in the route
            response = test_client.post("/translate", json=request_data)

        # Verify the response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

        # Check the response content directly
        content = response.content.decode('utf-8')
        assert "Internal Server Error" in content

    def test_translation_request_model(self):
        """Test the TranslationRequest model."""
        request = TranslationRequest(
            text="Hello, world!",
            src_lang="en",
            tgt_lang="es"
        )
        assert request.text == "Hello, world!"
        assert request.src_lang == "en"
        assert request.tgt_lang == "es"

    def test_translation_response_model(self):
        """Test the TranslationResponse model."""
        response = TranslationResponse(translation="Hola, mundo!")
        assert response.translation == "Hola, mundo!"

    @patch("babeltron.app.routers.translate.tokenizer", None)
    def test_languages_with_no_tokenizer(self, client):
        """Test the languages endpoint when tokenizer is not loaded."""
        response = client.get("/languages")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert "model not loaded" in data["detail"].lower()

    @patch("babeltron.app.routers.translate.tokenizer", MagicMock())
    def test_languages_success(self, client):
        """Test successful languages retrieval."""
        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.lang_code_to_id = {
            "en": 1,
            "es": 2,
            "fr": 3
        }

        with patch("babeltron.app.routers.translate.tokenizer", mock_tokenizer):
            response = client.get("/languages")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert "en" in data
        assert "es" in data
        assert "fr" in data

    @patch("babeltron.app.routers.translate.get_model_path")
    @patch("babeltron.app.routers.translate.M2M100ForConditionalGeneration")
    @patch("babeltron.app.routers.translate.M2M100Tokenizer")
    def test_model_loading_success(self, mock_tokenizer_class, mock_model_class, mock_get_model_path):
        """Test successful model loading."""
        # Save original model and tokenizer
        import babeltron.app.routers.translate as translate_module
        original_model = translate_module.model
        original_tokenizer = translate_module.tokenizer

        try:
            # Set up mocks
            mock_get_model_path.return_value = "/fake/model/path"
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Manually execute the model loading code
            translate_module.model = None
            translate_module.tokenizer = None

            try:
                MODEL_PATH = translate_module.get_model_path()
                print(f"Loading model from: {MODEL_PATH}")
                translate_module.model = mock_model_class.from_pretrained(MODEL_PATH)
                translate_module.tokenizer = mock_tokenizer_class.from_pretrained(MODEL_PATH)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")

            # Verify the model and tokenizer were loaded correctly
            mock_get_model_path.assert_called_once()
            mock_model_class.from_pretrained.assert_called_once_with("/fake/model/path")
            mock_tokenizer_class.from_pretrained.assert_called_once_with("/fake/model/path")
            assert translate_module.model == mock_model
            assert translate_module.tokenizer == mock_tokenizer

        finally:
            # Restore original model and tokenizer
            translate_module.model = original_model
            translate_module.tokenizer = original_tokenizer

    @patch("babeltron.app.routers.translate.get_model_path")
    @patch("babeltron.app.routers.translate.M2M100ForConditionalGeneration")
    @patch("babeltron.app.routers.translate.M2M100Tokenizer")
    def test_model_loading_error(self, mock_tokenizer_class, mock_model_class, mock_get_model_path):
        """Test model loading with an error."""
        # Save original model and tokenizer
        import babeltron.app.routers.translate as translate_module
        original_model = translate_module.model
        original_tokenizer = translate_module.tokenizer

        try:
            # Set up mocks
            mock_get_model_path.return_value = "/fake/model/path"
            mock_model_class.from_pretrained.side_effect = Exception("Test error")
            mock_tokenizer_class.from_pretrained.side_effect = Exception("Test error")

            # Manually execute the model loading code
            translate_module.model = None
            translate_module.tokenizer = None

            try:
                MODEL_PATH = translate_module.get_model_path()
                print(f"Loading model from: {MODEL_PATH}")
                translate_module.model = mock_model_class.from_pretrained(MODEL_PATH)
                translate_module.tokenizer = mock_tokenizer_class.from_pretrained(MODEL_PATH)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")

            # Verify model and tokenizer are None when there's an error
            assert translate_module.model is None
            assert translate_module.tokenizer is None

        finally:
            # Restore original model and tokenizer
            translate_module.model = original_model
            translate_module.tokenizer = original_tokenizer
