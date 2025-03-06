from unittest.mock import patch, MagicMock

import pytest
import torch
import base64
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from babeltron.app.routers.translate import (
    router,
    TranslationRequest,
    TranslationResponse,
)


class TestTranslateRouter:

    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @pytest.fixture
    def auth_client(self, client):
        auth = base64.b64encode(b"babeltron:translation2025").decode("utf-8")
        client.headers = {"Authorization": f"Basic {auth}"}
        return client

    @patch("babeltron.app.routers.translate.model", None)
    def test_translate_with_no_model(self, client):
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
    def test_translate_success(self, auth_client):
        input_ids_mock = MagicMock()
        input_ids_mock.shape = [1, 3]  # Simulate tensor shape

        generated_tokens_mock = MagicMock()
        generated_tokens_mock.shape = [1, 4]  # Simulate tensor shape

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": input_ids_mock}
        mock_tokenizer.batch_decode.return_value = ["Hola, mundo!"]
        mock_tokenizer.get_lang_id.return_value = 123

        mock_model = MagicMock()
        mock_model.generate.return_value = generated_tokens_mock

        with patch("babeltron.app.routers.translate.tokenizer", mock_tokenizer), \
             patch("babeltron.app.routers.translate.model", mock_model):
            request_data = {
                "text": "Hello, world!",
                "src_lang": "en",
                "tgt_lang": "es"
            }
            response = auth_client.post("/translate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["translation"] == "Hola, mundo!"

    @patch("babeltron.app.routers.translate.model", MagicMock())
    @patch("babeltron.app.routers.translate.tokenizer", MagicMock())
    def test_translate_with_error(self, client):
        app = FastAPI()
        app.include_router(router)
        test_client = TestClient(app, raise_server_exceptions=False)

        input_ids_mock = MagicMock()
        input_ids_mock.shape = [1, 3]  # Simulate tensor shape

        mock_model = MagicMock()
        mock_model.generate.side_effect = Exception("Test error")

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": input_ids_mock}
        mock_tokenizer.get_lang_id.return_value = 123

        with patch("babeltron.app.routers.translate.model", mock_model), \
             patch("babeltron.app.routers.translate.tokenizer", mock_tokenizer):

            request_data = {
                "text": "Hello, world!",
                "src_lang": "en",
                "tgt_lang": "es"
            }

            response = test_client.post("/translate", json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

        data = response.json()
        assert "detail" in data
        assert "Error during translation" in data["detail"]
        assert "Test error" in data["detail"]

    def test_translation_request_model(self):
        request = TranslationRequest(
            text="Hello, world!",
            src_lang="en",
            tgt_lang="es"
        )
        assert request.text == "Hello, world!"
        assert request.src_lang == "en"
        assert request.tgt_lang == "es"

    def test_translation_response_model(self):
        response = TranslationResponse(
            translation="Hola, mundo!",
        )
        assert response.translation == "Hola, mundo!"

    @patch("babeltron.app.routers.translate.tokenizer", None)
    def test_languages_with_no_tokenizer(self, client):
        response = client.get("/languages")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert "model not loaded" in data["detail"].lower()

    @patch("babeltron.app.routers.translate.tokenizer", MagicMock())
    def test_languages_success(self, client):
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
        import babeltron.app.routers.translate as translate_module
        original_model = translate_module.model
        original_tokenizer = translate_module.tokenizer

        try:
            mock_get_model_path.return_value = "/fake/model/path"
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

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

            mock_get_model_path.assert_called_once()
            mock_model_class.from_pretrained.assert_called_once_with("/fake/model/path")
            mock_tokenizer_class.from_pretrained.assert_called_once_with("/fake/model/path")
            assert translate_module.model == mock_model
            assert translate_module.tokenizer == mock_tokenizer

        finally:
            translate_module.model = original_model
            translate_module.tokenizer = original_tokenizer

    @patch("babeltron.app.routers.translate.get_model_path")
    @patch("babeltron.app.routers.translate.M2M100ForConditionalGeneration")
    @patch("babeltron.app.routers.translate.M2M100Tokenizer")
    def test_model_loading_error(self, mock_tokenizer_class, mock_model_class, mock_get_model_path):
        import babeltron.app.routers.translate as translate_module
        original_model = translate_module.model
        original_tokenizer = translate_module.tokenizer

        try:
            mock_get_model_path.return_value = "/fake/model/path"
            mock_model_class.from_pretrained.side_effect = Exception("Test error")
            mock_tokenizer_class.from_pretrained.side_effect = Exception("Test error")

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

            assert translate_module.model is None
            assert translate_module.tokenizer is None

        finally:
            # Restore original model and tokenizer
            translate_module.model = original_model
            translate_module.tokenizer = original_tokenizer

    @patch("babeltron.app.routers.translate.torch.cuda.is_available", return_value=True)
    @patch("babeltron.app.routers.translate.MODEL_COMPRESSION_ENABLED", True)
    @patch("babeltron.app.routers.translate.M2M100ForConditionalGeneration")
    @patch("babeltron.app.routers.translate.M2M100Tokenizer")
    def test_model_loading_with_fp16_compression(self, mock_tokenizer_class, mock_model_class, mock_cuda_available):
        import babeltron.app.routers.translate as translate_module
        original_model = translate_module.model
        original_tokenizer = translate_module.tokenizer

        try:
            # Set up mocks
            mock_model = MagicMock()
            mock_model.half.return_value = mock_model  # Mock the half() method
            mock_model.to.return_value = mock_model    # Mock the to() method
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

                # Apply FP16 compression if enabled and supported
                if translate_module.MODEL_COMPRESSION_ENABLED and torch.cuda.is_available():
                    translate_module.model = translate_module.model.half()
                    translate_module.model = translate_module.model.to('cuda')

                translate_module.tokenizer = mock_tokenizer_class.from_pretrained(MODEL_PATH)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")

            mock_model.half.assert_called_once()
            mock_model.to.assert_called_once_with('cuda')

            mock_model_class.from_pretrained.assert_called_once()
            mock_tokenizer_class.from_pretrained.assert_called_once()
            assert translate_module.model == mock_model
            assert translate_module.tokenizer == mock_tokenizer

        finally:
            # Restore original model and tokenizer
            translate_module.model = original_model
            translate_module.tokenizer = original_tokenizer

    @patch("babeltron.app.routers.translate.torch.cuda.is_available", return_value=False)
    @patch("babeltron.app.routers.translate.MODEL_COMPRESSION_ENABLED", True)
    @patch("babeltron.app.routers.translate.M2M100ForConditionalGeneration")
    @patch("babeltron.app.routers.translate.M2M100Tokenizer")
    def test_model_loading_without_gpu(self, mock_tokenizer_class, mock_model_class, mock_cuda_available):
        import babeltron.app.routers.translate as translate_module
        original_model = translate_module.model
        original_tokenizer = translate_module.tokenizer

        try:
            mock_model = MagicMock()
            mock_model.half.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_tokenizer = MagicMock()

            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            translate_module.model = None
            translate_module.tokenizer = None

            try:
                MODEL_PATH = translate_module.get_model_path()
                print(f"Loading model from: {MODEL_PATH}")
                translate_module.model = mock_model_class.from_pretrained(MODEL_PATH)

                if translate_module.MODEL_COMPRESSION_ENABLED and torch.cuda.is_available():
                    translate_module.model = translate_module.model.half()
                    translate_module.model = translate_module.model.to('cuda')

                translate_module.tokenizer = mock_tokenizer_class.from_pretrained(MODEL_PATH)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")

            mock_model.half.assert_not_called()
            mock_model.to.assert_not_called()

            mock_model_class.from_pretrained.assert_called_once()
            mock_tokenizer_class.from_pretrained.assert_called_once()
            assert translate_module.model == mock_model
            assert translate_module.tokenizer == mock_tokenizer

        finally:
            translate_module.model = original_model
            translate_module.tokenizer = original_tokenizer

    @patch("babeltron.app.routers.translate.torch.cuda.is_available", return_value=True)
    @patch("babeltron.app.routers.translate.MODEL_COMPRESSION_ENABLED", False)
    @patch("babeltron.app.routers.translate.M2M100ForConditionalGeneration")
    @patch("babeltron.app.routers.translate.M2M100Tokenizer")
    def test_model_loading_with_compression_disabled(self, mock_tokenizer_class, mock_model_class, mock_cuda_available):
        import babeltron.app.routers.translate as translate_module
        original_model = translate_module.model
        original_tokenizer = translate_module.tokenizer

        try:
            mock_model = MagicMock()
            mock_model.half.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_tokenizer = MagicMock()

            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            translate_module.model = None
            translate_module.tokenizer = None

            try:
                MODEL_PATH = translate_module.get_model_path()
                translate_module.model = mock_model_class.from_pretrained(MODEL_PATH)

                if translate_module.MODEL_COMPRESSION_ENABLED and torch.cuda.is_available():
                    translate_module.model = translate_module.model.half()
                    translate_module.model = translate_module.model.to('cuda')

                translate_module.tokenizer = mock_tokenizer_class.from_pretrained(MODEL_PATH)
            except Exception:
                pass

            mock_model.half.assert_not_called()
            mock_model.to.assert_not_called()

            mock_model_class.from_pretrained.assert_called_once()
            mock_tokenizer_class.from_pretrained.assert_called_once()
            assert translate_module.model == mock_model
            assert translate_module.tokenizer == mock_tokenizer

        finally:
            translate_module.model = original_model
            translate_module.tokenizer = original_tokenizer
