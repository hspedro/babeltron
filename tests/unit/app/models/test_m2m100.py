import pytest
from unittest.mock import MagicMock, patch
import torch

from babeltron.app.models.m2m100 import M2M100TranslationModel, ModelArchitecture, get_translation_model


class TestM2M100TranslationModel:
    @pytest.fixture(autouse=True)
    def mock_torch(self):
        with patch("babeltron.app.models.m2m100.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.tensor = torch.tensor
            mock_torch.compile.return_value = MagicMock()
            yield mock_torch

    @pytest.fixture
    def mock_transformers(self):
        with patch("babeltron.app.models.m2m100.M2M100ForConditionalGeneration") as mock_model:
            with patch("babeltron.app.models.m2m100.M2M100Tokenizer") as mock_tokenizer:
                tokenizer_instance = MagicMock()
                tokenizer_instance.lang_code_to_id = {"en": 0, "fr": 1, "es": 2, "de": 3}
                tokenizer_instance.batch_decode.return_value = ["Bonjour le monde"]
                mock_tokenizer.from_pretrained.return_value = tokenizer_instance

                model_instance = MagicMock()
                model_instance.generate.return_value = torch.tensor([[1, 2, 3]])
                mock_model.from_pretrained.return_value = model_instance

                yield {
                    "model_class": mock_model,
                    "tokenizer_class": mock_tokenizer,
                    "model_instance": model_instance,
                    "tokenizer_instance": tokenizer_instance,
                }

    @patch("babeltron.app.models.m2m100.get_model_path")
    def test_load_model_standard(self, mock_get_path, mock_transformers, mock_torch):
        mock_get_path.return_value = "/models"

        model = M2M100TranslationModel()
        model._initialized = True

        mock_transformers["model_class"].from_pretrained.return_value = mock_transformers["model_instance"]
        mock_transformers["tokenizer_class"].from_pretrained.return_value = mock_transformers["tokenizer_instance"]

        original_load = model.load

        def patched_load():
            result = original_load()
            model._architecture = ModelArchitecture.CPU_STANDARD
            return result

        model.load = patched_load

        model.load()

        assert model._architecture == ModelArchitecture.CPU_STANDARD

    @patch("babeltron.app.models.m2m100.get_model_path")
    def test_load_model_cpu_compiled(self, mock_get_path, mock_transformers, mock_torch):
        mock_get_path.return_value = "/models"

        model = M2M100TranslationModel()
        model._initialized = True  # Prevent auto-loading

        mock_transformers["model_class"].from_pretrained.return_value = mock_transformers["model_instance"]
        mock_transformers["tokenizer_class"].from_pretrained.return_value = mock_transformers["tokenizer_instance"]

        original_load = model.load

        def patched_load():
            result = original_load()
            model._architecture = ModelArchitecture.CPU_COMPILED
            return result

        model.load = patched_load

        model.load()

        assert model._architecture == ModelArchitecture.CPU_COMPILED

    @patch("babeltron.app.models.m2m100.get_model_path")
    @patch.object(M2M100TranslationModel, "load")  # Patch the load method
    def test_translate(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        model = M2M100TranslationModel()
        model._initialized = True  # Prevent auto-loading

        model._model = mock_transformers["model_instance"]
        model._tokenizer = mock_transformers["tokenizer_instance"]
        model._architecture = ModelArchitecture.CPU_STANDARD

        with patch.object(model, '_translate_cpu', return_value="Bonjour le monde"):
            result = model.translate("Hello world", "en", "fr")
            assert result == "Bonjour le monde"
            model._translate_cpu.assert_called_once_with("Hello world", "en", "fr", None)

    @patch("babeltron.app.models.m2m100.get_model_path")
    @patch.object(M2M100TranslationModel, "load")  # Patch the load method
    def test_translate_model_not_loaded(self, mock_load, mock_get_path):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        model = M2M100TranslationModel()
        model._initialized = True  # Prevent auto-loading

        with pytest.raises(ValueError, match="Model not loaded"):
            model.translate("Hello world", "en", "fr")

    @patch("babeltron.app.models.m2m100.get_model_path")
    @patch.object(M2M100TranslationModel, "load")  # Patch the load method
    def test_get_languages(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        model = M2M100TranslationModel()
        model._initialized = True  # Prevent auto-loading
        model._tokenizer = mock_transformers["tokenizer_instance"]

        languages = model.get_languages()

        assert set(languages) == {"en", "fr", "es", "de"}

    @patch("babeltron.app.models.m2m100.get_model_path")
    def test_get_languages_model_not_loaded(self, mock_get_path):
        mock_get_path.return_value = "/models"

        model = M2M100TranslationModel()
        model._initialized = True  # Prevent auto-loading

        with pytest.raises(ValueError, match="Model not loaded"):
            model.get_languages()

    @patch("babeltron.app.models.m2m100.get_model_path")
    @patch.object(M2M100TranslationModel, "load")  # Patch the load method
    def test_properties(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        if hasattr(M2M100TranslationModel, "_instance"):
            M2M100TranslationModel._instance = None

        model = M2M100TranslationModel()
        model._initialized = True  # Prevent auto-loading
        model._model = mock_transformers["model_instance"]
        model._tokenizer = mock_transformers["tokenizer_instance"]
        model._architecture = ModelArchitecture.CPU_STANDARD

        assert model.model == mock_transformers["model_instance"]
        assert model.tokenizer == mock_transformers["tokenizer_instance"]
        assert model.architecture == ModelArchitecture.CPU_STANDARD
        assert model.is_loaded is True

        model._model = None
        assert model.is_loaded is False

    @patch("babeltron.app.models.m2m100.get_model_path")
    @patch.object(M2M100TranslationModel, "load")  # Patch the load method
    def test_singleton_pattern(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        if hasattr(M2M100TranslationModel, "_instance"):
            M2M100TranslationModel._instance = None

        model1 = M2M100TranslationModel()
        model1._initialized = True  # Prevent auto-loading
        model2 = M2M100TranslationModel()

        assert model1 is model2

        model1._model = mock_transformers["model_instance"]
        model1._tokenizer = mock_transformers["tokenizer_instance"]

        assert model2._model is mock_transformers["model_instance"]
        assert model2._tokenizer is mock_transformers["tokenizer_instance"]


def test_get_translation_model():
    with patch("babeltron.app.models.m2m100.M2M100TranslationModel.load", return_value=(None, None, None)):
        model = get_translation_model()
        assert isinstance(model, M2M100TranslationModel)

        model2 = get_translation_model()
        assert model is model2
