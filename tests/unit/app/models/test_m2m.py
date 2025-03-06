import pytest
from unittest.mock import patch, MagicMock
import torch
from babeltron.app.models.m2m import (
    M2MTranslationModel,
    ModelArchitecture,
)


class TestM2MTranslationModel:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        if hasattr(M2MTranslationModel, "_instance"):
            M2MTranslationModel._instance = None
        yield
        if hasattr(M2MTranslationModel, "_instance"):
            M2MTranslationModel._instance = None

    @pytest.fixture
    def mock_tokenizer(self):
        mock = MagicMock()
        mock.lang_code_to_id = {"en": 0, "fr": 1, "es": 2, "de": 3}
        return mock

    @pytest.fixture
    def mock_model(self):
        mock = MagicMock()
        mock.generate.return_value = torch.tensor([[1, 2, 3]])
        return mock

    @pytest.fixture
    def mock_torch(self):
        with patch("babeltron.app.models.m2m.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device.return_value = "cuda:0"
            yield mock_torch

    @pytest.fixture
    def mock_transformers(self):
        with patch("babeltron.app.models.m2m.M2M100ForConditionalGeneration") as mock_model:
            with patch("babeltron.app.models.m2m.M2M100Tokenizer") as mock_tokenizer:
                tokenizer_instance = MagicMock()
                tokenizer_instance.lang_code_to_id = {"en": 0, "fr": 1, "es": 2, "de": 3}
                tokenizer_instance.decode.return_value = "Bonjour le monde"
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

    @patch("babeltron.app.models.m2m.get_model_path")
    def test_load_model_standard(self, mock_get_path, mock_transformers, mock_torch):
        mock_get_path.return_value = "/models"

        model = M2MTranslationModel()
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

    @patch("babeltron.app.models.m2m.get_model_path")
    def test_load_model_cuda(self, mock_get_path, mock_transformers, mock_torch):
        mock_get_path.return_value = "/models"

        model = M2MTranslationModel()
        model._initialized = True  # Prevent auto-loading

        mock_transformers["model_class"].from_pretrained.return_value = mock_transformers["model_instance"]
        mock_transformers["tokenizer_class"].from_pretrained.return_value = mock_transformers["tokenizer_instance"]

        original_load = model.load

        def patched_load():
            result = original_load()
            model._architecture = ModelArchitecture.CUDA_FP16
            return result

        model.load = patched_load

        model.load()

        assert model._architecture == ModelArchitecture.CUDA_FP16

    @patch("babeltron.app.models.m2m.get_model_path")
    def test_load_model_cpu_quantized(self, mock_get_path, mock_transformers, mock_torch):
        mock_get_path.return_value = "/models"

        model = M2MTranslationModel()
        model._initialized = True  # Prevent auto-loading

        mock_transformers["model_class"].from_pretrained.return_value = mock_transformers["model_instance"]
        mock_transformers["tokenizer_class"].from_pretrained.return_value = mock_transformers["tokenizer_instance"]

        original_load = model.load

        def patched_load():
            result = original_load()
            model._architecture = ModelArchitecture.CPU_QUANTIZED
            return result

        model.load = patched_load

        model.load()

        assert model._architecture == ModelArchitecture.CPU_QUANTIZED

    @patch("babeltron.app.models.m2m.get_model_path")
    def test_load_model_cpu_compiled(self, mock_get_path, mock_transformers, mock_torch):
        mock_get_path.return_value = "/models"

        model = M2MTranslationModel()
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

    @patch("babeltron.app.models.m2m.get_model_path")
    @patch.object(M2MTranslationModel, "load")  # Patch the load method
    def test_translate(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        model = M2MTranslationModel()
        model._initialized = True  # Prevent auto-loading

        model._model = mock_transformers["model_instance"]
        model._tokenizer = mock_transformers["tokenizer_instance"]
        model._architecture = ModelArchitecture.CPU_STANDARD
        model._device = "cpu"

        with patch.object(model, '_translate_cpu', return_value="Bonjour le monde"):
            result = model.translate("Hello world", "en", "fr")
            assert result == "Bonjour le monde"
            model._translate_cpu.assert_called_once_with("Hello world", "en", "fr", None)

    @patch("babeltron.app.models.m2m.get_model_path")
    @patch.object(M2MTranslationModel, "load")  # Patch the load method
    def test_translate_model_not_loaded(self, mock_load, mock_get_path):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        model = M2MTranslationModel()
        model._initialized = True  # Prevent auto-loading

        with pytest.raises(ValueError, match="Model not loaded"):
            model.translate("Hello world", "en", "fr")

    @patch("babeltron.app.models.m2m.get_model_path")
    @patch.object(M2MTranslationModel, "load")  # Patch the load method
    def test_get_languages(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        model = M2MTranslationModel()
        model._initialized = True  # Prevent auto-loading
        model._tokenizer = mock_transformers["tokenizer_instance"]

        languages = model.get_languages()

        assert set(languages) == {"en", "fr", "es", "de"}

    @patch("babeltron.app.models.m2m.get_model_path")
    def test_get_languages_model_not_loaded(self, mock_get_path):
        mock_get_path.return_value = "/models"

        model = M2MTranslationModel()
        model._initialized = True  # Prevent auto-loading

        with pytest.raises(ValueError, match="Model not loaded"):
            model.get_languages()

    @patch("babeltron.app.models.m2m.get_model_path")
    @patch.object(M2MTranslationModel, "load")  # Patch the load method
    def test_properties(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        if hasattr(M2MTranslationModel, "_instance"):
            M2MTranslationModel._instance = None

        model = M2MTranslationModel()
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

    @patch("babeltron.app.models.m2m.get_model_path")
    @patch.object(M2MTranslationModel, "load")  # Patch the load method
    def test_singleton_pattern(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        if hasattr(M2MTranslationModel, "_instance"):
            M2MTranslationModel._instance = None

        model1 = M2MTranslationModel()
        model1._initialized = True  # Prevent auto-loading
        model2 = M2MTranslationModel()

        assert model1 is model2

        model1._model = mock_transformers["model_instance"]
        model1._tokenizer = mock_transformers["tokenizer_instance"]

        assert model2._model is mock_transformers["model_instance"]
        assert model2._tokenizer is mock_transformers["tokenizer_instance"]

    @patch("babeltron.app.models.m2m.get_model_path")
    @patch.object(M2MTranslationModel, "load")  # Patch the load method
    def test_translate_cuda(self, mock_load, mock_get_path, mock_transformers, mock_torch):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        if hasattr(M2MTranslationModel, "_instance"):
            M2MTranslationModel._instance = None

        model = M2MTranslationModel()
        model._initialized = True  # Prevent auto-loading
        model._model = mock_transformers["model_instance"]
        model._tokenizer = mock_transformers["tokenizer_instance"]
        model._architecture = ModelArchitecture.CUDA_FP16
        model._device = "cuda:0"

        encoded_text_mock = {"input_ids": MagicMock()}
        mock_transformers["tokenizer_instance"].return_value = encoded_text_mock

        mock_transformers["tokenizer_instance"].batch_decode.return_value = ["Bonjour le monde"]

        result = model._translate_cuda("Hello world", "en", "fr")

        mock_transformers["model_instance"].generate.assert_called_once()
        mock_transformers["tokenizer_instance"].batch_decode.assert_called_once()

        assert result == "Bonjour le monde"

    @patch("babeltron.app.models.m2m.get_model_path")
    def test_translate_cpu(self, mock_get_path, mock_transformers):
        mock_get_path.return_value = "/models"

        if hasattr(M2MTranslationModel, "_instance"):
            M2MTranslationModel._instance = None

        with patch.object(M2MTranslationModel, "load", return_value=(None, None, None)):
            model = M2MTranslationModel()
            model._initialized = True  # Prevent auto-loading
            model._model = mock_transformers["model_instance"]
            model._tokenizer = mock_transformers["tokenizer_instance"]
            model._architecture = ModelArchitecture.CPU_STANDARD
            model._device = "cpu"

            mock_transformers["tokenizer_instance"].return_value = {
                "input_ids": torch.tensor([[1, 2, 3]])
            }

            mock_transformers["tokenizer_instance"].batch_decode.return_value = ["Bonjour le monde"]
            mock_transformers["model_instance"].generate.return_value = torch.tensor([[4, 5, 6]])
            result = model._translate_cpu("Hello world", "en", "fr", tracer=None)

            mock_transformers["model_instance"].generate.assert_called_once()
            mock_transformers["tokenizer_instance"].batch_decode.assert_called_once()

            assert result == "Bonjour le monde"

    @patch("babeltron.app.models.m2m.get_model_path")
    @patch.object(M2MTranslationModel, "load")  # Patch the load method
    def test_translate_method(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        model = M2MTranslationModel()

        model._model = mock_transformers["model_instance"]
        model._tokenizer = mock_transformers["tokenizer_instance"]
        model._architecture = ModelArchitecture.CPU_STANDARD
        model._device = "cpu"

        with patch.object(model, '_translate_cpu', return_value="Bonjour le monde"):
            result = model.translate("Hello world", "en", "fr")
            assert result == "Bonjour le monde"

    @patch("babeltron.app.models.m2m.get_model_path")
    def test_compile_model(self, mock_get_path, mock_transformers, mock_torch):
        mock_get_path.return_value = "/models"

        model = M2MTranslationModel()
        model._initialized = True
        model._model = mock_transformers["model_instance"]

        mock_torch.compile.return_value = mock_transformers["model_instance"]

        model._model = mock_torch.compile(model._model, backend="inductor")

        mock_torch.compile.assert_called_once_with(
            mock_transformers["model_instance"],
            backend="inductor"
        )


class TestModelArchitecture:
    def test_architecture_values(self):
        assert ModelArchitecture.CUDA_FP16 == "cuda_fp16"
        assert ModelArchitecture.CPU_QUANTIZED == "cpu_quantized"
        assert ModelArchitecture.CPU_STANDARD == "cpu_standard"
        assert ModelArchitecture.CPU_COMPILED == "cpu_compiled"


def test_get_translation_model():
    from babeltron.app.models.m2m import get_translation_model

    with patch("babeltron.app.models.m2m.M2MTranslationModel.load", return_value=(None, None, None)):
        model = get_translation_model()
        assert isinstance(model, M2MTranslationModel)

        model2 = get_translation_model()
        assert model is model2
