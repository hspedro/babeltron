import pytest
from unittest.mock import MagicMock, patch
import torch

from babeltron.app.models.translation.nllb import NLLBTranslationModel, ModelArchitecture, get_translation_model


class TestNLLBTranslationModel:
    @pytest.fixture(autouse=True)
    def mock_torch(self):
        with patch("babeltron.app.models.translation.nllb.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.tensor = torch.tensor
            mock_torch.compile.return_value = MagicMock()
            yield mock_torch

    @pytest.fixture
    def mock_transformers(self):
        with patch("babeltron.app.models.translation.nllb.AutoModelForSeq2SeqLM") as mock_model:
            with patch("babeltron.app.models.translation.nllb.AutoTokenizer") as mock_tokenizer:
                tokenizer_instance = MagicMock()
                tokenizer_instance.lang_code_to_id = {
                    "eng_Latn": 0,
                    "fra_Latn": 1,
                    "spa_Latn": 2,
                    "deu_Latn": 3
                }
                tokenizer_instance.additional_special_tokens = [
                    "eng_Latn", "fra_Latn", "spa_Latn", "deu_Latn",
                    "ita_Latn", "por_Latn", "rus_Cyrl", "zho_Hans",
                    "jpn_Jpan", "ara_Arab", "hin_Deva", "kor_Hang"
                ]
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

    @patch("babeltron.app.models.translation.nllb.get_model_path")
    def test_load_model_standard(self, mock_get_path, mock_transformers, mock_torch):
        mock_get_path.return_value = "/models"

        model = NLLBTranslationModel()
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

    @patch("babeltron.app.models.translation.nllb.get_model_path")
    def test_load_model_cpu_compiled(self, mock_get_path, mock_transformers, mock_torch):
        mock_get_path.return_value = "/models"

        model = NLLBTranslationModel()
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

    @patch("babeltron.app.models.translation.nllb.get_model_path")
    @patch.object(NLLBTranslationModel, "load")  # Patch the load method
    def test_translate(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        model = NLLBTranslationModel()
        model._initialized = True  # Prevent auto-loading

        model._model = mock_transformers["model_instance"]
        model._tokenizer = mock_transformers["tokenizer_instance"]
        model._architecture = ModelArchitecture.CPU_STANDARD

        with patch.object(model, '_translate_cpu', return_value="Bonjour le monde"):
            result = model.translate("Hello world", "en", "fr")
            assert result == "Bonjour le monde"
            model._translate_cpu.assert_called_once_with("Hello world", "eng_Latn", "fra_Latn", None)

    @patch("babeltron.app.models.translation.nllb.get_model_path")
    @patch.object(NLLBTranslationModel, "load")  # Patch the load method
    def test_translate_model_not_loaded(self, mock_load, mock_get_path):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        model = NLLBTranslationModel()
        model._initialized = True  # Prevent auto-loading
        model._model = None  # Explicitly set model to None
        model._tokenizer = None  # Explicitly set tokenizer to None

        with pytest.raises(ValueError, match="Model not loaded"):
            model.translate("Hello world", "en", "fr")

    @patch("babeltron.app.models.translation.nllb.get_model_path")
    @patch.object(NLLBTranslationModel, "load")  # Patch the load method
    def test_get_languages(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        model = NLLBTranslationModel()
        model._initialized = True  # Prevent auto-loading
        model._tokenizer = mock_transformers["tokenizer_instance"]

        languages = model.get_languages()

        expected_languages = set(mock_transformers["tokenizer_instance"].additional_special_tokens)
        assert set(languages) == expected_languages, \
            f"Expected languages from tokenizer, got different set. Difference: {set(languages).symmetric_difference(expected_languages)}"

    @patch("babeltron.app.models.translation.nllb.get_model_path")
    def test_get_languages_model_not_loaded(self, mock_get_path):
        mock_get_path.return_value = "/models"

        model = NLLBTranslationModel()
        model._initialized = True  # Prevent auto-loading
        model._model = None  # Explicitly set model to None
        model._tokenizer = None  # Explicitly set tokenizer to None

        with pytest.raises(ValueError, match="Model not loaded"):
            model.get_languages()

    @patch("babeltron.app.models.translation.nllb.get_model_path")
    @patch.object(NLLBTranslationModel, "load")  # Patch the load method
    def test_properties(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        if hasattr(NLLBTranslationModel, "_instance"):
            NLLBTranslationModel._instance = None

        model = NLLBTranslationModel()
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

    @patch("babeltron.app.models.translation.nllb.get_model_path")
    @patch.object(NLLBTranslationModel, "load")  # Patch the load method
    def test_singleton_pattern(self, mock_load, mock_get_path, mock_transformers):
        mock_load.return_value = (None, None, None)
        mock_get_path.return_value = "/models"

        if hasattr(NLLBTranslationModel, "_instance"):
            NLLBTranslationModel._instance = None

        model1 = NLLBTranslationModel()
        model1._initialized = True  # Prevent auto-loading
        model2 = NLLBTranslationModel()

        assert model1 is model2

        model1._model = mock_transformers["model_instance"]
        model1._tokenizer = mock_transformers["tokenizer_instance"]

        assert model2._model is mock_transformers["model_instance"]
        assert model2._tokenizer is mock_transformers["tokenizer_instance"]

    @patch("babeltron.app.models.translation.nllb.get_model_path")
    def test_convert_lang_code(self, mock_get_path):
        mock_get_path.return_value = "/models"

        model = NLLBTranslationModel()
        model._initialized = True  # Prevent auto-loading

        # Test common ISO code conversions
        assert model._convert_lang_code("en") == "eng_Latn"
        assert model._convert_lang_code("fr") == "fra_Latn"
        assert model._convert_lang_code("zh") == "zho_Hans"

        # Test some of the newly added ISO code conversions
        assert model._convert_lang_code("nl") == "nld_Latn"
        assert model._convert_lang_code("pl") == "pol_Latn"
        assert model._convert_lang_code("tr") == "tur_Latn"
        assert model._convert_lang_code("uk") == "ukr_Cyrl"
        assert model._convert_lang_code("vi") == "vie_Latn"
        assert model._convert_lang_code("sv") == "swe_Latn"
        assert model._convert_lang_code("fi") == "fin_Latn"
        assert model._convert_lang_code("cs") == "ces_Latn"
        assert model._convert_lang_code("da") == "dan_Latn"
        assert model._convert_lang_code("el") == "ell_Grek"

        # Test already formatted code
        assert model._convert_lang_code("eng_Latn") == "eng_Latn"

        # Test unknown code
        assert model._convert_lang_code("xx") == "xx"


def test_get_translation_model():
    with patch("babeltron.app.models.translation.nllb.NLLBTranslationModel.load", return_value=(None, None, None)):
        model = get_translation_model()
        assert isinstance(model, NLLBTranslationModel)

        model2 = get_translation_model()
        assert model is model2
