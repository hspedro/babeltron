import pytest
from unittest.mock import patch, MagicMock

from babeltron.app.models.translation.factory import get_translation_model, ModelType


@patch("babeltron.app.models.translation.m2m100.M2M100TranslationModel")
@patch("babeltron.app.models.translation.nllb.NLLBTranslationModel")
def test_get_translation_model_default(mock_nllb_class, mock_m2m100_class):
    """Test that the default model type is M2M100"""
    mock_m2m100_instance = MagicMock()
    mock_m2m100_class.return_value = mock_m2m100_instance

    model = get_translation_model()

    assert model == mock_m2m100_instance
    mock_m2m100_class.assert_called_once()
    mock_nllb_class.assert_not_called()


@patch("babeltron.app.models.translation.m2m100.M2M100TranslationModel")
@patch("babeltron.app.models.translation.nllb.NLLBTranslationModel")
def test_get_translation_model_m2m100(mock_nllb_class, mock_m2m100_class):
    """Test getting an M2M100 model explicitly"""
    mock_m2m100_instance = MagicMock()
    mock_m2m100_class.return_value = mock_m2m100_instance

    model = get_translation_model(ModelType.M2M100)

    assert model == mock_m2m100_instance
    mock_m2m100_class.assert_called_once()
    mock_nllb_class.assert_not_called()


@patch("babeltron.app.models.translation.m2m100.M2M100TranslationModel")
@patch("babeltron.app.models.translation.nllb.NLLBTranslationModel")
def test_get_translation_model_nllb(mock_nllb_class, mock_m2m100_class):
    """Test getting an NLLB model explicitly"""
    mock_nllb_instance = MagicMock()
    mock_nllb_class.return_value = mock_nllb_instance

    model = get_translation_model(ModelType.NLLB)

    assert model == mock_nllb_instance
    mock_nllb_class.assert_called_once()
    mock_m2m100_class.assert_not_called()


def test_get_translation_model_invalid():
    """Test that an invalid model type raises a ValueError"""
    with pytest.raises(ValueError, match="Unsupported model type"):
        get_translation_model("invalid_model_type")
