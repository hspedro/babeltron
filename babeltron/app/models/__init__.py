from babeltron.app.models.base import TranslationModelBase
from babeltron.app.models.factory import ModelFactory
from babeltron.app.models.m2m import M2MTranslationModel, get_translation_model

# For backward compatibility
get_model = get_translation_model

__all__ = [
    "TranslationModelBase",
    "ModelFactory",
    "M2MTranslationModel",
    "get_translation_model",
    "get_model",
]
