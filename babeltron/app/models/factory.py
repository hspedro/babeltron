from typing import Dict, Type

from babeltron.app.models.base import TranslationModelBase
from babeltron.app.models.m2m import M2MTranslationModel


class ModelFactory:
    """Factory for creating translation model instances"""

    _models: Dict[str, Type[TranslationModelBase]] = {
        "m2m100": M2MTranslationModel,
        # Add more models here as they are implemented
        # "nllb": NLLBTranslationModel,
        # "mbart": MBartTranslationModel,
    }

    @classmethod
    def get_model(cls, model_type: str = "m2m100") -> TranslationModelBase:
        """
        Get a translation model instance of the specified type.

        Args:
            model_type: The type of model to create. If None, uses the MODEL_TYPE env var
                       or defaults to "m2m100"

        Returns:
            An instance of the requested translation model
        """
        if model_type not in cls._models:
            raise ValueError(
                f"Unknown model type: {model_type}. Available types: {list(cls._models.keys())}"
            )

        return cls._models[model_type]()
