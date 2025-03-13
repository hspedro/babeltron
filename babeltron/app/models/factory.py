import logging
from typing import Dict

from babeltron.app.config import ModelType
from babeltron.app.models.base import TranslationModelBase
from babeltron.app.models.m2m100 import get_translation_model as get_m2m100_model
from babeltron.app.models.nllb import get_translation_model as get_nllb_model

# Registry of model types to their factory functions
MODEL_REGISTRY: Dict[str, callable] = {
    ModelType.M2M100: get_m2m100_model,
    ModelType.NLLB: get_nllb_model,
}


def get_translation_model(model_type: str = ModelType.M2M100) -> TranslationModelBase:
    """
    Factory function to get the appropriate translation model based on the model type.

    Args:
        model_type: The type of model to use (m2m100 or nllb)

    Returns:
        An instance of the appropriate translation model

    Raises:
        ValueError: If the model type is not supported
    """
    if model_type not in MODEL_REGISTRY:
        supported_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types: {supported_models}"
        )

    logging.info(f"Creating translation model of type: {model_type}")
    return MODEL_REGISTRY[model_type]()
