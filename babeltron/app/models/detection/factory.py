import logging
import os
from typing import Callable, Dict

from babeltron.app.config import ModelType
from babeltron.app.models.detection.base import DetectionModelBase

# Default model type from environment variable, fallback to Lingua
DEFAULT_DETECTION_MODEL_TYPE = os.environ.get(
    "DEFAULT_DETECTION_MODEL_TYPE", ModelType.LINGUA
)

# Registry of model types to their factory functions
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(model_type: str):
    """
    Decorator to register a model factory function in the MODEL_REGISTRY.

    Args:
        model_type: The type of model to register

    Returns:
        Decorator function that registers the model factory
    """

    def decorator(factory_func: Callable):
        MODEL_REGISTRY[model_type] = factory_func
        return factory_func

    return decorator


def get_detection_model(model_type: str = None) -> DetectionModelBase:
    """
    Factory function to get the appropriate detection model based on the model type.

    Args:
        model_type: The type of model to use, defaults to DEFAULT_DETECTION_MODEL_TYPE

    Returns:
        An instance of the appropriate detection model

    Raises:
        ValueError: If the model type is not supported
    """
    if model_type is None:
        model_type = DEFAULT_DETECTION_MODEL_TYPE

    if model_type not in MODEL_REGISTRY:
        supported_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types: {supported_models}"
        )

    logging.info(f"Creating detection model of type: {model_type}")
    return MODEL_REGISTRY[model_type]()


# Import models at the end to avoid circular imports
from babeltron.app.models.detection.lingua import get_lingua_model  # noqa

# Register the Lingua model
register_model(ModelType.LINGUA)(get_lingua_model)
