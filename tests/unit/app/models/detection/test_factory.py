import pytest
from unittest.mock import patch

from babeltron.app.models.detection.factory import get_detection_model, register_model, MODEL_REGISTRY
from babeltron.app.config import ModelType
from babeltron.app.models.detection.lingua import LinguaDetectionModel


def test_register_model():
    # Clear the registry for testing
    original_registry = MODEL_REGISTRY.copy()
    MODEL_REGISTRY.clear()

    try:
        # Define a test model factory
        def test_model_factory():
            return "test_model"

        # Register the test model factory
        register_model("test-model")(test_model_factory)

        # Check if the model factory was registered
        assert "test-model" in MODEL_REGISTRY
        assert MODEL_REGISTRY["test-model"] == test_model_factory
        assert MODEL_REGISTRY["test-model"]() == "test_model"
    finally:
        # Restore the original registry
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(original_registry)


def test_get_detection_model_with_valid_type():
    # Clear the registry for testing
    original_registry = MODEL_REGISTRY.copy()
    MODEL_REGISTRY.clear()

    try:
        # Define a test model factory
        def test_model_factory():
            return "test_model"

        # Register the test model factory
        register_model("test-model")(test_model_factory)

        # Get the model with a valid type
        model = get_detection_model("test-model")
        assert model == "test_model"
    finally:
        # Restore the original registry
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(original_registry)


def test_get_detection_model_with_invalid_type():
    # Clear the registry for testing
    original_registry = MODEL_REGISTRY.copy()
    MODEL_REGISTRY.clear()

    try:
        # Try to get a model with an invalid type
        with pytest.raises(ValueError, match="Unsupported model type"):
            get_detection_model("invalid-model")
    finally:
        # Restore the original registry
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(original_registry)


def test_get_detection_model_with_default_type():
    # Clear the registry for testing
    original_registry = MODEL_REGISTRY.copy()
    MODEL_REGISTRY.clear()

    try:
        # Define a test model factory
        def test_model_factory():
            return "test_model"

        # Register the test model factory
        register_model(ModelType.LINGUA)(test_model_factory)

        # Get the model with the default type
        with patch("babeltron.app.models.detection.factory.DEFAULT_DETECTION_MODEL_TYPE", ModelType.LINGUA):
            model = get_detection_model()
            assert model == "test_model"
    finally:
        # Restore the original registry
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(original_registry)


def test_lingua_model_registration():
    """Test that the Lingua model is properly registered in the factory."""
    # Get the model with the Lingua type
    model = get_detection_model(ModelType.LINGUA)

    # Check that it's an instance of LinguaDetectionModel
    assert isinstance(model, LinguaDetectionModel)
    assert model.architecture == "lingua"
    assert model.is_loaded is True
