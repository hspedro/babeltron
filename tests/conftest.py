import os
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the app after setting up the path
from babeltron.app.main import app  # noqa: E402

# Set up environment variables for testing
os.environ["MODEL_PATH"] = "/models"
os.environ["OTLP_MODE"] = "disabled"  # Disable actual tracing in tests
os.environ["JAEGER_AGENT_HOST"] = "localhost"  # Use localhost for Jaeger agent


@pytest.fixture
def client():
    """
    Create a test client for the FastAPI app.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_env_vars():
    """
    Mock environment variables for testing.
    """
    original_environ = os.environ.copy()
    os.environ["MODEL_PATH"] = "/models"
    os.environ["LOG_LEVEL"] = "DEBUG"
    yield
    os.environ.clear()
    os.environ.update(original_environ)


@pytest.fixture
def mock_model():
    """
    Create a mock translation model for testing.
    """
    mock = MagicMock()
    mock.translate.return_value = "Translated text"
    mock.is_loaded = True
    mock.architecture = "cpu_standard"
    return mock


@pytest.fixture
def mock_tokenizer():
    """
    Create a mock tokenizer for testing.
    """
    mock = MagicMock()
    mock.lang_code_to_id = {"en": 0, "fr": 1, "es": 2, "de": 3}
    mock.get_lang_id.side_effect = lambda x: mock.lang_code_to_id.get(x, 0)
    mock.batch_decode.return_value = ["Translated text"]
    return mock


@pytest.fixture
def mock_torch():
    """
    Create a mock torch module for testing.
    """
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.device.return_value = "cpu"
    return mock


@pytest.fixture(autouse=True)
def reset_singletons():
    """
    Reset singleton instances between tests.
    """
    # Import here to avoid circular imports
    try:
        from babeltron.app.models.m2m100 import M2M100TranslationModel
        if hasattr(M2M100TranslationModel, "_instance"):
            M2M100TranslationModel._instance = None
            warnings.warn("Reset M2M100TranslationModel singleton")
    except ImportError:
        warnings.warn("Could not import M2M100TranslationModel")

    try:
        from babeltron.app.models.nllb import NLLBTranslationModel
        if hasattr(NLLBTranslationModel, "_instance"):
            NLLBTranslationModel._instance = None
            warnings.warn("Reset NLLBTranslationModel singleton")
    except ImportError:
        warnings.warn("Could not import NLLBTranslationModel")

    yield


@pytest.fixture
def memory_tracer():
    # This is now a no-op since we're using mock OpenTelemetry
    yield None


@pytest.fixture(scope="session", autouse=True)
def filter_deprecation_warnings():
    # Filter out common deprecation warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

    # Filter Pydantic deprecation warnings
    warnings.filterwarnings("ignore", message="Using extra keyword arguments on `Field` is deprecated", category=UserWarning)
    warnings.filterwarnings("ignore", message="Support for class-based `config` is deprecated", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="Valid config keys have changed in V2", category=UserWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    yield
