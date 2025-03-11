import os
import sys
import pytest
import warnings
from unittest.mock import MagicMock
# Import our mock module
from tests.mocks.opentelemetry import (
    trace as mock_trace,
    MockOTLPSpanExporterGRPC,
    MockOTLPSpanExporterHTTP,
    MockBatchSpanProcessor,
)

# Set environment variables to disable telemetry for tests
os.environ["OTLP_GRPC_ENDPOINT"] = "disabled"
os.environ["OTLP_MODE"] = "disabled"  # Disable actual tracing in tests
os.environ["JAEGER_AGENT_HOST"] = "localhost"  # Use localhost for Jaeger agent


class MockOpenTelemetry:
    def __getattr__(self, name):
        if name == "trace":
            return mock_trace
        return MagicMock()


# Create a mock Jaeger exporter
class MockJaegerExporter:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def export(self, spans):
        # Just return success without actually exporting
        return True

    def shutdown(self):
        pass


@pytest.fixture(autouse=True, scope="session")
def mock_opentelemetry():
    # Replace the entire opentelemetry module with our mock
    sys.modules["opentelemetry"] = MockOpenTelemetry()
    sys.modules["opentelemetry.trace"] = mock_trace

    # Also mock any submodules that might be imported directly
    sys.modules["opentelemetry.sdk"] = MagicMock()
    sys.modules["opentelemetry.sdk.trace"] = MagicMock()
    sys.modules["opentelemetry.instrumentation"] = MagicMock()
    sys.modules["opentelemetry.instrumentation.fastapi"] = MagicMock()
    sys.modules["opentelemetry.instrumentation.logging"] = MagicMock()

    # Mock the exporters
    sys.modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"] = MagicMock()
    sys.modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"].OTLPSpanExporter = MockOTLPSpanExporterGRPC

    sys.modules["opentelemetry.exporter.otlp.proto.http.trace_exporter"] = MagicMock()
    sys.modules["opentelemetry.exporter.otlp.proto.http.trace_exporter"].OTLPSpanExporter = MockOTLPSpanExporterHTTP

    # Mock the Jaeger exporter
    sys.modules["opentelemetry.exporter.jaeger"] = MagicMock()
    sys.modules["opentelemetry.exporter.jaeger.thrift"] = MagicMock()
    sys.modules["opentelemetry.exporter.jaeger.thrift"].JaegerExporter = MockJaegerExporter

    sys.modules["opentelemetry.sdk.trace.export"] = MagicMock()
    sys.modules["opentelemetry.sdk.trace.export"].BatchSpanProcessor = MockBatchSpanProcessor
    sys.modules["opentelemetry.sdk.trace.export"].ConsoleSpanExporter = MagicMock()

    # Yield to allow tests to run
    yield

    # Clean up (optional, as pytest will terminate after tests)
    if "opentelemetry" in sys.modules:
        del sys.modules["opentelemetry"]
    if "opentelemetry.trace" in sys.modules:
        del sys.modules["opentelemetry.trace"]


# Reset any singleton models between tests
@pytest.fixture(autouse=True)
def reset_singletons():
    # Import here to avoid circular imports
    from babeltron.app.models.m2m import M2MTranslationModel

    # Reset the singleton instance
    if hasattr(M2MTranslationModel, "_instance"):
        M2MTranslationModel._instance = None

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
