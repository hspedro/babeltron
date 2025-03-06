import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import warnings

# Add the mock directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mocks"))

mock_span = MagicMock()
mock_span.set_attribute.return_value = mock_span
mock_span.add_event.return_value = mock_span
mock_span.record_exception.return_value = mock_span
mock_span.set_status.return_value = mock_span

mock_tracer = MagicMock()
mock_tracer.start_span.return_value = mock_span
mock_tracer.start_as_current_span.return_value = mock_span

mock_trace = MagicMock()
mock_trace.get_tracer.return_value = mock_tracer
mock_trace.get_current_span.return_value = mock_span


@pytest.fixture(scope="session", autouse=True)
def mock_opentelemetry():
    patches = [
        patch("opentelemetry.trace.get_tracer", return_value=mock_tracer),
        patch("opentelemetry.trace.get_current_span", return_value=mock_span),
        patch("babeltron.app.tracing.setup_jaeger"),
    ]

    # Apply all patches
    for p in patches:
        p.start()

    os.environ["OTLP_GRPC_ENDPOINT"] = "disabled"

    yield

    # Remove patches
    for p in patches:
        p.stop()


@pytest.fixture
def memory_tracer():
    # This is now a no-op since we're using mock OpenTelemetry
    yield None


# Filter out the pkg_resources deprecation warning
@pytest.fixture(scope="session", autouse=True)
def filter_deprecation_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
    yield
