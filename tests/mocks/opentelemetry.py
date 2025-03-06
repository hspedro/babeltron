"""
Mock implementation of OpenTelemetry for testing
"""
from unittest.mock import MagicMock

# Create a mock span
class MockSpan:
    def __init__(self):
        self.attributes = {}
        self.events = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exception):
        self.events.append(("exception", exception))

    def get_span_context(self):
        return MagicMock()

# Create a mock tracer
class MockTracer:
    def __init__(self):
        self.current_span = MockSpan()

    def start_as_current_span(self, name, **kwargs):
        class ContextManager:
            def __init__(self, span):
                self.span = span

            def __enter__(self):
                return self.span

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Important: Don't swallow exceptions
                return False

        return ContextManager(MockSpan())

    def get_current_span(self):
        return self.current_span

    def start_span(self, name, **kwargs):
        return MockSpan()

# Create a mock trace module
class MockTrace:
    def __init__(self):
        self.SpanKind = MagicMock()
        self.SpanKind.INTERNAL = 0
        self.SpanKind.SERVER = 1
        self.SpanKind.CLIENT = 2

    def get_tracer(self, name, **kwargs):
        return MockTracer()

    def get_current_span(self):
        return MockSpan()

    def set_span_in_context(self, span, context=None):
        return {}

# Create a mock exporter
class MockExporter:
    def __init__(self, *args, **kwargs):
        pass

    def export(self, spans):
        return 0  # Success

    def shutdown(self):
        pass

# Create instances to be imported
trace = MockTrace()

# Create mock exporters
class MockOTLPSpanExporterGRPC:
    def __init__(self, *args, **kwargs):
        pass

class MockOTLPSpanExporterHTTP:
    def __init__(self, *args, **kwargs):
        pass

class MockBatchSpanProcessor:
    def __init__(self, *args, **kwargs):
        pass

    def on_end(self, span):
        pass

    def shutdown(self):
        pass
