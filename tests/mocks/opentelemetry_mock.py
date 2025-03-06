

class MockSpan:
    def __init__(self, name="mock_span", context=None):
        self.name = name
        self.context = context
        self.attributes = {}
        self.events = []
        self.links = []
        self.status = None
        self.end_time = None
        self.is_recording = True

    def set_attribute(self, key, value):
        self.attributes[key] = value
        return self

    def add_event(self, name, attributes=None, timestamp=None):
        self.events.append((name, attributes, timestamp))
        return self

    def record_exception(self, exception, attributes=None, timestamp=None):
        self.events.append(("exception", exception, attributes, timestamp))
        return self

    def set_status(self, status, description=None):
        self.status = (status, description)
        return self

    def end(self, end_time=None):
        self.end_time = end_time
        self.is_recording = False


class MockTracer:
    def __init__(self, name="mock_tracer"):
        self.name = name
        self.spans = []

    def start_span(self, name, context=None, kind=None, attributes=None, links=None, start_time=None):
        span = MockSpan(name, context)
        self.spans.append(span)
        return span

    def start_as_current_span(self, name, context=None, kind=None, attributes=None, links=None, start_time=None):
        span = self.start_span(name, context, kind, attributes, links, start_time)
        return span


class MockTracerProvider:
    def __init__(self):
        self.tracers = {}

    def get_tracer(self, name, version=None, schema_url=None):
        if name not in self.tracers:
            self.tracers[name] = MockTracer(name)
        return self.tracers[name]


# Create a global tracer provider
_TRACER_PROVIDER = MockTracerProvider()


def get_tracer(name, version=None, schema_url=None):
    return _TRACER_PROVIDER.get_tracer(name, version, schema_url)


def get_current_span():
    return MockSpan()


def set_tracer_provider(provider):
    global _TRACER_PROVIDER
    _TRACER_PROVIDER = provider


def get_tracer_provider():
    return _TRACER_PROVIDER
