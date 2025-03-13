import logging
import os

from fastapi import FastAPI

from babeltron.app.config import (
    IN_TEST,
    JAEGER_AGENT_HOST,
    JAEGER_AGENT_PORT,
    OTLP_COLLECTOR_HOST,
    OTLP_COLLECTOR_PORT,
    OTLP_MODE,
    SERVICE_NAME,
)

if not IN_TEST:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OTLPSpanExporterGRPC,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPSpanExporterHTTP,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.semconv.resource import ResourceAttributes
else:

    class DummyTracerProvider:
        def add_span_processor(self, processor):
            pass

    class DummyTrace:
        @staticmethod
        def set_tracer_provider(provider):
            pass

    trace = DummyTrace()
    TracerProvider = DummyTracerProvider

    class DummyInstrumentor:
        @staticmethod
        def instrument(app, **kwargs):
            pass

        @staticmethod
        def instrument_app(app, **kwargs):
            pass

    FastAPIInstrumentor = DummyInstrumentor
    LoggingInstrumentor = DummyInstrumentor


def setup_jaeger(app: FastAPI, log_correlation: bool = True) -> None:
    if IN_TEST:
        logging.info("Using minimal OpenTelemetry setup for test environment")
        # For tests, we'll set up a minimal tracer that doesn't try to connect to external services
        resource = Resource.create(
            {ResourceAttributes.SERVICE_NAME: f"{SERVICE_NAME}-test"}
        )
        tracer = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer)

        # In tests, we'll use the console exporter which doesn't require external connections
        # This is optional and can be disabled if you don't want any tracing in tests
        if os.environ.get("OTEL_TEST_EXPORT", "false").lower() == "true":
            tracer.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logging.info("Console span exporter enabled for tests")

        # Instrument the app for tests if needed
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=tracer,
            excluded_urls="/metrics,/healthz,/readyz,/docs,/redoc,/openapi.json",
        )
        return

    if OTLP_MODE.lower() == "disabled":
        logging.info("OpenTelemetry tracing is disabled")
        return

    # Create a resource with service name
    resource = Resource.create({ResourceAttributes.SERVICE_NAME: SERVICE_NAME})

    # Create tracer provider with resource
    tracer = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer)

    if OTLP_MODE == "otlp-grpc":
        # Use OTLP gRPC exporter to send to the OpenTelemetry Collector
        endpoint = f"{OTLP_COLLECTOR_HOST}:{OTLP_COLLECTOR_PORT}"
        tracer.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporterGRPC(endpoint=endpoint, insecure=True))
        )
        logging.info(f"OTLP gRPC exporter enabled with endpoint: {endpoint}")
    elif OTLP_MODE == "otlp-http":
        # Use OTLP HTTP exporter to send to the OpenTelemetry Collector
        endpoint = f"{OTLP_COLLECTOR_HOST}:{OTLP_COLLECTOR_PORT}"
        tracer.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporterHTTP(endpoint=endpoint))
        )
        logging.info(f"OTLP HTTP exporter enabled with endpoint: {endpoint}")
    else:
        # Use Jaeger Thrift exporter (deprecated but still functional)
        jaeger_exporter = JaegerExporter(
            agent_host_name=JAEGER_AGENT_HOST,
            agent_port=JAEGER_AGENT_PORT,
        )
        tracer.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        logging.info(
            f"Jaeger Thrift exporter enabled with agent: {JAEGER_AGENT_HOST}:{JAEGER_AGENT_PORT}"
        )
        logging.warning(
            "Note: The Jaeger Thrift exporter is deprecated. Consider migrating to OTLP."
        )

    if log_correlation:
        LoggingInstrumentor().instrument(set_logging_format=True)

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer,
        excluded_urls="/metrics,/healthz,/readyz,/docs,/redoc,/openapi.json",
    )
