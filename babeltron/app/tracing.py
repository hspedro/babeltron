import logging
import os

from fastapi import FastAPI

# Check if we're in a test environment
IN_TEST = os.environ.get("PYTEST_CURRENT_TEST") is not None

# Only import OpenTelemetry if we're not in a test environment
if not IN_TEST:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OTLPSpanExporterGRPC,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPSpanExporterHTTP,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
else:
    # Create dummy classes for test environment
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

OTLP_MODE = os.environ.get("OTLP_MODE", "otlp-grpc")
OTLP_GRPC_ENDPOINT = os.environ.get("OTLP_GRPC_ENDPOINT", "otel-collector:4317")
OTLP_HTTP_ENDPOINT = os.environ.get(
    "OTLP_HTTP_ENDPOINT", "http://otel-collector:4318/v1/traces"
)


def setup_jaeger(app: FastAPI, log_correlation: bool = True) -> None:
    if IN_TEST:
        logging.info("Skipping OpenTelemetry setup in test environment")
        return

    # Check if tracing is disabled
    if OTLP_GRPC_ENDPOINT.lower() == "disabled":
        logging.info("OpenTelemetry tracing is disabled")
        return

    tracer = TracerProvider()
    trace.set_tracer_provider(tracer)

    if OTLP_MODE == "otlp-grpc":
        if not IN_TEST:  # Extra check to be safe
            tracer.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporterGRPC(endpoint=OTLP_GRPC_ENDPOINT, insecure=True)
                )
            )
    elif OTLP_MODE == "otlp-http":
        if not IN_TEST:  # Extra check to be safe
            tracer.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporterHTTP(endpoint=OTLP_HTTP_ENDPOINT))
            )
    else:
        if not IN_TEST:  # Extra check to be safe
            tracer.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporterGRPC(endpoint=OTLP_GRPC_ENDPOINT, insecure=True)
                )
            )

    if log_correlation and not IN_TEST:
        LoggingInstrumentor().instrument(set_logging_format=True)

    if not IN_TEST:
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=tracer,
            excluded_urls="/metrics,/healthz,/readyz,/docs,/redoc,/openapi.json",
        )

    logging.info(f"OpenTelemetry tracing enabled with endpoint: {OTLP_GRPC_ENDPOINT}")
