import logging
import os

from fastapi import FastAPI
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

# Check if we're in a test environment
IN_TEST = os.environ.get("PYTEST_CURRENT_TEST") is not None

OTLP_MODE = os.environ.get("OTLP_MODE", "otlp-grpc")
OTLP_GRPC_ENDPOINT = os.environ.get("OTLP_GRPC_ENDPOINT", "otel-collector:4317")
OTLP_HTTP_ENDPOINT = os.environ.get(
    "OTLP_HTTP_ENDPOINT", "http://otel-collector:4318/v1/traces"
)


def setup_jaeger(app: FastAPI, log_correlation: bool = True) -> None:
    # Skip setup if we're in a test environment
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
        tracer.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporterGRPC(endpoint=OTLP_GRPC_ENDPOINT, insecure=True)
            )
        )
    elif OTLP_MODE == "otlp-http":
        tracer.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporterHTTP(endpoint=OTLP_HTTP_ENDPOINT))
        )
    else:
        tracer.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporterGRPC(endpoint=OTLP_GRPC_ENDPOINT, insecure=True)
            )
        )

    if log_correlation:
        LoggingInstrumentor().instrument(set_logging_format=True)

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer,
        excluded_urls="/metrics,/healthz,/readyz,/docs,/redoc,/openapi.json",
    )

    logging.info(f"OpenTelemetry tracing enabled with endpoint: {OTLP_GRPC_ENDPOINT}")
