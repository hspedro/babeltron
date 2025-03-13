import os


class ModelType:
    M2M100 = "m2m100"
    NLLB = "nllb"


BABELTRON_MODEL_TYPE = os.getenv("BABELTRON_MODEL_TYPE", ModelType.M2M100)
MODEL_PATH = os.getenv("MODEL_PATH", "./models")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
IN_TEST = os.environ.get("PYTEST_CURRENT_TEST") is not None

# Tracing options
OTLP_MODE = os.environ.get("OTLP_MODE", "otlp-grpc")  # Default to otlp-grpc
OTLP_COLLECTOR_HOST = os.environ.get("OTLP_COLLECTOR_HOST", "otel-collector")
OTLP_COLLECTOR_PORT = os.environ.get("OTLP_COLLECTOR_PORT", "4317")
JAEGER_AGENT_HOST = os.environ.get(
    "JAEGER_AGENT_HOST", "localhost" if IN_TEST else "jaeger"
)
JAEGER_AGENT_PORT = int(os.environ.get("JAEGER_AGENT_PORT", "6831"))
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "babeltron")
