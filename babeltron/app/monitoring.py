import time

from fastapi import Request
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

# Create a registry
registry = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total count of HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        25.0,
        50.0,
    ),
    registry=registry,
)

ERROR_COUNT = Counter(
    "http_request_errors_total",
    "Total count of HTTP request errors",
    ["method", "endpoint", "exception_type"],
    registry=registry,
)

TRANSLATION_COUNT = Counter(
    "translation_requests_total",
    "Total count of translation requests",
    ["src_lang", "tgt_lang", "detection_used"],
    registry=registry,
)

TRANSLATION_LATENCY = Histogram(
    "translation_duration_seconds",
    "Translation processing time in seconds",
    ["src_lang", "tgt_lang", "detection_used"],
    buckets=(
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4,
        4.5,
        5.0,
        7.5,
        10.0,
        15.0,
        20.0,
        30.0,
        60.0,
    ),
    registry=registry,
)

CACHE_HIT_COUNT = Counter(
    "cache_hits_total", "Total count of cache hits", ["endpoint"], registry=registry
)

CACHE_MISS_COUNT = Counter(
    "cache_misses_total", "Total count of cache misses", ["endpoint"], registry=registry
)

MODEL_LOAD_TIME = Histogram(
    "model_load_time_seconds",
    "Time taken to load the model",
    ["model_size"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    registry=registry,
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app=None):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Get the route path for the request
        route = request.url.path
        method = request.method

        try:
            response = await call_next(request)

            # Record request count and latency
            REQUEST_COUNT.labels(
                method=method, endpoint=route, status_code=response.status_code
            ).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=route).observe(
                time.time() - start_time
            )

            # Record error if status code is 4xx or 5xx
            if 400 <= response.status_code < 600:
                ERROR_COUNT.labels(
                    method=method,
                    endpoint=route,
                    exception_type=f"HTTP{response.status_code}",
                ).inc()

            return response

        except Exception as e:
            # Record exception
            ERROR_COUNT.labels(
                method=method, endpoint=route, exception_type=type(e).__name__
            ).inc()
            raise


def track_dynamic_translation_metrics():
    def decorator(func):
        from functools import wraps

        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not request and args:
                request = args[0]

            start_time = time.time()

            # Check if the request has src_lang and tgt_lang attributes (for translation)
            # or if it's a detection request
            src_lang = getattr(request, "src_lang", "unknown")
            tgt_lang = getattr(request, "tgt_lang", "unknown")

            # Determine if language detection will be used
            detection_used = "false"
            if hasattr(request, "src_lang"):
                if (
                    request.src_lang is None
                    or request.src_lang == ""
                    or (
                        isinstance(request.src_lang, str)
                        and request.src_lang.lower() == "auto"
                    )
                ):
                    detection_used = "true"

            # For detection requests, use special labels
            if hasattr(request, "text") and not hasattr(request, "src_lang"):
                src_lang = "detect"
                tgt_lang = "detect"
                detection_used = "true"  # Detection-only endpoint

            TRANSLATION_COUNT.labels(
                src_lang=src_lang, tgt_lang=tgt_lang, detection_used=detection_used
            ).inc()

            result = await func(*args, **kwargs)

            TRANSLATION_LATENCY.labels(
                src_lang=src_lang, tgt_lang=tgt_lang, detection_used=detection_used
            ).observe(time.time() - start_time)

            return result

        return wrapper

    return decorator


def metrics_endpoint():
    """Generate latest metrics in Prometheus format"""
    return generate_latest(registry)
