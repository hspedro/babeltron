import logging
import os
from importlib.metadata import version

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from babeltron.app.middlewares.auth import BasicAuthMiddleware
from babeltron.app.monitoring import PrometheusMiddleware, metrics_endpoint
from babeltron.app.tracing import setup_jaeger
from babeltron.app.utils import include_routers

try:
    __version__ = version("babeltron")
except ImportError:
    __version__ = "0.2.0-dev"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [%(trace_id)s %(span_id)s %(resource.service.name)s %(trace_sampled)s] - %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Babeltron Translation API",
        description="""
        A multilingual translation API.

        ## Authentication

        This API uses Basic Authentication. Include an Authorization header with your requests:

        ```
        Authorization: Basic <base64-encoded-credentials>
        ```

        Where `<base64-encoded-credentials>` is the Base64 encoding of `username:password`.
        """,
        version="0.2.0",
        contact={
            "name": "Pedro Soares",
            "url": "https://github.com/hspedro",
            "email": "pedrofigueiredoc@gmail.com",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    if (api_username := os.environ.get("API_USERNAME")) and (
        api_password := os.environ.get("API_PASSWORD")
    ):
        app.add_middleware(
            BasicAuthMiddleware,
            username=api_username,
            password=api_password,
            exclude_paths=[
                "/docs",
                "/redoc",
                "/openapi.json",
                "/healthz",
                "/readyz",
                "/metrics",
            ],
        )

    # Set up Jaeger tracing
    setup_jaeger(app)

    # Include all routers
    include_routers(app)

    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)

    return app


app = create_app()


# Add metrics endpoint
@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(content=metrics_endpoint(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] - %(message)s"
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=log_config)
