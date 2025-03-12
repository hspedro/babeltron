import importlib.metadata
import logging
import os

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse

from babeltron.app.middlewares.auth import BasicAuthMiddleware
from babeltron.app.models.m2m import M2MTranslationModel
from babeltron.app.monitoring import PrometheusMiddleware, metrics_endpoint
from babeltron.app.tracing import setup_jaeger
from babeltron.app.utils import include_routers

# Get version from package metadata
try:
    __version__ = importlib.metadata.version("babeltron")
except importlib.metadata.PackageNotFoundError:
    # If package is not installed, try to get version from pyproject.toml
    try:
        import tomli

        with open("pyproject.toml", "rb") as f:
            pyproject = tomli.load(f)
            __version__ = pyproject["project"]["version"]
    except (FileNotFoundError, KeyError, ImportError):
        __version__ = "dev"


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
        version=__version__,
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
        root_path="/api/v1",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
                "/version",  # Add version endpoint to excluded paths
                "/version-badge",  # Add version badge endpoint to excluded paths
            ],
        )

    # Set up Jaeger tracing
    setup_jaeger(app)

    # Include all routers
    include_routers(app)

    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)

    # Add version endpoint for badge
    @app.get("/version", response_class=PlainTextResponse, include_in_schema=False)
    async def get_version():
        return __version__

    # Add version badge endpoint for Shields.io
    @app.get("/version-badge", response_class=JSONResponse, include_in_schema=False)
    async def get_version_badge():
        return {
            "schemaVersion": 1,
            "label": "version",
            "message": __version__,
            "color": "green",
            "cacheSeconds": 3600,  # Cache for 1 hour
        }

    @app.get("/healthz", include_in_schema=False)
    async def health():
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def ready():
        # Check if model is loaded
        try:
            # Just initialize the model to check if it loads correctly
            M2MTranslationModel()
            return {"status": "ready"}
        except Exception as e:
            logging.error(f"Readiness check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready",
            )

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return PlainTextResponse(content=metrics_endpoint())

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] - %(message)s"
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=log_config)
