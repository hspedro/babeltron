from importlib.metadata import version

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from babeltron.app.monitoring import PrometheusMiddleware, metrics_endpoint
from babeltron.app.tracing import setup_jaeger
from babeltron.app.utils import include_routers

try:
    __version__ = version("babeltron")
except ImportError:
    __version__ = "0.1.0-dev"


app = FastAPI(
    title="Babeltron Translation API",
    description="API for machine translation using NLLB models",
    version="0.1.0",
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

# Set up Jaeger tracing
setup_jaeger(app)

# Include all routers
include_routers(app)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)


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
