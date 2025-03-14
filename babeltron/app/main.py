import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse

from babeltron.app.config import (
    AUTH_EXCLUDE_PATHS,
    AUTH_PASSWORD,
    AUTH_USERNAME,
    LOG_LEVEL,
)
from babeltron.app.middlewares.auth import BasicAuthMiddleware
from babeltron.app.monitoring import PrometheusMiddleware, metrics_endpoint
from babeltron.app.routers import detect, healthcheck, translate
from babeltron.version import __version__

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create the FastAPI app
app = FastAPI(
    title="Babeltron Translation API",
    description="API for translating text between languages using neural machine translation models",
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Add Basic Auth middleware
app.add_middleware(
    BasicAuthMiddleware,
    username=AUTH_USERNAME,
    password=AUTH_PASSWORD,
    exclude_paths=AUTH_EXCLUDE_PATHS,
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# Include routers
app.include_router(translate.router, prefix="/api/v1")
app.include_router(healthcheck.router, prefix="/api/v1")
app.include_router(detect.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/api/docs")


@app.get("/api", include_in_schema=False)
async def api_root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/api/docs")


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(content=metrics_endpoint())


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )
