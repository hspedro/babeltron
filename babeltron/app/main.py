import os
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import AsyncIterator

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

from babeltron.app.monitoring import PrometheusMiddleware, metrics_endpoint
from babeltron.app.utils import include_routers

try:
    __version__ = version("babeltron")
except ImportError:
    __version__ = "0.1.0-dev"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    cache_url = os.environ.get("CACHE_URL", "")

    if cache_url.startswith("in-memory"):
        FastAPICache.init(InMemoryBackend(), prefix="babeltron")
        print("Using in-memory cache")
    elif cache_url.startswith("redis"):
        redis = aioredis.from_url(cache_url)
        FastAPICache.init(RedisBackend(redis), prefix="babeltron")
        print("Using Redis cache")
    else:
        print("No cache_url provided, not using cache")

    yield


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
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

    uvicorn.run(app, host="0.0.0.0", port=8000)
