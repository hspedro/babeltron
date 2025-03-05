from importlib.metadata import version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
        "name": "Your Name",
        "url": "https://your-website.com",
        "email": "your-email@example.com",
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

# Include all routers
include_routers(app)

# This allows running the app directly with uvicorn when this file is executed
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
