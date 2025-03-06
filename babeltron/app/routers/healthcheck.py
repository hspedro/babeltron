from typing import Optional

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from babeltron.app.main import __version__
from babeltron.app.models.factory import ModelFactory

router = APIRouter()

# Get the default translation model
translation_model = ModelFactory.get_model()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_architecture: Optional[str] = None
    version: Optional[str] = None


@router.get(
    "/healthz",
    summary="Check API health",
    description="Returns the health status of the API and whether the translation model is loaded",
    response_model=HealthResponse,
    tags=["Control"],
)
async def healthcheck():
    return {
        "status": "ok",
        "model_loaded": translation_model.is_loaded,
        "model_architecture": translation_model.architecture
        if translation_model.is_loaded
        else None,
        "version": __version__,
    }


class ReadinessResponse(BaseModel):
    status: str
    version: Optional[str] = None
    error: Optional[str] = None
    model_architecture: Optional[str] = None


@router.get(
    "/readyz",
    summary="Check API Readiness",
    description="Returns the readiness status of the API. Able to process requests.",
    response_model=ReadinessResponse,
    tags=["Control"],
)
async def readiness():
    try:
        if not translation_model.is_loaded:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not ready",
                    "error": "Model not loaded",
                    "version": __version__,
                },
            )

        # Test a simple translation to verify the model is working
        test_sentence = "hello"
        try:
            _ = translation_model.translate(test_sentence, "en", "fr")
            return {
                "status": "ready",
                "version": __version__,
                "model_architecture": translation_model.architecture,
            }
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not ready",
                    "error": f"Model test failed: {str(e)}",
                    "version": __version__,
                },
            )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "error": str(e), "version": __version__},
        )
