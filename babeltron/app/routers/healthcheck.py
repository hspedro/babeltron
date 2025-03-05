from typing import Optional

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from babeltron.app.main import __version__
from babeltron.app.routers.translate import model, tokenizer

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: Optional[str] = None


@router.get("/healthcheck", summary="Healthcheck")
@router.get(
    "/healthz",
    summary="Check API health",
    description="Returns the health status of the API and whether the translation model is loaded",
    response_model=HealthResponse,
    tags=["Control"],
)
async def healthcheck():
    return {"status": "ok", "model_loaded": model is not None, "version": __version__}


class ReadinessResponse(BaseModel):
    status: str
    version: Optional[str] = None
    error: Optional[str] = None


@router.get("/readiness", summary="Readiness Probe")
@router.get(
    "/readyz",
    summary="Check API Readiness",
    description="Returns the readiness status of the API. Able to process requests.",
    response_model=ReadinessResponse,
    tags=["Control"],
)
async def readiness():
    try:
        if model is None or tokenizer is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not ready",
                    "error": "Model or tokenizer not loaded",
                    "version": __version__,
                },
            )

        test_sentence = "hello"
        tokenizer.src_lang = "en"
        encoded_text = tokenizer(test_sentence, return_tensors="pt")
        _ = model.generate(**encoded_text)

        return {"status": "ready", "version": __version__}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "error": str(e), "version": __version__},
        )
