import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from opentelemetry import trace
from pydantic import BaseModel, Field

from babeltron.app.cache.service import CacheService
from babeltron.app.config import DETECTION_MODEL_TYPE
from babeltron.app.models.detection.factory import get_detection_model
from babeltron.app.monitoring import track_dynamic_translation_metrics

router = APIRouter(tags=["Detection"])

detection_model = get_detection_model(model_type=DETECTION_MODEL_TYPE)

cache_service = CacheService[Dict[str, Any]]()


class DetectionRequest(BaseModel):
    text: str = Field(
        ...,
        description="The text to detect source language",
        example="Hello, how are you?",
    )
    cache: bool = Field(
        True,
        description="Whether to use and store results in cache. Set to false to bypass cache.",
        example=True,
    )

    class Config:
        json_schema_extra = {"example": {"text": "Hello, how are you?", "cache": True}}


class DetectionResponse(BaseModel):
    language: str = Field(..., description="The detected language")
    confidence: float = Field(..., description="The confidence score of the detection")
    cached: bool = Field(
        False, description="Whether the result was retrieved from cache"
    )


@router.post(
    "/detect",
    summary="Detect language of text",
    response_model=DetectionResponse,
    description="""
    Detects the language of text using the Lingua language detector.

    Lingua is a natural language detection library that's designed to be
    highly accurate even for short text snippets.

    Provide the text to detect source language.

    Set cache=false to bypass the cache service and always perform a fresh detection.
    """,
    response_description="The detected language",
    status_code=status.HTTP_200_OK,
)
@track_dynamic_translation_metrics()
async def detect(request: DetectionRequest):
    current_span = trace.get_current_span()
    current_span.set_attribute("text_length", len(request.text))
    current_span.set_attribute("cache_enabled", request.cache)

    # Check cache for existing detection result only if caching is enabled
    cached_result = None
    if request.cache:
        cached_result = cache_service.get_detection(request.text)
        if cached_result:
            logging.info("Cache hit for language detection")
            current_span.set_attribute("cache_hit", True)

            cached_result["cached"] = True
            return cached_result

    current_span.set_attribute("cache_hit", False)

    # Use the pre-loaded model based on model_type
    model = detection_model

    # Check if model is None
    if model is None:
        current_span.set_attribute("error", "model_not_loaded")
        logging.error("Detection model is None")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detection model not loaded. Please check server logs.",
        )

    logging.info(f"Detecting language of text using {model.architecture} model")

    if not model.is_loaded:
        current_span.set_attribute("error", "model_not_loaded")
        logging.error("Detection model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detection model not loaded. Please check server logs.",
        )

    try:
        start_time = time.time()
        tracer = trace.get_tracer(__name__)

        language, confidence = model.detect(request.text, tracer)

        detection_time = time.time() - start_time
        current_span.set_attribute("detection_time_seconds", detection_time)
        current_span.set_attribute("model_architecture", model.architecture)

        logging.info(
            f"Detection completed in {detection_time:.2f}s using {model.architecture}. "
            f"Detected: {language} with confidence {confidence:.4f}"
        )

        # Prepare the response
        response = {
            "language": language,
            "confidence": confidence,
            "cached": False,
        }

        if request.cache:
            cache_service.save_detection(request.text, response)

        return response

    except Exception as e:
        current_span.record_exception(e)
        current_span.set_attribute("error", str(e))
        logging.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection error: {str(e)}",
        )
