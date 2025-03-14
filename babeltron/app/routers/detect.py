import logging
import time

from fastapi import APIRouter, HTTPException, status
from opentelemetry import trace
from pydantic import BaseModel, Field

from babeltron.app.config import DETECTION_MODEL_TYPE
from babeltron.app.models.detection.factory import get_detection_model
from babeltron.app.monitoring import track_dynamic_translation_metrics

router = APIRouter(tags=["Detection"])

detection_model = get_detection_model(model_type=DETECTION_MODEL_TYPE)


class DetectionRequest(BaseModel):
    text: str = Field(
        ...,
        description="The text to detect source language",
        example="Hello, how are you?",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you?",
            }
        }


class DetectionResponse(BaseModel):
    language: str = Field(..., description="The detected language")
    confidence: float = Field(..., description="The confidence score of the detection")


@router.post(
    "/detect",
    summary="Detect language of text",
    response_model=DetectionResponse,
    description="""
    Detects the language of text using the Lingua language detector.

    Lingua is a natural language detection library that's designed to be
    highly accurate even for short text snippets.

    Provide the text to detect source language.
    """,
    response_description="The detected language",
    status_code=status.HTTP_200_OK,
)
@track_dynamic_translation_metrics()
async def detect(request: DetectionRequest):
    current_span = trace.get_current_span()
    current_span.set_attribute("text_length", len(request.text))

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

        return {
            "language": language,
            "confidence": confidence,
        }

    except Exception as e:
        current_span.record_exception(e)
        current_span.set_attribute("error", str(e))
        logging.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection error: {str(e)}",
        )
