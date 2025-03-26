import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from opentelemetry import trace
from pydantic import BaseModel, Field

from babeltron.app.cache.service import CacheService
from babeltron.app.config import BABELTRON_MODEL_TYPE, DETECTION_MODEL_TYPE
from babeltron.app.models.detection.factory import get_detection_model
from babeltron.app.models.translation.factory import get_translation_model
from babeltron.app.monitoring import track_dynamic_translation_metrics

router = APIRouter(tags=["Translation"])

# Get the default translation model
translation_model = get_translation_model(BABELTRON_MODEL_TYPE)

# Get the default detection model
detection_model = get_detection_model(model_type=DETECTION_MODEL_TYPE)

# Initialize the cache service
cache_service = CacheService[Dict[str, Any]]()


class TranslationRequest(BaseModel):
    text: str = Field(
        ..., description="The text to translate", example="Hello, how are you?"
    )
    src_lang: Optional[str] = Field(
        None,
        description="Source language code (ISO 639-1). Use 'auto' or leave empty for automatic detection",
        example="en",
    )
    tgt_lang: str = Field(
        ..., description="Target language code (ISO 639-1)", example="es"
    )
    cache: bool = Field(
        True,
        description="Whether to use and store results in cache. Set to false to bypass cache.",
        example=True,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "src_lang": "en",
                "tgt_lang": "es",
                "cache": True,
            }
        }


class TranslationResponse(BaseModel):
    translation: str = Field(..., description="The translated text")
    model_type: str = Field(..., description="The model type used for translation")
    architecture: str = Field(..., description="The model architecture used")
    detected_lang: Optional[str] = Field(
        None, description="The detected source language (if auto-detected)"
    )
    detection_confidence: Optional[float] = Field(
        None, description="Confidence score of language detection (if auto-detected)"
    )
    cached: bool = Field(
        False, description="Whether the result was retrieved from cache"
    )


@router.post(
    "/translate",
    summary="Translate text between languages",
    response_model=TranslationResponse,
    description="""
    Translates text from one language to another using the loaded model.

    Provide the text to translate, source language code, and target language code.
    Language codes should follow the ISO 639-1 standard (e.g., 'en' for English, 'es' for Spanish).

    For automatic source language detection, set src_lang to "auto" or leave it empty.

    The model used for translation is determined by the BABELTRON_MODEL_TYPE environment variable.

    Set cache=false to bypass the cache service and always perform a fresh translation.
    """,
    response_description="The translated text in the target language",
    status_code=status.HTTP_200_OK,
)
@track_dynamic_translation_metrics()
async def translate(request: TranslationRequest):
    current_span = trace.get_current_span()
    current_span.set_attribute("text_length", len(request.text))
    current_span.set_attribute("tgt_lang", request.tgt_lang)
    current_span.set_attribute("cache_enabled", request.cache)

    current_span.set_attribute("model_type", translation_model.model_type)

    # Check if source language needs to be detected
    detected_lang = None
    detection_confidence = None
    src_lang = request.src_lang

    if src_lang is None or src_lang == "" or src_lang.lower() == "auto":
        # We need to detect the language
        if detection_model is None or not detection_model.is_loaded:
            current_span.set_attribute("error", "detection_model_not_loaded")
            logging.error("Language detection model not loaded")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Language detection model not loaded. Please check server logs.",
            )

        logging.info(
            f"Detecting source language using {detection_model.architecture} model"
        )
        current_span.set_attribute("auto_detection", True)

        try:
            tracer = trace.get_tracer(__name__)
            detection_start_time = time.time()

            # Detect the language
            detected_lang, detection_confidence = detection_model.detect(
                request.text, tracer
            )

            detection_time = time.time() - detection_start_time
            current_span.set_attribute("detection_time_seconds", detection_time)
            current_span.set_attribute(
                "detection_model_architecture", detection_model.architecture
            )
            current_span.set_attribute("detected_lang", detected_lang)
            current_span.set_attribute("detection_confidence", detection_confidence)

            logging.info(
                f"Language detection completed in {detection_time:.2f}s. "
                f"Detected: {detected_lang} with confidence {detection_confidence:.4f}"
            )

            # Use the detected language as source language
            src_lang = detected_lang

        except Exception as e:
            current_span.record_exception(e)
            current_span.set_attribute("error", f"detection_error: {str(e)}")
            logging.error(f"Language detection error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Language detection error: {str(e)}",
            )
    else:
        # Source language was provided
        current_span.set_attribute("src_lang", src_lang)
        current_span.set_attribute("auto_detection", False)

    # Check cache for existing translation only if caching is enabled
    cached_result = None
    if request.cache:
        cached_result = cache_service.get_translation(
            request.text, src_lang, request.tgt_lang
        )
        if cached_result:
            logging.info(f"Cache hit for translation: {src_lang} -> {request.tgt_lang}")
            current_span.set_attribute("cache_hit", True)

            # Add the cached flag to the response
            cached_result["cached"] = True
            return cached_result

    current_span.set_attribute("cache_hit", False)

    if not translation_model.is_loaded:
        current_span.set_attribute("error", "model_not_loaded")
        logging.error("Translation model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation model not loaded. Please check server logs.",
        )

    try:
        start_time = time.time()
        tracer = trace.get_tracer(__name__)

        # Translate the text
        translation = translation_model.translate(
            request.text, src_lang, request.tgt_lang, tracer
        )

        # Record metrics
        translation_time = time.time() - start_time
        current_span.set_attribute("translation_time_seconds", translation_time)
        current_span.set_attribute("translation_length", len(translation))
        current_span.set_attribute("model_architecture", translation_model.architecture)

        logging.info(
            f"Translation completed in {translation_time:.2f}s using {translation_model.architecture}"
        )

        # Prepare the response
        response = {
            "translation": translation,
            "model_type": translation_model.model_type,
            "architecture": translation_model.architecture,
            "detected_lang": detected_lang,
            "detection_confidence": detection_confidence,
            "cached": False,
        }

        # Cache the result only if caching is enabled
        if request.cache:
            cache_service.save_translation(
                request.text, src_lang, request.tgt_lang, response
            )

        return response

    except Exception as e:
        current_span.record_exception(e)
        current_span.set_attribute("error", str(e))
        logging.error(f"Translation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation error: {str(e)}",
        )


@router.get(
    "/languages",
    summary="List supported languages",
    description="Returns a list of supported language codes for the currently loaded model",
)
async def languages():
    model = translation_model

    if not model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation model not loaded. Please check server logs.",
        )

    return {"languages": model.get_languages()}
