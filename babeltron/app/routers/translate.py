import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from opentelemetry import trace
from pydantic import BaseModel, Field

from babeltron.app.config import BABELTRON_MODEL_TYPE
from babeltron.app.models.factory import get_translation_model
from babeltron.app.monitoring import track_dynamic_translation_metrics

router = APIRouter(tags=["Translation"])

# Get the default translation model
translation_model = get_translation_model(BABELTRON_MODEL_TYPE)


class TranslationRequest(BaseModel):
    text: str = Field(
        ..., description="The text to translate", example="Hello, how are you?"
    )
    src_lang: str = Field(
        ..., description="Source language code (ISO 639-1)", example="en"
    )
    tgt_lang: str = Field(
        ..., description="Target language code (ISO 639-1)", example="es"
    )
    model_type: Optional[str] = Field(
        None,
        description="Model type to use for translation (m2m100 or nllb)",
        example="m2m100",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "src_lang": "en",
                "tgt_lang": "es",
            }
        }


class TranslationResponse(BaseModel):
    translation: str = Field(..., description="The translated text")
    model_type: str = Field(..., description="The model type used for translation")
    architecture: str = Field(..., description="The model architecture used")


@router.post(
    "/translate",
    summary="Translate text between languages",
    response_model=TranslationResponse,
    description="""
    Translates text from one language to another using the loaded model.

    Provide the text to translate, source language code, and target language code.
    Language codes should follow the ISO 639-1 standard (e.g., 'en' for English, 'es' for Spanish).

    The model used for translation is determined by the BABELTRON_MODEL_TYPE environment variable.
    """,
    response_description="The translated text in the target language",
    status_code=status.HTTP_200_OK,
)
@track_dynamic_translation_metrics()
async def translate(request: TranslationRequest):
    current_span = trace.get_current_span()
    current_span.set_attribute("src_lang", request.src_lang)
    current_span.set_attribute("tgt_lang", request.tgt_lang)
    current_span.set_attribute("text_length", len(request.text))

    # Use the model_type from the request if provided, otherwise use the default
    model_type = request.model_type or BABELTRON_MODEL_TYPE
    current_span.set_attribute("model_type", model_type)

    # Use the pre-loaded model based on model_type
    model = translation_model
    if request.model_type and request.model_type != BABELTRON_MODEL_TYPE:
        model = get_translation_model(request.model_type)

    logging.info(
        f"Translating text from {request.src_lang} to {request.tgt_lang} using {model_type} model"
    )

    if not model.is_loaded:
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
        translation = model.translate(
            request.text, request.src_lang, request.tgt_lang, tracer
        )

        # Record metrics
        translation_time = time.time() - start_time
        current_span.set_attribute("translation_time_seconds", translation_time)
        current_span.set_attribute("translation_length", len(translation))
        current_span.set_attribute("model_architecture", model.architecture)

        logging.info(
            f"Translation completed in {translation_time:.2f}s using {model.architecture}"
        )

        return {
            "translation": translation,
            "model_type": model_type,
            "architecture": model.architecture,
        }

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

    try:
        # Get languages from the model
        lang_names = model.get_languages()
        return lang_names

    except Exception as e:
        logging.error(f"Error getting languages: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving languages: {str(e)}",
        )
