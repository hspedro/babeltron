import logging
import time

from fastapi import APIRouter, HTTPException, status
from opentelemetry import trace
from pydantic import BaseModel, Field

from babeltron.app.models.factory import ModelFactory
from babeltron.app.monitoring import track_dynamic_translation_metrics

router = APIRouter(tags=["Translation"])

# Get the translation model using the factory
translation_model = ModelFactory.get_model()


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
    model_type: str = Field(
        None, description="Optional model type to use for translation"
    )

    class Config:
        schema_extra = {
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
    Translates text from one language to another using the specified model.

    Provide the text to translate, source language code, and target language code.
    Language codes should follow the ISO 639-1 standard (e.g., 'en' for English, 'es' for Spanish).

    You can optionally specify a model type to use for translation.
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

    # Get the appropriate model based on the request
    model = translation_model
    if request.model_type:
        try:
            model = ModelFactory.get_model(request.model_type)
            current_span.set_attribute("model_type", request.model_type)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    logging.info(f"Translating text from {request.src_lang} to {request.tgt_lang}")

    if not model.is_loaded:
        current_span.set_attribute("error", "model_not_loaded")
        logging.error("Translation model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation model not loaded. Please check server logs.",
        )

    try:
        tracer = trace.get_tracer(__name__)
        current_span.set_attribute("model_architecture", model.architecture)

        # Create a span for the translation process
        with tracer.start_as_current_span("model_translation") as translation_span:
            translation_span.set_attribute("model_type", request.model_type or "m2m100")
            translation_span.set_attribute("architecture", model.architecture)
            translation_span.set_attribute("src_lang", request.src_lang)
            translation_span.set_attribute("tgt_lang", request.tgt_lang)

            # Call the model's translate method without passing the tracer
            # This keeps the tracing logic in the router, not in the model
            start_time = time.time()
            translation = model.translate(
                request.text, request.src_lang, request.tgt_lang
            )

            # Record the translation time in the span
            translation_time = time.time() - start_time
            translation_span.set_attribute("translation_time_seconds", translation_time)
            translation_span.set_attribute("translation_length", len(translation))

        current_span.set_attribute("translation_length", len(translation))
        logging.info(
            f"Translation completed: {len(translation)} characters in {translation_time:.2f}s"
        )

        return {
            "translation": translation,
            "model_type": request.model_type or "m2m100",
            "architecture": model.architecture,
        }

    except Exception as e:
        current_span.record_exception(e)
        current_span.set_attribute("error", str(e))
        current_span.set_attribute("error_type", type(e).__name__)

        logging.error(f"Error translating text: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during translation: {str(e)}",
        )


@router.get(
    "/languages",
    summary="List supported languages",
    description="Returns a list of supported language codes for the specified model",
)
async def languages(model_type: str = None):
    model = translation_model
    if model_type:
        try:
            model = ModelFactory.get_model(model_type)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

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
