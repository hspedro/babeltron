import logging
import os
import time

import torch
from fastapi import APIRouter, HTTPException, status
from opentelemetry import trace
from pydantic import BaseModel, Field
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from babeltron.app.monitoring import track_dynamic_translation_metrics
from babeltron.app.utils import get_model_path

router = APIRouter(tags=["Translation"])

MODEL_COMPRESSION_ENABLED = os.environ.get(
    "MODEL_COMPRESSION_ENABLED", "true"
).lower() in ("true", "1", "yes")

try:
    MODEL_PATH = get_model_path()
    logging.info(f"Loading model from: {MODEL_PATH}")
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_PATH)

    # Apply FP16 compression if enabled and supported
    if MODEL_COMPRESSION_ENABLED and torch.cuda.is_available():
        logging.info("Applying FP16 model compression")
        model = model.half()  # Convert to FP16 precision
        model = model.to("cuda")  # Move to GPU
    elif MODEL_COMPRESSION_ENABLED:
        logging.info("FP16 compression enabled but GPU not available, using CPU")
    else:
        logging.info("Model compression disabled")

    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_PATH)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None
    tokenizer = None


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


@router.post(
    "/translate",
    summary="Translate text between languages",
    response_model=TranslationResponse,
    description="""
    Translates text from one language to another using the M2M-100 model.

    Provide the text to translate, source language code, and target language code.
    Language codes should follow the ISO 639-1 standard (e.g., 'en' for English, 'es' for Spanish).
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

    logging.info(f"Translating text from {request.src_lang} to {request.tgt_lang}")

    if model is None or tokenizer is None:
        current_span.set_attribute("error", "model_not_loaded")
        logging.error("Translation model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation model not loaded. Please check server logs.",
        )

    try:
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("tokenization") as tokenize_span:
            start_time = time.time()
            tokenizer.src_lang = request.src_lang
            encoded_text = tokenizer(request.text, return_tensors="pt")
            tokenize_span.set_attribute(
                "token_count", encoded_text["input_ids"].shape[1]
            )
            tokenize_span.set_attribute(
                "duration_ms", (time.time() - start_time) * 1000
            )

        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            with tracer.start_as_current_span("move_to_gpu") as gpu_span:
                start_time = time.time()
                encoded_text = {k: v.to("cuda") for k, v in encoded_text.items()}
                gpu_span.set_attribute("duration_ms", (time.time() - start_time) * 1000)

        with tracer.start_as_current_span("model_inference") as inference_span:
            start_time = time.time()
            generated_tokens = model.generate(
                **encoded_text,
                forced_bos_token_id=tokenizer.get_lang_id(request.tgt_lang),
            )
            inference_time = time.time() - start_time
            inference_span.set_attribute("inference_time_seconds", inference_time)
            inference_span.set_attribute(
                "output_token_count", generated_tokens.shape[1]
            )
            inference_span.set_attribute("duration_ms", inference_time * 1000)

        with tracer.start_as_current_span("decoding") as decode_span:
            start_time = time.time()
            translation = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]
            decode_span.set_attribute("duration_ms", (time.time() - start_time) * 1000)

        current_span.set_attribute("translation_length", len(translation))

        logging.info(f"Translation completed: {len(translation)} characters")
        return {"translation": translation}
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
    description="Returns a list of supported language codes and their names",
)
async def languages():
    if tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation model not loaded. Please check server logs.",
        )

    # Get language codes from the tokenizer
    lang_codes = tokenizer.lang_code_to_id

    # Create a dictionary of language codes to language names
    # The tokenizer only provides codes, so we need to map them to human-readable names
    # This uses ISO 639-1 codes as keys
    lang_names = [code for code in lang_codes.keys()]

    return lang_names
