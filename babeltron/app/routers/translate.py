from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from babeltron.app.utils import get_model_path

router = APIRouter(tags=["Translation"])

# Load model and tokenizer
try:
    MODEL_PATH = get_model_path()
    print(f"Loading model from: {MODEL_PATH}")
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
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
async def translate(request: TranslationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation model not loaded. Please check server logs.",
        )

    tokenizer.src_lang = request.src_lang
    encoded_text = tokenizer(request.text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_text, forced_bos_token_id=tokenizer.get_lang_id(request.tgt_lang)
    )
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return {"translation": translation}


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
