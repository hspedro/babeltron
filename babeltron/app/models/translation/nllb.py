import logging
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from babeltron.app.config import MODEL_PATH
from babeltron.app.models.translation.base import TranslationModelBase


def get_model_path() -> str:
    """
    Get the path to the NLLB model directory.

    This function looks for model directories in the following order:
    1. The MODEL_PATH environment variable
    2. Any directory matching nllb-* in /models, project_root/models, or ./models
    3. Any directory with a config.json file in /models, project_root/models, or ./models

    Returns:
        str: Path to the model directory
    """
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent

    possible_paths = [
        Path(MODEL_PATH),
        project_root / MODEL_PATH,
        Path("./models"),
    ]

    # First, look for NLLB model directories
    for base_path in possible_paths:
        if not base_path.exists():
            continue

        # Look for directories matching the pattern nllb-*
        nllb_dirs = list(base_path.glob("nllb-*"))
        if nllb_dirs:
            # Use the first matching directory
            logging.info(f"Found NLLB model directory: {nllb_dirs[0]}")
            return str(nllb_dirs[0])

        # For backward compatibility, look for directories matching nllb_*
        legacy_dirs = list(base_path.glob("nllb_*"))
        if legacy_dirs:
            logging.info(f"Found legacy NLLB model directory: {legacy_dirs[0]}")
            return str(legacy_dirs[0])

    # Fallback: look for any directory with a config.json file
    for path in possible_paths:
        if path.exists() and any(path.glob("**/config.json")):
            logging.info(f"Found model directory with config.json: {path}")
            return str(path)

    logging.warning("No model directory found, using default: ./models")
    return "./models"


class ModelArchitecture:
    CUDA_FP16 = "cuda_fp16"
    MPS_FP16 = "mps_fp16"  # Metal Performance Shaders for Apple Silicon
    ROCM_FP16 = "rocm_fp16"  # AMD GPUs with ROCm
    CPU_QUANTIZED = "cpu_quantized"
    CPU_COMPILED = "cpu_compiled"
    CPU_STANDARD = "cpu_standard"


class NLLBTranslationModel(TranslationModelBase):
    """NLLB translation model implementation with optimizations based on architecture"""

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLLBTranslationModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            self._model = None
            self._tokenizer = None
            self._architecture = None
            self._model_path = get_model_path()
            self._initialized = True
            self.load()

    def load(self) -> Tuple[Any, Any, str]:
        """Load and optimize the model based on available hardware"""
        try:
            logging.info(f"Loading model from: {self._model_path}")

            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)

            # Check for available hardware and optimize accordingly
            if torch.cuda.is_available():
                # Check if we're using ROCm (AMD) or CUDA (NVIDIA)
                if hasattr(torch.version, "hip") and torch.version.hip is not None:
                    self._optimize_for_rocm()
                else:
                    self._optimize_for_cuda()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                self._optimize_for_mps()
            else:
                self._optimize_for_cpu()

            logging.info(
                f"Model loaded successfully with architecture: {self._architecture}"
            )

            return self._model, self._tokenizer, self._architecture

        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            self._model = None
            self._tokenizer = None
            self._architecture = None
            return None, None, None

    def _optimize_for_cuda(self) -> None:
        """Optimize model for CUDA execution"""
        self._model.to("cuda")
        self._model.half()  # Use FP16 for faster inference
        self._architecture = ModelArchitecture.CUDA_FP16
        logging.info("Model optimized for CUDA with FP16")

    def _optimize_for_mps(self) -> None:
        """Optimize model for MPS (Metal Performance Shaders) execution on Apple Silicon"""
        try:
            self._model.to("mps")
            self._model.half()  # Use FP16 for faster inference
            self._architecture = ModelArchitecture.MPS_FP16
            logging.info("Model optimized for MPS (Apple Silicon) with FP16")
        except Exception as e:
            logging.warning(f"MPS optimization failed: {e}, falling back to CPU")
            self._optimize_for_cpu()

    def _optimize_for_rocm(self) -> None:
        """Optimize model for ROCm (AMD GPU) execution"""
        try:
            self._model.to("cuda")  # ROCm uses the same device name as CUDA
            self._model.half()  # Use FP16 for faster inference
            self._architecture = ModelArchitecture.ROCM_FP16
            logging.info("Model optimized for ROCm (AMD GPU) with FP16")
        except Exception as e:
            logging.warning(f"ROCm optimization failed: {e}, falling back to CPU")
            self._optimize_for_cpu()

    def _optimize_for_cpu(self) -> None:
        """Optimize model for CPU execution"""
        if self._try_quantization():
            self._architecture = ModelArchitecture.CPU_QUANTIZED
            logging.info("Model quantized for CPU execution")
        elif self._try_compilation():
            self._architecture = ModelArchitecture.CPU_COMPILED
            logging.info("Model compiled for CPU execution")
        else:
            self._architecture = ModelArchitecture.CPU_STANDARD
            logging.info("Using standard CPU execution")

    def _try_quantization(self) -> bool:
        """Try to quantize the model for CPU execution"""
        try:
            self._model.half()  # Quantize to FP16
            # Test if the model works with quantization
            dummy_input = self._tokenizer("Hello world", return_tensors="pt")
            self._model.generate(**dummy_input, max_length=20)
            return True
        except Exception as e:
            logging.warning(f"Quantization failed: {e}")
            # Revert to full precision
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path)
            return False

    def _try_compilation(self) -> bool:
        """Try to compile the model for CPU execution"""
        try:
            # Check if torch.compile is available (requires PyTorch 2.0+)
            if hasattr(torch, "compile"):
                self._model = torch.compile(self._model, backend="inductor")
                # Test if the compiled model works
                dummy_input = self._tokenizer("Hello world", return_tensors="pt")
                self._model.generate(**dummy_input, max_length=20)
                return True
            return False
        except Exception as e:
            logging.warning(f"Compilation failed: {e}")
            # Revert to standard model
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path)
            return False

    def translate(self, text: str, src_lang: str, tgt_lang: str, tracer=None) -> str:
        """
        Translate text using the appropriate method for the current architecture

        Args:
            text: The text to translate
            src_lang: Source language code (ISO 639-1)
            tgt_lang: Target language code (ISO 639-1)
            tracer: Optional OpenTelemetry tracer for spans

        Returns:
            The translated text
        """
        if self._model is None or self._tokenizer is None:
            raise ValueError("Model not loaded. Please check logs for errors.")

        # Convert ISO 639-1 codes to NLLB format (e.g., "en" -> "eng_Latn")
        src_lang_nllb = self._convert_lang_code(src_lang)
        tgt_lang_nllb = self._convert_lang_code(tgt_lang)

        # Choose the appropriate translation method based on architecture
        if self._architecture == ModelArchitecture.CUDA_FP16:
            return self._translate_gpu(
                text, src_lang_nllb, tgt_lang_nllb, tracer, "cuda"
            )
        elif self._architecture == ModelArchitecture.ROCM_FP16:
            return self._translate_gpu(
                text, src_lang_nllb, tgt_lang_nllb, tracer, "cuda"
            )  # ROCm uses "cuda" as device name
        elif self._architecture == ModelArchitecture.MPS_FP16:
            return self._translate_gpu(
                text, src_lang_nllb, tgt_lang_nllb, tracer, "mps"
            )
        else:
            return self._translate_cpu(text, src_lang_nllb, tgt_lang_nllb, tracer)

    def _convert_lang_code(self, lang_code: str) -> str:
        """
        Convert ISO language code to NLLB language code if needed.
        NLLB uses codes like 'eng_Latn', 'fra_Latn', etc.
        """
        # Mapping for common ISO codes to NLLB codes
        iso_to_nllb = {
            "en": "eng_Latn",
            "fr": "fra_Latn",
            "es": "spa_Latn",
            "de": "deu_Latn",
            "zh": "zho_Hans",
            "ar": "ara_Arab",
            "ru": "rus_Cyrl",
            "pt": "por_Latn",
            "it": "ita_Latn",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "hi": "hin_Deva",
            "nl": "nld_Latn",
            "pl": "pol_Latn",
            "tr": "tur_Latn",
            "uk": "ukr_Cyrl",
            "vi": "vie_Latn",
            "sv": "swe_Latn",
            "fi": "fin_Latn",
            "cs": "ces_Latn",
            "da": "dan_Latn",
            "el": "ell_Grek",
            "hu": "hun_Latn",
            "no": "nob_Latn",
            "ro": "ron_Latn",
            "th": "tha_Thai",
            "id": "ind_Latn",
            "he": "heb_Hebr",
            "fa": "pes_Arab",
            "bg": "bul_Cyrl",
            "ca": "cat_Latn",
            "hr": "hrv_Latn",
            "sk": "slk_Latn",
            "lt": "lit_Latn",
            "lv": "lvs_Latn",
            "et": "est_Latn",
            "sr": "srp_Cyrl",
            "sl": "slv_Latn",
            "bn": "ben_Beng",
            "ms": "zsm_Latn",
            "af": "afr_Latn",
            "sq": "als_Latn",
            "am": "amh_Ethi",
            "hy": "hye_Armn",
            "az": "azj_Latn",
            "eu": "eus_Latn",
            "be": "bel_Cyrl",
            "bs": "bos_Latn",
            "my": "mya_Mymr",
            "cy": "cym_Latn",
            "ka": "kat_Geor",
            "kk": "kaz_Cyrl",
            "km": "khm_Khmr",
            "ky": "kir_Cyrl",
            "lo": "lao_Laoo",
            "mk": "mkd_Cyrl",
            "ml": "mal_Mlym",
            "mr": "mar_Deva",
            "mn": "khk_Cyrl",
            "ne": "npi_Deva",
            "pa": "pan_Guru",
            "si": "sin_Sinh",
            "ta": "tam_Taml",
            "te": "tel_Telu",
            "ur": "urd_Arab",
            "uz": "uzn_Latn",
            "zu": "zul_Latn",
            "sw": "swh_Latn",
        }

        # If the code is already in NLLB format, return it as is
        if "_" in lang_code:
            return lang_code

        # If we have a mapping, use it
        if lang_code in iso_to_nllb:
            return iso_to_nllb[lang_code]

        # If no mapping is found, return the original code
        # This might cause errors, but it's better than silently failing
        logging.warning(f"No NLLB mapping found for language code: {lang_code}")
        return lang_code

    def _translate_gpu(
        self, text: str, src_lang: str, tgt_lang: str, tracer=None, device: str = "cuda"
    ) -> str:
        """Translate using GPU model"""
        start_time = time.time()

        # Get a more descriptive name for the architecture
        if self._architecture == ModelArchitecture.CUDA_FP16:
            arch_name = "NVIDIA CUDA"
        elif self._architecture == ModelArchitecture.ROCM_FP16:
            arch_name = "AMD ROCm"
        elif self._architecture == ModelArchitecture.MPS_FP16:
            arch_name = "Apple MPS"
        else:
            arch_name = device.upper()

        tokenize_span = inference_span = decode_span = None

        try:
            if tracer:
                with tracer.start_as_current_span("tokenization") as span:
                    tokenize_span = span
                    tokenize_start = time.time()
                    encoded_text = self._tokenizer(text, return_tensors="pt")
                    tokenize_span.set_attribute(
                        "token_count", encoded_text["input_ids"].shape[1]
                    )
                    tokenize_span.set_attribute(
                        "duration_ms", (time.time() - tokenize_start) * 1000
                    )
            else:
                encoded_text = self._tokenizer(text, return_tensors="pt")

            encoded_text = {k: v.to(device) for k, v in encoded_text.items()}

            if tracer:
                with tracer.start_as_current_span("model_inference") as span:
                    inference_span = span
                    inference_start = time.time()
                    generated_tokens = self._model.generate(
                        **encoded_text,
                        forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(
                            tgt_lang
                        ),
                    )
                    inference_time = time.time() - inference_start
                    inference_span.set_attribute(
                        "inference_time_seconds", inference_time
                    )
                    inference_span.set_attribute(
                        "output_token_count", generated_tokens.shape[1]
                    )
                    inference_span.set_attribute("duration_ms", inference_time * 1000)
                    inference_span.set_attribute("architecture", self._architecture)
            else:
                generated_tokens = self._model.generate(
                    **encoded_text,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_lang),
                )

            if tracer:
                with tracer.start_as_current_span("decoding") as span:
                    decode_span = span
                    decode_start = time.time()
                    translation = self._tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )[0]
                    decode_span.set_attribute(
                        "duration_ms", (time.time() - decode_start) * 1000
                    )
            else:
                translation = self._tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]

            logging.info(
                f"{arch_name} translation completed in {time.time() - start_time:.2f}s"
            )
            return translation

        except Exception as e:
            logging.error(f"Error in {arch_name} translation: {e}", exc_info=True)
            raise

    def _translate_cpu(
        self, text: str, src_lang: str, tgt_lang: str, tracer=None
    ) -> str:
        """Translate using CPU model (standard, quantized, or compiled)"""
        start_time = time.time()

        try:
            # Set source language for tokenizer (if needed)
            # NLLB tokenizer doesn't use src_lang parameter directly in the call
            tokenize_start = time.time()
            encoded_text = self._tokenizer(text, return_tensors="pt")
            tokenize_time = time.time() - tokenize_start

            if tracer:
                with tracer.start_as_current_span("tokenization") as span:
                    span.set_attribute(
                        "token_count", encoded_text["input_ids"].shape[1]
                    )
                    span.set_attribute("duration_ms", tokenize_time * 1000)

            inference_start = time.time()
            if tracer:
                with tracer.start_as_current_span("model_inference") as span:
                    generated_tokens = self._model.generate(
                        **encoded_text,
                        forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(
                            tgt_lang
                        ),
                    )
                    inference_time = time.time() - inference_start
                    span.set_attribute("inference_time_seconds", inference_time)
                    span.set_attribute("output_token_count", generated_tokens.shape[1])
                    span.set_attribute("duration_ms", inference_time * 1000)
                    span.set_attribute("architecture", self._architecture)
            else:
                generated_tokens = self._model.generate(
                    **encoded_text,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_lang),
                )

            decode_start = time.time()
            if tracer:
                with tracer.start_as_current_span("decoding") as span:
                    translation = self._tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )[0]
                    span.set_attribute(
                        "duration_ms", (time.time() - decode_start) * 1000
                    )
            else:
                translation = self._tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]

            logging.info(
                f"CPU translation ({self._architecture}) completed in {time.time() - start_time:.2f}s"
            )
            return translation

        except Exception as e:
            logging.error(f"Error in CPU translation: {e}", exc_info=True)
            raise

    def get_languages(self) -> List[str]:
        """Get a list of supported language codes in NLLB format"""
        if self._tokenizer is None:
            raise ValueError("Model not loaded. Please check logs for errors.")
        return self._tokenizer.additional_special_tokens

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def architecture(self):
        return self._architecture

    @property
    def is_loaded(self):
        return self._model is not None and self._tokenizer is not None

    @property
    def model_type(self):
        return "nllb"


def get_translation_model() -> NLLBTranslationModel:
    return NLLBTranslationModel()
