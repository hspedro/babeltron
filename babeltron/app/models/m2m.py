import logging
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from babeltron.app.models.base import TranslationModelBase


def get_model_path() -> str:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return env_path

    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent

    possible_paths = [
        Path("/models"),
        project_root / "models",
        Path("./models"),
    ]

    for path in possible_paths:
        if path.exists() and any(path.glob("**/config.json")):
            return str(path)

    return "./models"


class ModelArchitecture:
    CUDA_FP16 = "cuda_fp16"
    CPU_QUANTIZED = "cpu_quantized"
    CPU_COMPILED = "cpu_compiled"
    CPU_STANDARD = "cpu_standard"


class M2MTranslationModel(TranslationModelBase):
    """M2M100 translation model implementation with optimizations based on architecture"""

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(M2MTranslationModel, cls).__new__(cls)
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

            self._model = M2M100ForConditionalGeneration.from_pretrained(
                self._model_path
            )
            self._tokenizer = M2M100Tokenizer.from_pretrained(self._model_path)

            if torch.cuda.is_available():
                self._optimize_for_cuda()
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
        """Optimize the model for CUDA (GPU) execution"""
        try:
            logging.info("Optimizing model for CUDA execution with FP16")
            self._model = self._model.half()  # FP16 precision
            self._model = self._model.to("cuda")
            self._architecture = ModelArchitecture.CUDA_FP16
        except Exception as e:
            logging.error(f"Failed to optimize for CUDA: {e}", exc_info=True)
            self._optimize_for_cpu()

    def _optimize_for_cpu(self) -> None:
        """Try various CPU optimization strategies"""
        if self._try_quantization():
            return

        if self._try_compilation():
            return

        logging.info("Using standard CPU execution (no optimizations)")
        self._architecture = ModelArchitecture.CPU_STANDARD

    def _try_quantization(self) -> bool:
        """Try to optimize using PyTorch quantization"""
        try:
            logging.info("Attempting quantization optimization")
            torch.backends.quantized.engine = "fbgemm"
            quantized_model = torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8
            )
            self._model = quantized_model
            self._architecture = ModelArchitecture.CPU_QUANTIZED
            logging.info("Successfully optimized model with quantization")
            return True

        except Exception as e:
            logging.warning(f"Quantization optimization failed: {e}")
            return False

    def _try_compilation(self) -> bool:
        """Try to optimize using torch.compile"""
        try:
            if not hasattr(torch, "compile"):
                logging.info("torch.compile not available (requires PyTorch 2.0+)")
                return False

            logging.info("Attempting compilation optimization")
            compiled_model = torch.compile(self._model)
            self._model = compiled_model
            self._architecture = ModelArchitecture.CPU_COMPILED
            logging.info("Successfully optimized model with compilation")
            return True

        except Exception as e:
            logging.warning(f"Compilation optimization failed: {e}")
            return False

    def translate(self, text: str, src_lang: str, tgt_lang: str, tracer=None) -> str:
        """
        Translate text using the appropriate method for the current architecture

        Args:
            text: The text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            tracer: Optional OpenTelemetry tracer for spans

        Returns:
            The translated text
        """
        if self._model is None or self._tokenizer is None:
            raise ValueError("Model not loaded. Please check logs for errors.")

        # Set the source language for the tokenizer
        self._tokenizer.src_lang = src_lang

        # Choose the appropriate translation method based on architecture
        if self._architecture == ModelArchitecture.CUDA_FP16:
            return self._translate_cuda(text, src_lang, tgt_lang, tracer)
        else:
            return self._translate_cpu(text, src_lang, tgt_lang, tracer)

    def _translate_cuda(
        self, text: str, src_lang: str, tgt_lang: str, tracer=None
    ) -> str:
        """Translate using CUDA-optimized model"""
        start_time = time.time()

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

            encoded_text = {k: v.to("cuda") for k, v in encoded_text.items()}

            if tracer:
                with tracer.start_as_current_span("model_inference") as span:
                    inference_span = span
                    inference_start = time.time()
                    generated_tokens = self._model.generate(
                        **encoded_text,
                        forced_bos_token_id=self._tokenizer.get_lang_id(tgt_lang),
                    )
                    inference_time = time.time() - inference_start
                    inference_span.set_attribute(
                        "inference_time_seconds", inference_time
                    )
                    inference_span.set_attribute(
                        "output_token_count", generated_tokens.shape[1]
                    )
                    inference_span.set_attribute("duration_ms", inference_time * 1000)
            else:
                generated_tokens = self._model.generate(
                    **encoded_text,
                    forced_bos_token_id=self._tokenizer.get_lang_id(tgt_lang),
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
                f"CUDA translation completed in {time.time() - start_time:.2f}s"
            )
            return translation

        except Exception as e:
            logging.error(f"Error in CUDA translation: {e}", exc_info=True)
            if tracer and tracer.get_current_span():
                tracer.get_current_span().record_exception(e)
            raise

    def _translate_cpu(
        self, text: str, src_lang: str, tgt_lang: str, tracer=None
    ) -> str:
        """Translate using CPU model (standard, quantized, or compiled)"""
        start_time = time.time()

        try:
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
                        forced_bos_token_id=self._tokenizer.get_lang_id(tgt_lang),
                    )
                    inference_time = time.time() - inference_start
                    span.set_attribute("inference_time_seconds", inference_time)
                    span.set_attribute("output_token_count", generated_tokens.shape[1])
                    span.set_attribute("duration_ms", inference_time * 1000)
                    span.set_attribute("architecture", self._architecture)
            else:
                generated_tokens = self._model.generate(
                    **encoded_text,
                    forced_bos_token_id=self._tokenizer.get_lang_id(tgt_lang),
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
            if tracer and tracer.get_current_span():
                tracer.get_current_span().record_exception(e)
            raise

    def get_languages(self) -> List[str]:
        """Get list of supported languages"""
        if self._tokenizer is None:
            raise ValueError("Model not loaded. Please check logs for errors.")

        lang_codes = self._tokenizer.lang_code_to_id

        return list(lang_codes.keys())

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


def get_translation_model() -> M2MTranslationModel:
    return M2MTranslationModel()
