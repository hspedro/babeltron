import logging
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from babeltron.app.config import MODEL_PATH
from babeltron.app.models.translation.base import TranslationModelBase


def get_model_path() -> str:
    """
    Get the path to the M2M100 model directory.

    This function looks for model directories in the following order:
    1. The MODEL_PATH environment variable
    2. Any directory matching m2m100-* in /models, project_root/models, or ./models
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

    # First, look for M2M100 model directories
    for base_path in possible_paths:
        if not base_path.exists():
            continue

        m2m_dirs = list(base_path.glob("m2m*"))
        if m2m_dirs:
            # Use the first matching directory
            logging.info(f"Found M2M100 model directory: {m2m_dirs[0]}")
            return str(m2m_dirs[0])

    # Fallback: look for any directory with a config.json file
    for path in possible_paths:
        if path.exists() and any(path.glob("**/config.json")):
            logging.info(f"Found model directory with config.json: {path}")
            return str(path)

    logging.warning("No model directory found, using default: ./models")
    return "./models"


class ModelArchitecture:
    CUDA_FP16 = "cuda_fp16"
    MPS_FP16 = "mps_fp16"  # New architecture for Apple Silicon GPUs
    ROCM_FP16 = "rocm_fp16"  # AMD GPUs with ROCm
    CPU_QUANTIZED = "cpu_quantized"
    CPU_COMPILED = "cpu_compiled"
    CPU_STANDARD = "cpu_standard"


class M2M100TranslationModel(TranslationModelBase):
    """M2M100 translation model implementation with optimizations based on architecture"""

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(M2M100TranslationModel, cls).__new__(cls)
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
            logging.warning(f"MPS optimization failed: {e}")
            # Fallback to CPU if MPS fails
            self._optimize_for_cpu()

    def _optimize_for_rocm(self) -> None:
        """Optimize model for ROCm (AMD GPU) execution"""
        try:
            self._model.to("cuda")  # ROCm uses the same device name as CUDA
            self._model.half()  # Use FP16 for faster inference
            self._architecture = ModelArchitecture.ROCM_FP16
            logging.info("Model optimized for ROCm (AMD GPU) with FP16")
        except Exception as e:
            logging.warning(f"ROCm optimization failed: {e}")
            # Fallback to CPU if ROCm fails
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
            self._model = M2M100ForConditionalGeneration.from_pretrained(
                self._model_path
            )
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
            return self._translate_gpu(text, src_lang, tgt_lang, tracer, "cuda")
        elif self._architecture == ModelArchitecture.ROCM_FP16:
            return self._translate_gpu(
                text, src_lang, tgt_lang, tracer, "cuda"
            )  # ROCm uses "cuda" as device name
        elif self._architecture == ModelArchitecture.MPS_FP16:
            return self._translate_gpu(text, src_lang, tgt_lang, tracer, "mps")
        else:
            return self._translate_cpu(text, src_lang, tgt_lang, tracer)

    def _translate_gpu(
        self, text: str, src_lang: str, tgt_lang: str, tracer=None, device="cuda"
    ) -> str:
        """Translate using GPU-optimized model (CUDA or MPS)"""
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
                    inference_span.set_attribute("architecture", self._architecture)
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
                f"{arch_name} translation completed in {time.time() - start_time:.2f}s"
            )
            return translation

        except Exception as e:
            logging.error(f"Error in {arch_name} translation: {e}", exc_info=True)
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
        """Get a list of supported language codes"""
        if self._tokenizer is None:
            raise ValueError("Model not loaded. Please check logs for errors.")
        return list(self._tokenizer.lang_code_to_id.keys())

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


def get_translation_model() -> M2M100TranslationModel:
    return M2M100TranslationModel()
