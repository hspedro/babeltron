import logging
from typing import Any, Tuple

from lingua import Language, LanguageDetectorBuilder

from babeltron.app.models.detection.base import DetectionModelBase

# Initialize Lingua language detector with all supported languages
LINGUA_DETECTOR = LanguageDetectorBuilder.from_all_languages().build()

# Mapping from Lingua Language enum to ISO 639-1 codes
LINGUA_TO_ISO = {
    Language.AFRIKAANS: "af",
    Language.ALBANIAN: "sq",
    Language.ARABIC: "ar",
    Language.ARMENIAN: "hy",
    Language.AZERBAIJANI: "az",
    Language.BASQUE: "eu",
    Language.BELARUSIAN: "be",
    Language.BENGALI: "bn",
    Language.BOKMAL: "nb",
    Language.BOSNIAN: "bs",
    Language.BULGARIAN: "bg",
    Language.CATALAN: "ca",
    Language.CHINESE: "zh",
    Language.CROATIAN: "hr",
    Language.CZECH: "cs",
    Language.DANISH: "da",
    Language.DUTCH: "nl",
    Language.ENGLISH: "en",
    Language.ESPERANTO: "eo",
    Language.ESTONIAN: "et",
    Language.FINNISH: "fi",
    Language.FRENCH: "fr",
    Language.GANDA: "lg",
    Language.GEORGIAN: "ka",
    Language.GERMAN: "de",
    Language.GREEK: "el",
    Language.GUJARATI: "gu",
    Language.HEBREW: "he",
    Language.HINDI: "hi",
    Language.HUNGARIAN: "hu",
    Language.ICELANDIC: "is",
    Language.INDONESIAN: "id",
    Language.IRISH: "ga",
    Language.ITALIAN: "it",
    Language.JAPANESE: "ja",
    Language.KAZAKH: "kk",
    Language.KOREAN: "ko",
    Language.LATIN: "la",
    Language.LATVIAN: "lv",
    Language.LITHUANIAN: "lt",
    Language.MACEDONIAN: "mk",
    Language.MALAY: "ms",
    Language.MAORI: "mi",
    Language.MARATHI: "mr",
    Language.MONGOLIAN: "mn",
    Language.NYNORSK: "nn",
    Language.PERSIAN: "fa",
    Language.POLISH: "pl",
    Language.PORTUGUESE: "pt",
    Language.PUNJABI: "pa",
    Language.ROMANIAN: "ro",
    Language.RUSSIAN: "ru",
    Language.SERBIAN: "sr",
    Language.SHONA: "sn",
    Language.SLOVAK: "sk",
    Language.SLOVENE: "sl",
    Language.SOMALI: "so",
    Language.SOTHO: "st",
    Language.SPANISH: "es",
    Language.SWAHILI: "sw",
    Language.SWEDISH: "sv",
    Language.TAGALOG: "tl",
    Language.TAMIL: "ta",
    Language.TELUGU: "te",
    Language.THAI: "th",
    Language.TSONGA: "ts",
    Language.TSWANA: "tn",
    Language.TURKISH: "tr",
    Language.UKRAINIAN: "uk",
    Language.URDU: "ur",
    Language.VIETNAMESE: "vi",
    Language.WELSH: "cy",
    Language.XHOSA: "xh",
    Language.YORUBA: "yo",
    Language.ZULU: "zu",
}


class LinguaDetectionModel(DetectionModelBase):
    """
    Language detection model using Lingua library.

    Lingua is a natural language detection library that's designed to be
    highly accurate even for short text snippets.
    """

    def __init__(self):
        self.detector = LINGUA_DETECTOR
        self.is_loaded = True
        self.architecture = "lingua"
        self.model_type = "lingua"
        logging.info("Initialized Lingua language detection model")

    def load(self) -> Tuple[Any, Any, str]:
        """
        Load the model. For Lingua, this is a no-op as the detector is already initialized.

        Returns:
            Tuple containing (detector, None, architecture)
        """
        return self.detector, None, self.architecture

    def detect(self, text: str, tracer=None) -> Tuple[str, float]:
        """
        Detect language of text using Lingua.

        Args:
            text: The text to detect language
            tracer: Optional OpenTelemetry tracer for spans (can be None)

        Returns:
            Tuple of (language_code, confidence)
        """
        # Create a span if tracer is provided
        if tracer:
            with tracer.start_as_current_span("lingua_detect") as span:
                span.set_attribute("text_length", len(text))
                return self._detect_language(text)
        else:
            return self._detect_language(text)

    def _detect_language(self, text: str) -> Tuple[str, float]:
        """
        Internal method to detect language using Lingua.

        Args:
            text: The text to detect language

        Returns:
            Tuple of (language_code, confidence)
        """
        # Detect the language using Lingua
        detected_language = self.detector.detect_language_of(text)

        # If no language was detected, return a default
        if detected_language is None:
            return "en", 0.0

        # Get the ISO code for the detected language
        language_code = LINGUA_TO_ISO.get(detected_language, "en")

        # Get the confidence score
        confidence = self.detector.compute_language_confidence(text, detected_language)

        return language_code, confidence


def get_lingua_model() -> LinguaDetectionModel:
    """
    Factory function to get a Lingua detection model instance.

    Returns:
        An instance of LinguaDetectionModel
    """
    return LinguaDetectionModel()
