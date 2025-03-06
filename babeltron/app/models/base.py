from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class TranslationModelBase(ABC):
    """
    Abstract base class for translation models.

    Any concrete implementation must provide methods for:
    - Loading the model
    - Translating text
    - Getting supported languages
    """

    @abstractmethod
    def load(self) -> Tuple[Any, Any, str]:
        """
        Load the model and tokenizer, and determine the architecture.

        Returns:
            Tuple containing (model, tokenizer, architecture)
        """
        pass

    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str, tracer=None) -> str:
        """
        Translate text from source language to target language.

        Args:
            text: The text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            tracer: Optional OpenTelemetry tracer for spans (can be None)

        Returns:
            The translated text
        """
        pass

    @abstractmethod
    def get_languages(self) -> List[str]:
        """
        Get a list of supported language codes.

        Returns:
            List of language codes supported by the model
        """
        pass
