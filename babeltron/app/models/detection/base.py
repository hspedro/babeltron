from abc import ABC, abstractmethod
from typing import Any, Tuple


class DetectionModelBase(ABC):
    """
    Abstract base class for detection models.

    Any concrete implementation must provide methods for:
    - Loading the model
    - Detecting language of text
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
    def detect(self, text: str, tracer=None) -> str:
        """
        Detect language of text.

        Args:
            text: The text to detect language
            tracer: Optional OpenTelemetry tracer for spans (can be None)

        Returns:
            The detected language
        """
        pass
