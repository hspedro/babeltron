import pytest
from unittest.mock import patch, MagicMock

from lingua import Language
from babeltron.app.models.detection.lingua import (
    LinguaDetectionModel,
    get_lingua_model,
    LINGUA_TO_ISO,
)


class TestLinguaDetectionModel:
    def test_init(self):
        """Test that the model initializes correctly."""
        model = LinguaDetectionModel()
        assert model.is_loaded is True
        assert model.architecture == "lingua"
        assert model.detector is not None

    def test_load(self):
        """Test that the load method returns the expected values."""
        model = LinguaDetectionModel()
        detector, tokenizer, architecture = model.load()
        assert detector is model.detector
        assert tokenizer is None
        assert architecture == "lingua"

    @patch("babeltron.app.models.detection.lingua.LINGUA_DETECTOR")
    def test_detect_with_tracer(self, mock_detector):
        """Test detection with a tracer."""
        # Setup mock detector
        mock_detector.detect_language_of.return_value = Language.ENGLISH
        mock_detector.compute_language_confidence.return_value = 0.95

        # Setup mock tracer
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        # Create model and detect
        model = LinguaDetectionModel()
        model.detector = mock_detector

        language, confidence = model.detect("Hello, world!", mock_tracer)

        # Verify results
        assert language == "en"
        assert confidence == 0.95
        mock_detector.detect_language_of.assert_called_once_with("Hello, world!")
        mock_detector.compute_language_confidence.assert_called_once_with(
            "Hello, world!", Language.ENGLISH
        )
        mock_tracer.start_as_current_span.assert_called_once_with("lingua_detect")
        mock_span.set_attribute.assert_called_once_with("text_length", 13)

    @patch("babeltron.app.models.detection.lingua.LINGUA_DETECTOR")
    def test_detect_without_tracer(self, mock_detector):
        """Test detection without a tracer."""
        # Setup mock detector
        mock_detector.detect_language_of.return_value = Language.SPANISH
        mock_detector.compute_language_confidence.return_value = 0.85

        # Create model and detect
        model = LinguaDetectionModel()
        model.detector = mock_detector

        language, confidence = model.detect("Hola, mundo!", None)

        # Verify results
        assert language == "es"
        assert confidence == 0.85
        mock_detector.detect_language_of.assert_called_once_with("Hola, mundo!")
        mock_detector.compute_language_confidence.assert_called_once_with(
            "Hola, mundo!", Language.SPANISH
        )

    @patch("babeltron.app.models.detection.lingua.LINGUA_DETECTOR")
    def test_detect_no_language_detected(self, mock_detector):
        """Test detection when no language is detected."""
        # Setup mock detector to return None
        mock_detector.detect_language_of.return_value = None

        # Create model and detect
        model = LinguaDetectionModel()
        model.detector = mock_detector

        language, confidence = model.detect("", None)

        # Verify results
        assert language == "en"  # Default language
        assert confidence == 0.0
        mock_detector.detect_language_of.assert_called_once_with("")
        mock_detector.compute_language_confidence.assert_not_called()

    @patch("babeltron.app.models.detection.lingua.LINGUA_DETECTOR")
    def test_detect_unknown_language(self, mock_detector):
        """Test detection with a language not in the mapping."""
        # Create a mock language that's not in our mapping
        mock_language = MagicMock()

        # Setup mock detector
        mock_detector.detect_language_of.return_value = mock_language
        mock_detector.compute_language_confidence.return_value = 0.5

        # Create model and detect
        model = LinguaDetectionModel()
        model.detector = mock_detector

        language, confidence = model.detect("Unknown text", None)

        # Verify results
        assert language == "en"  # Default language for unknown
        assert confidence == 0.5
        mock_detector.detect_language_of.assert_called_once_with("Unknown text")
        mock_detector.compute_language_confidence.assert_called_once_with(
            "Unknown text", mock_language
        )

    def test_get_lingua_model(self):
        """Test the factory function returns a LinguaDetectionModel instance."""
        model = get_lingua_model()
        assert isinstance(model, LinguaDetectionModel)

    def test_lingua_to_iso_mapping(self):
        """Test that the LINGUA_TO_ISO mapping contains all expected languages."""
        # Check a few key languages
        assert LINGUA_TO_ISO[Language.ENGLISH] == "en"
        assert LINGUA_TO_ISO[Language.SPANISH] == "es"
        assert LINGUA_TO_ISO[Language.FRENCH] == "fr"
        assert LINGUA_TO_ISO[Language.GERMAN] == "de"
        assert LINGUA_TO_ISO[Language.CHINESE] == "zh"
        assert LINGUA_TO_ISO[Language.JAPANESE] == "ja"
        assert LINGUA_TO_ISO[Language.ARABIC] == "ar"

        # Check that we have mappings for all 75 languages
        assert len(LINGUA_TO_ISO) == 75
