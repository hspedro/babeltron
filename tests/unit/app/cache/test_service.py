import pytest
from unittest.mock import MagicMock

from babeltron.app.cache.service import CacheService


class TestCacheService:
    """Tests for the CacheService class"""

    @pytest.fixture
    def mock_cache(self):
        """Mock cache client"""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def cache_service(self, mock_cache):
        """Cache service with mock cache client"""
        return CacheService(cache_client=mock_cache)

    def test_get_translation_hit(self, cache_service, mock_cache):
        """Test getting a cached translation (cache hit)"""
        # Setup
        mock_cache.get.return_value = {"translated_text": "Bonjour le monde"}

        # Execute
        result = cache_service.get_translation("Hello world", "en", "fr")

        # Verify
        assert result == {"translated_text": "Bonjour le monde"}
        mock_cache.get.assert_called_once()

    def test_get_translation_miss(self, cache_service, mock_cache):
        """Test getting a translation that's not in cache (cache miss)"""
        # Setup
        mock_cache.get.return_value = None

        # Execute
        result = cache_service.get_translation("Hello world", "en", "fr")

        # Verify
        assert result is None
        mock_cache.get.assert_called_once()

    def test_save_translation(self, cache_service, mock_cache):
        """Test saving a translation to cache"""
        # Setup
        translation_result = {"translated_text": "Bonjour le monde"}

        # Execute
        cache_service.save_translation("Hello world", "en", "fr", translation_result)

        # Verify
        mock_cache.save.assert_called_once()
        # Check that the third argument (ttl) is passed
        assert len(mock_cache.save.call_args[0]) == 3

    def test_save_translation_error(self, cache_service, mock_cache):
        """Test error handling when saving a translation to cache"""
        # Setup
        translation_result = {"translated_text": "Bonjour le monde"}
        mock_cache.save.side_effect = Exception("Save failed")

        # Execute - should not raise an exception
        cache_service.save_translation("Hello world", "en", "fr", translation_result)

        # Verify
        mock_cache.save.assert_called_once()

    def test_get_detection_hit(self, cache_service, mock_cache):
        """Test getting a cached language detection (cache hit)"""
        # Setup
        mock_cache.get.return_value = {"detected_language": "en"}

        # Execute
        result = cache_service.get_detection("Hello world")

        # Verify
        assert result == {"detected_language": "en"}
        mock_cache.get.assert_called_once()

    def test_get_detection_miss(self, cache_service, mock_cache):
        """Test getting a language detection that's not in cache (cache miss)"""
        # Setup
        mock_cache.get.return_value = None

        # Execute
        result = cache_service.get_detection("Hello world")

        # Verify
        assert result is None
        mock_cache.get.assert_called_once()

    def test_save_detection(self, cache_service, mock_cache):
        """Test saving a language detection to cache"""
        # Setup
        detection_result = {"detected_language": "en"}

        # Execute
        cache_service.save_detection("Hello world", detection_result)

        # Verify
        mock_cache.save.assert_called_once()
        # Check that the third argument (ttl) is passed
        assert len(mock_cache.save.call_args[0]) == 3

    def test_save_detection_error(self, cache_service, mock_cache):
        """Test error handling when saving a language detection to cache"""
        # Setup
        detection_result = {"detected_language": "en"}
        mock_cache.save.side_effect = Exception("Save failed")

        # Execute - should not raise an exception
        cache_service.save_detection("Hello world", detection_result)

        # Verify
        mock_cache.save.assert_called_once()
