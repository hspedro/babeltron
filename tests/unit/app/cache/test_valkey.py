import pytest
from unittest.mock import MagicMock, patch

from babeltron.app.cache.valkey import ValkeyCache


class TestValkeyCache:
    """Tests for the ValkeyCache class"""

    @pytest.fixture
    def mock_valkey(self):
        with patch('valkey.Valkey') as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            yield mock_client

    def test_singleton_pattern(self, mock_valkey):
        """Test that ValkeyCache follows the singleton pattern"""
        # Reset the singleton for testing
        ValkeyCache._instance = None

        # Create two instances
        cache1 = ValkeyCache(host="localhost", port=6379)
        cache2 = ValkeyCache(host="different-host", port=1234)

        # They should be the same object
        assert cache1 is cache2

        # The second initialization should not change the connection parameters
        assert cache1.host == "localhost"
        assert cache1.port == 6379

    def test_initialization_error(self):
        """Test that initialization errors are properly handled"""
        # Reset the singleton for testing
        ValkeyCache._instance = None

        with patch('valkey.Valkey', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                ValkeyCache(host="localhost", port=6379)

    def test_save_success(self, mock_valkey):
        """Test successful save operation"""
        # Reset the singleton for testing
        ValkeyCache._instance = None

        mock_valkey.set.return_value = "OK"

        cache = ValkeyCache(host="localhost", port=6379)
        cache.save("test-key", "test-value", 60)

        mock_valkey.set.assert_called_once_with("test-key", "test-value", 60)

    def test_save_error(self, mock_valkey):
        """Test error handling during save operation"""
        # Reset the singleton for testing
        ValkeyCache._instance = None

        mock_valkey.set.side_effect = Exception("Save failed")

        cache = ValkeyCache(host="localhost", port=6379)
        result = cache.save("test-key", "test-value", 60)

        assert result is None
        mock_valkey.set.assert_called_once_with("test-key", "test-value", 60)

    def test_get_success(self, mock_valkey):
        """Test successful get operation"""
        # Reset the singleton for testing
        ValkeyCache._instance = None

        mock_valkey.get.return_value = "cached-value"

        cache = ValkeyCache(host="localhost", port=6379)
        result = cache.get("test-key")

        assert result == "cached-value"
        mock_valkey.get.assert_called_once_with("test-key")

    def test_get_error(self, mock_valkey):
        """Test error handling during get operation"""
        # Reset the singleton for testing
        ValkeyCache._instance = None

        mock_valkey.get.side_effect = Exception("Get failed")

        cache = ValkeyCache(host="localhost", port=6379)
        result = cache.get("test-key")

        assert result is None
        mock_valkey.get.assert_called_once_with("test-key")

    def test_delete_success(self, mock_valkey):
        """Test successful delete operation"""
        # Reset the singleton for testing
        ValkeyCache._instance = None

        mock_valkey.delete.return_value = 1

        cache = ValkeyCache(host="localhost", port=6379)
        cache.delete("test-key")

        mock_valkey.delete.assert_called_once_with("test-key")

    def test_delete_error(self, mock_valkey):
        """Test error handling during delete operation"""
        # Reset the singleton for testing
        ValkeyCache._instance = None

        mock_valkey.delete.side_effect = Exception("Delete failed")

        cache = ValkeyCache(host="localhost", port=6379)
        result = cache.delete("test-key")

        assert result is None
        mock_valkey.delete.assert_called_once_with("test-key")
