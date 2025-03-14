from babeltron.app.cache.utils import sanitize_text, generate_cache_key


class TestCacheUtils:
    """Tests for cache utility functions"""

    def test_sanitize_text_normalization(self):
        """Test Unicode normalization in text sanitization"""
        # Test with decomposed character (é = e + ´)
        text_with_decomposed = "caf\u0065\u0301"  # café with decomposed é
        sanitized = sanitize_text(text_with_decomposed)

        # Should be normalized to composed form (é)
        assert "café" in sanitized

    def test_sanitize_text_whitespace(self):
        """Test whitespace handling in text sanitization"""
        # Test with excess whitespace
        text_with_whitespace = "  Hello   world!  "
        sanitized = sanitize_text(text_with_whitespace)

        assert sanitized == "hello world!"

        # Test with newlines and tabs
        text_with_newlines = "Hello\n\n  world!\t\tHow are you?"
        sanitized = sanitize_text(text_with_newlines)

        assert sanitized == "hello world! how are you?"

    def test_sanitize_text_case(self):
        """Test case normalization in text sanitization"""
        # Test with mixed case
        text_with_mixed_case = "Hello World! THIS is A TeSt."
        sanitized = sanitize_text(text_with_mixed_case)

        assert sanitized == "hello world! this is a test."

    def test_sanitize_text_punctuation(self):
        """Test punctuation preservation in text sanitization"""
        # Test with important punctuation
        text_with_punctuation = "Let's eat, grandma! Don't remove this."
        sanitized = sanitize_text(text_with_punctuation)

        assert "let's eat, grandma!" in sanitized
        assert "don't remove this" in sanitized

    def test_generate_cache_key_translate(self):
        """Test cache key generation for translation"""
        # Test with translation parameters
        key = generate_cache_key("translate", "Hello world", "en", "fr")

        # Key should be an MD5 hash (32 hex characters)
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

        # Same input should produce the same key
        key2 = generate_cache_key("translate", "Hello world", "en", "fr")
        assert key == key2

        # Different input should produce different keys
        key3 = generate_cache_key("translate", "Hello world", "en", "es")
        assert key != key3

    def test_generate_cache_key_detect(self):
        """Test cache key generation for language detection"""
        # Test with detection parameters
        key = generate_cache_key("detect", "Hello world")

        # Key should be an MD5 hash (32 hex characters)
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

        # Same input should produce the same key
        key2 = generate_cache_key("detect", "Hello world")
        assert key == key2

        # Different input should produce different keys
        key3 = generate_cache_key("detect", "Bonjour le monde")
        assert key != key3
