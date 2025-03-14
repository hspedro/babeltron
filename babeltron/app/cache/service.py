import logging
from typing import Generic, Optional, TypeVar, cast

from babeltron.app.cache.base import CacheInterface
from babeltron.app.cache.utils import generate_cache_key
from babeltron.app.cache.valkey import ValkeyCache
from babeltron.app.config import CACHE_TTL_SECONDS

# Type variable for generic response types
T = TypeVar("T")


class CacheService(Generic[T]):
    """
    Service for caching translation and detection results
    """

    def __init__(
        self,
        cache_client: Optional[CacheInterface] = None,
        ttl: int = CACHE_TTL_SECONDS,
    ):
        """
        Initialize the cache service

        Args:
            cache_client: Optional cache client, defaults to ValkeyCache
            ttl: Time to live for cache entries in seconds
        """
        self.cache = cache_client or ValkeyCache()
        self.ttl = ttl
        self.logger = logging.getLogger(__name__)

    def get_translation(self, text: str, src_lang: str, tgt_lang: str) -> Optional[T]:
        """
        Get a cached translation result

        Args:
            text: The text to translate
            src_lang: Source language
            tgt_lang: Target language

        Returns:
            Cached translation result or None if not found
        """
        cache_key = generate_cache_key("translate", text, src_lang, tgt_lang)
        self.logger.debug(f"Looking up translation in cache with key: {cache_key}")

        cached_result = self.cache.get(cache_key)

        if cached_result:
            self.logger.info(f"Cache hit for translation: {src_lang} -> {tgt_lang}")
            return cast(T, cached_result)

        self.logger.info(f"Cache miss for translation: {src_lang} -> {tgt_lang}")
        return None

    def save_translation(
        self, text: str, src_lang: str, tgt_lang: str, result: T
    ) -> None:
        """
        Save a translation result to cache

        Args:
            text: The text that was translated
            src_lang: Source language
            tgt_lang: Target language
            result: Translation result to cache
        """
        cache_key = generate_cache_key("translate", text, src_lang, tgt_lang)
        self.logger.debug(f"Saving translation to cache with key: {cache_key}")

        try:
            self.cache.save(cache_key, result, self.ttl)
            self.logger.info(f"Cached translation result: {src_lang} -> {tgt_lang}")
        except Exception as e:
            self.logger.error(f"Failed to cache translation result: {str(e)}")

    def get_detection(self, text: str) -> Optional[T]:
        """
        Get a cached language detection result

        Args:
            text: The text to detect language for

        Returns:
            Cached detection result or None if not found
        """
        cache_key = generate_cache_key("detect", text)
        self.logger.debug(
            f"Looking up language detection in cache with key: {cache_key}"
        )

        cached_result = self.cache.get(cache_key)

        if cached_result:
            self.logger.info("Cache hit for language detection")
            return cast(T, cached_result)

        self.logger.info("Cache miss for language detection")
        return None

    def save_detection(self, text: str, result: T) -> None:
        """
        Save a language detection result to cache

        Args:
            text: The text that was analyzed
            result: Detection result to cache
        """
        cache_key = generate_cache_key("detect", text)
        self.logger.debug(f"Saving language detection to cache with key: {cache_key}")

        try:
            self.cache.save(cache_key, result, self.ttl)
            self.logger.info("Cached language detection result")
        except Exception as e:
            self.logger.error(f"Failed to cache language detection result: {str(e)}")
