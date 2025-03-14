import logging
from functools import wraps
from typing import Any, Optional

import valkey

from babeltron.app.cache.base import CacheInterface
from babeltron.app.config import CACHE_HOST, CACHE_PORT, CACHE_TTL_SECONDS


def handle_cache_errors(func):
    """Decorator to handle cache operation errors"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except valkey.ConnectionError as e:
            logging.error(f"Cache connection error in {func.__name__}: {str(e)}")
            return None
        except valkey.TimeoutError as e:
            logging.error(f"Cache timeout error in {func.__name__}: {str(e)}")
            return None
        except Exception as e:
            logging.error(
                f"Unexpected error in cache operation {func.__name__}: {str(e)}"
            )
            return None

    return wrapper


class ValkeyCache(CacheInterface):
    """Valkey-based cache implementation with singleton pattern"""

    _instance: Optional["ValkeyCache"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ValkeyCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        host: str = CACHE_HOST,
        port: int = CACHE_PORT,
        ttl: int = CACHE_TTL_SECONDS,
    ):
        if self._initialized:
            return

        if not host or not port:
            raise ValueError("Cache host and port must be provided")
        self.host = host
        self.port = port
        self.ttl = ttl

        try:
            self.client = valkey.Valkey(host=self.host, port=self.port)
            logging.info(f"Successfully connected to Valkey cache at {host}:{port}")
        except Exception as e:
            logging.error(f"Failed to initialize Valkey cache: {str(e)}")
            raise

        self._initialized = True

    @handle_cache_errors
    def save(self, key: str, value: Any, ttl: int) -> None:
        """
        Save a value to the cache with the specified TTL.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Time to live in seconds
        """
        self.client.set(key, value, ttl)

    @handle_cache_errors
    def get(self, key: str) -> Any:
        """
        Get a value from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found or on error
        """
        return self.client.get(key)

    @handle_cache_errors
    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.

        Args:
            key: The cache key
        """
        self.client.delete(key)
