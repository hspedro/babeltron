from abc import ABC, abstractmethod
from typing import Any


class CacheInterface(ABC):
    """
    Abstract base class for cache interface.

    Any concrete implementation must provide methods for:
    - Save data to cache with a TTL
    - Get data from cache
    - Delete data from cache
    """

    @abstractmethod
    def save(self, key: str, value: Any, ttl: int) -> None:
        """
        Save data to cache with a TTL.

        Args:
            key: The key to save the data to
            value: The data to save
            ttl: The time to live for the data
        """
        pass

    @abstractmethod
    def get(self, key: str) -> Any:
        """
        Get data from cache.

        Args:
            key: The key to retrieve the data from

        Returns:
            The data retrieved from cache
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete data from cache.
        """
        pass
