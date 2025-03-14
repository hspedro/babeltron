import hashlib
import re
import unicodedata
from typing import Optional


def sanitize_text(text: str) -> str:
    """
    Sanitize text for cache key generation:
    1. Normalize Unicode characters to their canonical form
    2. Preserve important punctuation
    3. Trim excess whitespace
    4. Convert to lowercase for consistency

    Args:
        text: The text to sanitize

    Returns:
        Sanitized text
    """
    # Normalize Unicode characters (NFC form)
    normalized = unicodedata.normalize("NFC", text)

    # Trim excess whitespace (replace multiple spaces with a single space)
    # but preserve newlines and other meaningful spacing
    trimmed = re.sub(r"\s+", " ", normalized).strip()

    # Convert to lowercase for consistency
    lowercased = trimmed.lower()

    return lowercased


def generate_cache_key(
    prefix: str,
    text: str,
    src_lang: Optional[str] = None,
    tgt_lang: Optional[str] = None,
) -> str:
    """
    Generate a cache key for the given text and languages.

    Args:
        prefix: The prefix for the cache key (e.g., 'translate', 'detect')
        text: The text to generate a key for
        src_lang: Optional source language
        tgt_lang: Optional target language

    Returns:
        A cache key string
    """
    # Sanitize the text
    sanitized = sanitize_text(text)

    # Build the key components
    key_parts = [prefix]

    if src_lang:
        key_parts.append(src_lang)

    if tgt_lang:
        key_parts.append(tgt_lang)

    key_parts.append(sanitized)

    # Join the parts with a colon
    key_string = ":".join(key_parts)

    # Generate MD5 hash
    md5_hash = hashlib.md5(key_string.encode("utf-8")).hexdigest()

    return md5_hash
