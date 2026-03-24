"""Tokenizer implementations and factory.

This module provides various tokenizer implementations:
- ByteTokenizer: Simple byte-level tokenization
- BigramHashTokenizer: Hash-based bigram tokenization
- TrigramHashTokenizer: Hash-based trigram tokenization
- FallbackTokenizer: Fallback mechanisms
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


class TokenizerProtocol(Protocol):
    """Protocol for tokenizers."""

    def encode(self, text: str) -> list[int]:
        """Encode text to tokens."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens to text."""
        ...

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer."""

    type: str = "byte"  # byte, bigram_hash, trigram_hash, fallback
    vocab_size: int = 256
    byte_fallback: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenizerConfig":
        """Create from dictionary."""
        return cls(
            type=d.get("type", "byte"),
            vocab_size=d.get("vocab_size", 256),
            byte_fallback=d.get("byte_fallback", True),
        )


class BaseTokenizer(ABC):
    """Base class for tokenizers."""

    def __init__(self, vocab_size: int = 256):
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to tokens."""
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode tokens to text."""
        pass

    def get_token_count(self, text: str) -> int:
        """Get number of tokens for text."""
        return len(self.encode(text))

    def get_stats(self, text: str) -> dict[str, Any]:
        """Get tokenization statistics."""
        tokens = self.encode(text)
        return {
            "num_tokens": len(tokens),
            "num_bytes": len(text.encode("utf-8")),
            "bytes_per_token": len(text.encode("utf-8")) / max(len(tokens), 1),
            "unique_tokens": len(set(tokens)),
            "vocab_utilization": len(set(tokens)) / self.vocab_size,
        }


class ByteTokenizer(BaseTokenizer):
    """Byte-level tokenizer.

    Simple tokenizer that works at the byte level.
    Each byte becomes a token.
    """

    def __init__(self, vocab_size: int = 256):
        super().__init__(min(vocab_size, 256))

    def encode(self, text: str) -> list[int]:
        """Encode text to byte tokens."""
        return [b for b in text.encode("utf-8")]

    def decode(self, tokens: list[int]) -> str:
        """Decode byte tokens to text."""
        try:
            return bytes(tokens).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as e:
            # Handle invalid bytes
            result = []
            for t in tokens:
                try:
                    result.append(bytes([t]).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    result.append("?")
            return "".join(result)


class BigramHashTokenizer(BaseTokenizer):
    """Bigram hash tokenizer.

    Uses hashing to represent bigrams in a fixed-size vocabulary.
    First byte is kept as-is (fallback), subsequent tokens are hashed bigrams.
    """

    def __init__(self, vocab_size: int = 4096, byte_fallback: bool = True):
        super().__init__(vocab_size)
        self.byte_fallback = byte_fallback
        self._hash_base = 256  # First 256 tokens reserved for bytes

    def encode(self, text: str) -> list[int]:
        """Encode text to bigram tokens."""
        if not text:
            return []

        bytes_data = text.encode("utf-8")
        tokens = []

        # First byte as fallback
        if self.byte_fallback and len(bytes_data) > 0:
            tokens.append(bytes_data[0])

        # Hash bigrams
        for i in range(len(bytes_data) - 1):
            bigram_token = self._hash_bigram(bytes_data[i], bytes_data[i + 1])
            tokens.append(bigram_token)

        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens (best effort - lossy for hashed tokens)."""
        result = []
        for token in tokens:
            if token < 256:
                # Byte fallback
                try:
                    result.append(bytes([token]).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    result.append("?")
            else:
                # Can't decode hashed bigrams exactly
                result.append("?")
        return "".join(result)

    def _hash_bigram(self, b1: int, b2: int) -> int:
        """Hash a bigram to a token."""
        # Simple hash: (b1 * prime1 + b2 * prime2) % vocab_range
        prime1 = 31
        prime2 = 37
        hash_val = (b1 * prime1 + b2 * prime2) % (self.vocab_size - self._hash_base)
        return hash_val + self._hash_base


class TrigramHashTokenizer(BaseTokenizer):
    """Trigram hash tokenizer.

    Uses hashing to represent trigrams in a fixed-size vocabulary.
    First two bytes are kept as-is (fallback), subsequent tokens are hashed trigrams.
    """

    def __init__(self, vocab_size: int = 8192, byte_fallback: bool = True):
        super().__init__(vocab_size)
        self.byte_fallback = byte_fallback
        self._hash_base = 256

    def encode(self, text: str) -> list[int]:
        """Encode text to trigram tokens."""
        if not text:
            return []

        bytes_data = text.encode("utf-8")
        tokens = []

        # First two bytes as fallback
        if self.byte_fallback:
            for i in range(min(2, len(bytes_data))):
                tokens.append(bytes_data[i])

        # Hash trigrams
        for i in range(len(bytes_data) - 2):
            trigram_token = self._hash_trigram(
                bytes_data[i], bytes_data[i + 1], bytes_data[i + 2]
            )
            tokens.append(trigram_token)

        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens (best effort - lossy for hashed tokens)."""
        result = []
        for token in tokens:
            if token < 256:
                try:
                    result.append(bytes([token]).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    result.append("?")
            else:
                result.append("?")
        return "".join(result)

    def _hash_trigram(self, b1: int, b2: int, b3: int) -> int:
        """Hash a trigram to a token."""
        prime1 = 31
        prime2 = 37
        prime3 = 41
        hash_val = (
            b1 * prime1 + b2 * prime2 + b3 * prime3
        ) % (self.vocab_size - self._hash_base)
        return hash_val + self._hash_base


class FallbackTokenizer(BaseTokenizer):
    """Fallback tokenizer that tries multiple strategies.

    Tries to use a primary tokenizer, falls back to byte-level
    if encoding fails.
    """

    def __init__(
        self,
        primary_type: str = "bigram_hash",
        vocab_size: int = 4096,
        byte_fallback: bool = True,
    ):
        super().__init__(vocab_size)
        self.primary_type = primary_type
        self.byte_fallback = byte_fallback

        # Create primary tokenizer
        if primary_type == "bigram_hash":
            self.primary = BigramHashTokenizer(vocab_size, byte_fallback)
        elif primary_type == "trigram_hash":
            self.primary = TrigramHashTokenizer(vocab_size, byte_fallback)
        else:
            self.primary = ByteTokenizer(vocab_size)

        self.fallback = ByteTokenizer(256)

    def encode(self, text: str) -> list[int]:
        """Encode text, falling back to byte tokenizer if needed."""
        try:
            tokens = self.primary.encode(text)
            if tokens:
                return tokens
        except (ValueError, UnicodeDecodeError, TypeError) as e:
            # Log warning about fallback (requires logger import)
            import logging
            logging.getLogger(f"tokenizer.{self.primary_type}").warning(
                f"Primary tokenizer failed: {e}, falling back to byte tokenizer"
            )
        return self.fallback.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens."""
        try:
            return self.primary.decode(tokens)
        except (ValueError, UnicodeDecodeError, TypeError) as e:
            import logging
            logging.getLogger(f"tokenizer.{self.primary_type}").warning(
                f"Primary tokenizer decode failed: {e}, falling back to byte tokenizer"
            )
            return self.fallback.decode(tokens)

    def get_primary_stats(self, text: str) -> dict[str, Any]:
        """Get statistics for primary tokenizer."""
        return self.primary.get_stats(text)


class TokenizerFactory:
    """Factory for creating tokenizers."""

    @staticmethod
    def create(config: TokenizerConfig | dict[str, Any]) -> BaseTokenizer:
        """Create a tokenizer from configuration.

        Args:
            config: TokenizerConfig or dictionary with configuration

        Returns:
            Created tokenizer instance
        """
        if isinstance(config, dict):
            config = TokenizerConfig.from_dict(config)

        tokenizer_type = config.type.lower()

        if tokenizer_type == "byte":
            return ByteTokenizer(config.vocab_size)
        elif tokenizer_type == "bigram_hash":
            return BigramHashTokenizer(config.vocab_size, config.byte_fallback)
        elif tokenizer_type == "trigram_hash":
            return TrigramHashTokenizer(config.vocab_size, config.byte_fallback)
        elif tokenizer_type == "fallback":
            return FallbackTokenizer(
                vocab_size=config.vocab_size,
                byte_fallback=config.byte_fallback,
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def create_tokenizer(
    tokenizer_type: str = "byte",
    vocab_size: int = 256,
    byte_fallback: bool = True,
) -> BaseTokenizer:
    """Convenience function to create a tokenizer.

    Args:
        tokenizer_type: Type of tokenizer
        vocab_size: Vocabulary size
        byte_fallback: Whether to use byte fallback

    Returns:
        Created tokenizer
    """
    config = TokenizerConfig(
        type=tokenizer_type,
        vocab_size=vocab_size,
        byte_fallback=byte_fallback,
    )
    return TokenizerFactory.create(config)
