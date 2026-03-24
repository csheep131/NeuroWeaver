"""Tokenizers module for the ablation machine."""

from .tokenizers import (
    TokenizerProtocol,
    TokenizerConfig,
    BaseTokenizer,
    ByteTokenizer,
    BigramHashTokenizer,
    TrigramHashTokenizer,
    FallbackTokenizer,
    TokenizerFactory,
    create_tokenizer,
)

# Try to import Rust implementations
try:
    import rust_core

    RustByteTokenizer = rust_core.ByteTokenizer
    RustBigramHashTokenizer = rust_core.BigramHashTokenizer
    RustTrigramHashTokenizer = rust_core.TrigramHashTokenizer
    RUST_AVAILABLE = True
except (ImportError, AttributeError):
    RUST_AVAILABLE = False

    class _StubClass:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("rust-core not compiled")

    RustByteTokenizer = _StubClass
    RustBigramHashTokenizer = _StubClass
    RustTrigramHashTokenizer = _StubClass

__all__ = [
    "TokenizerProtocol",
    "TokenizerConfig",
    "BaseTokenizer",
    "ByteTokenizer",
    "BigramHashTokenizer",
    "TrigramHashTokenizer",
    "FallbackTokenizer",
    "TokenizerFactory",
    "create_tokenizer",
    "RustByteTokenizer",
    "RustBigramHashTokenizer",
    "RustTrigramHashTokenizer",
    "RUST_AVAILABLE",
]
