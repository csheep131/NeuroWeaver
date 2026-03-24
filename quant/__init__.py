"""Quantization module for the ablation machine."""

from .quantizers import (
    QuantizerProtocol,
    QuantizerConfig,
    BaseQuantizer,
    Int6Quantizer,
    Int5Quantizer,
    MixedQuantizer,
    GPTQLiteQuantizer,
    QuantizerFactory,
    create_quantizer,
    compute_quantization_metrics,
)

# Try to import Rust implementations
try:
    import rust_core

    RustQuantizer = rust_core.Quantizer
    RustInt6Quantizer = rust_core.Int6Quantizer
    RUST_AVAILABLE = True
except (ImportError, AttributeError):
    RUST_AVAILABLE = False

    class _StubClass:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("rust-core not compiled")

    RustQuantizer = _StubClass
    RustInt6Quantizer = _StubClass

__all__ = [
    "QuantizerProtocol",
    "QuantizerConfig",
    "BaseQuantizer",
    "Int6Quantizer",
    "Int5Quantizer",
    "MixedQuantizer",
    "GPTQLiteQuantizer",
    "QuantizerFactory",
    "create_quantizer",
    "compute_quantization_metrics",
    "RustQuantizer",
    "RustInt6Quantizer",
    "RUST_AVAILABLE",
]
