"""Python bindings for rust-core.

This module provides Python wrappers for the high-performance Rust implementations.
"""

from __future__ import annotations

RUST_AVAILABLE = False

# Default stub classes
class _StubClass:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "rust-core not compiled. Run 'maturin develop' in rust-core/ directory."
        )

# Initialize all exports with stubs
ByteTokenizer = _StubClass
BigramHashTokenizer = _StubClass
TrigramHashTokenizer = _StubClass
Quantizer = _StubClass
Int6Quantizer = _StubClass
Backbone = _StubClass
BackboneConfig = _StubClass
Activations = _StubClass
MultiHeadAttention = _StubClass
GroupedQueryAttention = _StubClass
XSAModule = _StubClass
FiLMModule = _StubClass
RecurrentBlock = _StubClass
BPBComputer = _StubClass
SlidingWindowEval = _StubClass

# Try to import the actual Rust module
try:
    # Import the compiled extension directly
    # This assumes maturin/PyO3 has created a proper extension module
    import rust_core as _rust_core  # type: ignore

    # Now try to get attributes
    try:
        ByteTokenizer = _rust_core.ByteTokenizer
        BigramHashTokenizer = _rust_core.BigramHashTokenizer
        TrigramHashTokenizer = _rust_core.TrigramHashTokenizer
        Quantizer = _rust_core.Quantizer
        Int6Quantizer = _rust_core.Int6Quantizer
        Backbone = _rust_core.Backbone
        BackboneConfig = _rust_core.BackboneConfig
        Activations = _rust_core.Activations
        MultiHeadAttention = _rust_core.MultiHeadAttention
        GroupedQueryAttention = _rust_core.GroupedQueryAttention
        XSAModule = _rust_core.XSAModule
        FiLMModule = _rust_core.FiLMModule
        RecurrentBlock = _rust_core.RecurrentBlock
        BPBComputer = _rust_core.BPBComputer
        SlidingWindowEval = _rust_core.SlidingWindowEval

        RUST_AVAILABLE = True
    except AttributeError as e:
        # Some attributes missing, keep stubs
        pass
except (ImportError, AttributeError, OSError):
    # Keep stubs
    pass


__all__ = [
    "ByteTokenizer",
    "BigramHashTokenizer",
    "TrigramHashTokenizer",
    "Quantizer",
    "Int6Quantizer",
    "Backbone",
    "BackboneConfig",
    "Activations",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "XSAModule",
    "FiLMModule",
    "RecurrentBlock",
    "BPBComputer",
    "SlidingWindowEval",
    "RUST_AVAILABLE",
]
