"""Seed management for reproducibility."""

import os
import random

import numpy as np

# Global seed state
_current_seed: int | None = None


def set_seed(seed: int) -> None:
    """Set seed for all random number generators.

    This ensures reproducibility across Python, NumPy, and PyTorch (if available).
    """
    global _current_seed
    _current_seed = seed

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Set environment variable for subprocesses
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_seed() -> int | None:
    """Get the current seed."""
    return _current_seed


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Get a NumPy random number generator with the given seed.

    Use this when you need a local RNG that doesn't affect global state.
    """
    if seed is None:
        seed = _current_seed if _current_seed is not None else 42
    return np.random.default_rng(seed)
