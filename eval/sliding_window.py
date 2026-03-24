"""Sliding window evaluation utilities."""

from dataclasses import dataclass
from typing import Iterator


@dataclass
class Window:
    """A sliding window over text."""

    text: str
    start: int
    end: int
    index: int


def create_sliding_windows(
    text: str,
    window_size: int,
    stride: int | None = None,
    include_partial: bool = True,
) -> list[Window]:
    """Create sliding windows over text.

    Args:
        text: The text to create windows from
        window_size: Size of each window in characters
        stride: Step size between windows (default: window_size // 2)
        include_partial: Whether to include partial windows at the end

    Returns:
        List of Window objects
    """
    if stride is None:
        stride = window_size // 2

    windows = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + window_size, len(text))
        window_text = text[start:end]

        # Skip empty windows
        if not window_text:
            start += stride
            index += 1
            continue

        # Check if this is a partial window
        is_partial = len(window_text) < window_size

        if is_partial and not include_partial:
            break

        windows.append(Window(
            text=window_text,
            start=start,
            end=end,
            index=index,
        ))

        start += stride
        index += 1

        # If we've reached the end, break
        if end >= len(text):
            break

    return windows


def window_iterator(
    text: str,
    window_size: int,
    stride: int | None = None,
    include_partial: bool = True,
) -> Iterator[Window]:
    """Iterate over sliding windows lazily.

    Args:
        text: The text to create windows from
        window_size: Size of each window in characters
        stride: Step size between windows (default: window_size // 2)
        include_partial: Whether to include partial windows at the end

    Yields:
        Window objects
    """
    if stride is None:
        stride = window_size // 2

    start = 0
    index = 0

    while start < len(text):
        end = min(start + window_size, len(text))
        window_text = text[start:end]

        if not window_text:
            start += stride
            index += 1
            continue

        is_partial = len(window_text) < window_size

        if is_partial and not include_partial:
            break

        yield Window(
            text=window_text,
            start=start,
            end=end,
            index=index,
        )

        start += stride
        index += 1

        if end >= len(text):
            break


def compute_num_windows(
    text_length: int,
    window_size: int,
    stride: int | None = None,
    include_partial: bool = True,
) -> int:
    """Compute the number of windows for a given text length."""
    if stride is None:
        stride = window_size // 2

    if text_length <= window_size:
        return 1

    if include_partial:
        return (text_length - window_size + stride - 1) // stride + 1
    else:
        return (text_length - window_size) // stride + 1
