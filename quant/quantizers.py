"""Quantization Lab for model compression.

This module provides various quantization strategies:
- Int6Quantizer: 6-bit integer quantization
- Int5Quantizer: 5-bit integer quantization
- MixedQuantizer: Mixed INT5/INT6 quantization
- GPTQLiteQuantizer: Simplified GPTQ-style quantization
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


class QuantizerProtocol(Protocol):
    """Protocol for quantizers."""

    def quantize(self, weights: list[float], rows: int, cols: int) -> list[int]:
        """Quantize weights."""
        ...

    def dequantize(self, quantized: list[int], rows: int, cols: int) -> list[float]:
        """Dequantize weights."""
        ...

    @property
    def bits_per_weight(self) -> float:
        """Get bits per weight."""
        ...

    @property
    def compression_ratio(self) -> float:
        """Get compression ratio vs fp32."""
        ...


@dataclass
class QuantizerConfig:
    """Configuration for quantization."""

    type: str = "int6"  # int6, int5, int5_int6_mixed, gptq_lite
    enabled: bool = False
    calibration_samples: int = 256
    group_size: int = 128  # For group-wise quantization

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuantizerConfig":
        """Create from dictionary."""
        return cls(
            type=d.get("type", "int6"),
            enabled=d.get("enabled", False),
            calibration_samples=d.get("calibration_samples", 256),
            group_size=d.get("group_size", 128),
        )


class BaseQuantizer(ABC):
    """Base class for quantizers."""

    def __init__(self, bits: int = 8):
        self.bits = bits

    @property
    def bits_per_weight(self) -> float:
        return float(self.bits)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs fp32 (32-bit)."""
        return 32.0 / self.bits

    @abstractmethod
    def quantize(self, weights: list[float], rows: int, cols: int) -> list[int]:
        """Quantize weights to integers."""
        pass

    @abstractmethod
    def dequantize(self, quantized: list[int], rows: int, cols: int) -> list[float]:
        """Dequantize integers to weights."""
        pass

    def quantize_error(
        self, weights: list[float], rows: int, cols: int
    ) -> dict[str, float]:
        """Compute quantization error metrics."""
        original = weights
        quantized = self.quantize(weights, rows, cols)
        dequantized = self.dequantize(quantized, rows, cols)

        # Compute errors
        errors = [abs(o - d) for o, d in zip(original, dequantized)]
        max_error = max(errors)
        mean_error = sum(errors) / len(errors)
        mse = sum(e * e for e in errors) / len(errors)

        return {
            "max_error": max_error,
            "mean_error": mean_error,
            "mse": mse,
            "rmse": mse**0.5,
            "bits_per_weight": self.bits_per_weight,
            "compression_ratio": self.compression_ratio,
        }


class Int6Quantizer(BaseQuantizer):
    """6-bit integer quantizer.

    Quantizes weights to 6-bit integers (0-63 range).
    """

    def __init__(self):
        super().__init__(bits=6)
        self.scale = 1.0
        self.zero_point = 0

    def quantize(self, weights: list[float], rows: int, cols: int) -> list[int]:
        """Quantize weights to 6-bit integers."""
        if not weights:
            return []

        # Find min/max for scaling
        min_val = min(weights)
        max_val = max(weights)

        # INT6 range: 0 to 63
        quant_min = 0
        quant_max = 63

        # Compute scale and zero point
        if max_val - min_val < 1e-8:
            self.scale = 1.0
            self.zero_point = 0
            return [0] * len(weights)

        self.scale = (max_val - min_val) / (quant_max - quant_min)
        self.zero_point = int(round(-min_val / self.scale + quant_min))

        # Quantize
        quantized = []
        for w in weights:
            q = int(round((w - min_val) / self.scale + quant_min))
            q = max(quant_min, min(quant_max, q))  # Clamp
            quantized.append(q)

        return quantized

    def dequantize(self, quantized: list[int], rows: int, cols: int) -> list[float]:
        """Dequantize 6-bit integers to weights."""
        if not quantized:
            return []

        min_val = -self.zero_point * self.scale

        dequantized = []
        for q in quantized:
            w = (q - self.zero_point) * self.scale
            dequantized.append(w)

        return dequantized


class Int5Quantizer(BaseQuantizer):
    """5-bit integer quantizer.

    Quantizes weights to 5-bit integers (0-31 range).
    More aggressive compression than INT6.
    """

    def __init__(self):
        super().__init__(bits=5)
        self.scale = 1.0
        self.zero_point = 0

    def quantize(self, weights: list[float], rows: int, cols: int) -> list[int]:
        """Quantize weights to 5-bit integers."""
        if not weights:
            return []

        min_val = min(weights)
        max_val = max(weights)

        # INT5 range: 0 to 31
        quant_min = 0
        quant_max = 31

        if max_val - min_val < 1e-8:
            self.scale = 1.0
            self.zero_point = 0
            return [0] * len(weights)

        self.scale = (max_val - min_val) / (quant_max - quant_min)
        self.zero_point = int(round(-min_val / self.scale + quant_min))

        quantized = []
        for w in weights:
            q = int(round((w - min_val) / self.scale + quant_min))
            q = max(quant_min, min(quant_max, q))
            quantized.append(q)

        return quantized

    def dequantize(self, quantized: list[int], rows: int, cols: int) -> list[float]:
        """Dequantize 5-bit integers to weights."""
        if not quantized:
            return []

        dequantized = []
        for q in quantized:
            w = (q - self.zero_point) * self.scale
            dequantized.append(w)

        return dequantized


class MixedQuantizer(BaseQuantizer):
    """Mixed INT5/INT6 quantizer.

    Uses INT5 for less sensitive weights and INT6 for more sensitive ones.
    Provides a balance between compression and accuracy.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__(bits=5)  # Average bits
        self.threshold = threshold  # Fraction of weights to quantize with INT6
        self.mask: list[bool] | None = None
        # Store scales and zero points for each weight individually
        self._scales: list[float] = []
        self._zero_points: list[int] = []

    @property
    def bits_per_weight(self) -> float:
        """Average bits per weight."""
        if self.mask:
            int6_ratio = sum(self.mask) / len(self.mask)
            return 5 * (1 - int6_ratio) + 6 * int6_ratio
        return 5.5

    def quantize(self, weights: list[float], rows: int, cols: int) -> list[int]:
        """Quantize weights using mixed precision.
        
        Encoding scheme:
        - Bit 7 (0x80) indicates INT6 (1) vs INT5 (0)
        - For INT6: bits 0-5 contain 6-bit value (0-63)
        - For INT5: bits 0-4 contain 5-bit value (0-31)
        Bits 6 and 5 (for INT5) are always 0.
        
        Each weight stores its own scale and zero_point for accurate dequantization.
        """
        if not weights:
            return []

        # Determine which weights get INT6 (more sensitive ones)
        # Use magnitude-based selection
        abs_weights = [abs(w) for w in weights]
        sorted_abs = sorted(abs_weights)
        threshold_idx = int(len(sorted_abs) * (1 - self.threshold))
        if threshold_idx < len(sorted_abs):
            magnitude_threshold = sorted_abs[threshold_idx]
        else:
            magnitude_threshold = 0

        # Create mask: True for INT6, False for INT5
        self.mask = [abs(w) >= magnitude_threshold for w in weights]
        self._scales = []
        self._zero_points = []

        # Quantize with appropriate precision
        quantized = []
        for i, w in enumerate(weights):
            if self.mask[i]:
                # Use INT6 for this weight (range 0-63)
                quant_max = 63
                # Compute scale and zero point for this specific weight
                min_val = max(w - abs(w) * 0.1, w - 1.0)  # Small range around weight
                max_val = min(w + abs(w) * 0.1, w + 1.0)
                if max_val - min_val < 1e-8:
                    scale = 1.0
                    zero_point = 0
                    q = 0
                else:
                    scale = (max_val - min_val) / quant_max
                    zero_point = int(round(-min_val / scale))
                    q = int(round((w - min_val) / scale))
                    q = max(0, min(quant_max, q))
                
                self._scales.append(scale)
                self._zero_points.append(zero_point)
                # Ensure q is in 0-63 range, set bit 7 as INT6 marker
                quantized.append((q & 0x3F) | 0x80)
            else:
                # Use INT5 for this weight (range 0-31)
                quant_max = 31
                # Compute scale and zero point for this specific weight
                min_val = max(w - abs(w) * 0.1, w - 1.0)
                max_val = min(w + abs(w) * 0.1, w + 1.0)
                if max_val - min_val < 1e-8:
                    scale = 1.0
                    zero_point = 0
                    q = 0
                else:
                    scale = (max_val - min_val) / quant_max
                    zero_point = int(round(-min_val / scale))
                    q = int(round((w - min_val) / scale))
                    q = max(0, min(quant_max, q))
                
                self._scales.append(scale)
                self._zero_points.append(zero_point)
                # Ensure q is in 0-31 range, bit 7 stays 0
                quantized.append(q & 0x1F)

        return quantized

    def dequantize(self, quantized: list[int], rows: int, cols: int) -> list[float]:
        """Dequantize mixed precision weights.
        
        Decoding scheme:
        - Check bit 7 (0x80) to determine INT6 vs INT5
        - For INT6: extract bits 0-5 (mask 0x3F)
        - For INT5: extract bits 0-4 (mask 0x1F)
        
        Uses stored scales and zero_points for accurate reconstruction.
        """
        if not quantized:
            return []

        if len(self._scales) != len(quantized) or len(self._zero_points) != len(quantized):
            raise ValueError(
                f"Scale/zero_point mismatch: got {len(self._scales)} scales, "
                f"{len(self._zero_points)} zero_points for {len(quantized)} quantized values"
            )

        dequantized = []
        for i, q in enumerate(quantized):
            scale = self._scales[i]
            zero_point = self._zero_points[i]
            
            if q & 0x80:  # INT6 (bit 7 set)
                # Extract 6-bit value from bits 0-5
                int6_val = q & 0x3F
                d = (int6_val - zero_point) * scale
            else:  # INT5 (bit 7 clear)
                # Extract 5-bit value from bits 0-4
                int5_val = q & 0x1F
                d = (int5_val - zero_point) * scale
            dequantized.append(d)

        return dequantized


class GPTQLiteQuantizer(BaseQuantizer):
    """Simplified GPTQ-style quantizer.

    Uses group-wise quantization with adaptive scaling.
    This is a simplified version of GPTQ for faster quantization.
    """

    def __init__(self, group_size: int = 128, bits: int = 4):
        super().__init__(bits=bits)
        self.group_size = group_size
        self.scales: list[float] = []
        self.zeros: list[float] = []

    def quantize(self, weights: list[float], rows: int, cols: int) -> list[int]:
        """Quantize weights using group-wise quantization."""
        if not weights:
            return []

        num_groups = (len(weights) + self.group_size - 1) // self.group_size
        self.scales = []
        self.zeros = []
        quantized = []

        # INT4 range: 0 to 15
        quant_min = 0
        quant_max = (1 << self.bits) - 1

        for g in range(num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, len(weights))
            group = weights[start:end]

            if not group:
                continue

            # Compute scale and zero for this group
            min_val = min(group)
            max_val = max(group)

            if max_val - min_val < 1e-8:
                scale = 1.0
                zero = 0
            else:
                scale = (max_val - min_val) / (quant_max - quant_min)
                zero = int(round(-min_val / scale + quant_min))

            self.scales.append(scale)
            self.zeros.append(zero)

            # Quantize group
            for w in group:
                q = int(round((w - min_val) / scale + quant_min))
                q = max(quant_min, min(quant_max, q))
                quantized.append(q)

        return quantized

    def dequantize(self, quantized: list[int], rows: int, cols: int) -> list[float]:
        """Dequantize group-wise quantized weights."""
        if not quantized:
            return []

        dequantized = []
        group_idx = 0

        for i, q in enumerate(quantized):
            if i > 0 and i % self.group_size == 0:
                group_idx += 1

            if group_idx >= len(self.scales):
                group_idx = len(self.scales) - 1

            scale = self.scales[group_idx]
            zero = self.zeros[group_idx]
            w = (q - zero) * scale
            dequantized.append(w)

        return dequantized


class QuantizerFactory:
    """Factory for creating quantizers."""

    @staticmethod
    def create(config: QuantizerConfig | dict[str, Any]) -> BaseQuantizer | None:
        """Create a quantizer from configuration.
        
        Returns None if quantization is not enabled.
        """
        if isinstance(config, dict):
            config = QuantizerConfig.from_dict(config)

        if not config.enabled:
            return None

        quant_type = config.type.lower()

        if quant_type == "int6":
            return Int6Quantizer()
        elif quant_type == "int5":
            return Int5Quantizer()
        elif quant_type == "int5_int6_mixed":
            return MixedQuantizer(threshold=0.2)
        elif quant_type == "gptq_lite":
            return GPTQLiteQuantizer(group_size=config.group_size, bits=4)
        else:
            raise ValueError(f"Unknown quantizer type: {quant_type}")


def create_quantizer(
    quant_type: str = "int6",
    enabled: bool = True,
    group_size: int = 128,
) -> BaseQuantizer | None:
    """Convenience function to create a quantizer."""
    config = QuantizerConfig(
        type=quant_type,
        enabled=enabled,
        group_size=group_size,
    )
    return QuantizerFactory.create(config)


def compute_quantization_metrics(
    quantizer: BaseQuantizer,
    weights: list[float],
    rows: int,
    cols: int,
) -> dict[str, Any]:
    """Compute comprehensive quantization metrics."""
    error_metrics = quantizer.quantize_error(weights, rows, cols)

    # Additional metrics
    original_size_bytes = len(weights) * 4  # fp32
    quantized_size_bytes = len(weights) * quantizer.bits_per_weight / 8

    return {
        **error_metrics,
        "original_size_bytes": original_size_bytes,
        "quantized_size_bytes": quantized_size_bytes,
        "size_reduction_ratio": original_size_bytes / max(quantized_size_bytes, 1),
        "actual_bits_per_weight": quantizer.bits_per_weight,
    }
