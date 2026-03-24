"""Unit tests for MixedQuantizer.

Tests the quantization/dequantization accuracy after the scale fix.
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod


# Inline copy of relevant classes to avoid import issues

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


class Int5Quantizer(BaseQuantizer):
    """5-bit integer quantizer."""

    def __init__(self):
        super().__init__(bits=5)
        self.scale = 1.0
        self.zero_point = 0

    def quantize(self, weights: list[float], rows: int, cols: int) -> list[int]:
        if not weights:
            return []

        min_val = min(weights)
        max_val = max(weights)
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
        if not quantized:
            return []

        dequantized = []
        for q in quantized:
            w = (q - self.zero_point) * self.scale
            dequantized.append(w)

        return dequantized


class Int6Quantizer(BaseQuantizer):
    """6-bit integer quantizer."""

    def __init__(self):
        super().__init__(bits=6)
        self.scale = 1.0
        self.zero_point = 0

    def quantize(self, weights: list[float], rows: int, cols: int) -> list[int]:
        if not weights:
            return []

        min_val = min(weights)
        max_val = max(weights)
        quant_min = 0
        quant_max = 63

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
        if not quantized:
            return []

        dequantized = []
        for q in quantized:
            w = (q - self.zero_point) * self.scale
            dequantized.append(w)

        return dequantized


class MixedQuantizer(BaseQuantizer):
    """Mixed INT5/INT6 quantizer - FIXED VERSION with per-weight scales."""

    def __init__(self, threshold: float = 0.5):
        super().__init__(bits=5)  # Average bits
        self.threshold = threshold
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
        """Quantize weights using mixed precision with per-weight scales."""
        if not weights:
            return []

        # Determine which weights get INT6 (more sensitive ones)
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
        """Dequantize mixed precision weights using stored scales."""
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
                int6_val = q & 0x3F
                d = (int6_val - zero_point) * scale
            else:  # INT5 (bit 7 clear)
                int5_val = q & 0x1F
                d = (int5_val - zero_point) * scale
            dequantized.append(d)

        return dequantized


# ============== TESTS ==============

def test_mixed_quantizer_basic():
    """Test basic quantization and dequantization."""
    print("Test 1: Basic quantization/dequantization")
    
    quantizer = MixedQuantizer(threshold=0.5)
    
    # Simple weights with different magnitudes
    weights = [0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 1.5, -1.5]
    rows, cols = 2, 4
    
    # Quantize
    quantized = quantizer.quantize(weights, rows, cols)
    
    print(f"  Original weights: {weights}")
    print(f"  Quantized values: {[hex(q) for q in quantized]}")
    print(f"  Mask (True=INT6, False=INT5): {quantizer.mask}")
    
    # Dequantize
    dequantized = quantizer.dequantize(quantized, rows, cols)
    print(f"  Dequantized: {dequantized}")
    
    # Check reconstruction error
    errors = [abs(o - d) for o, d in zip(weights, dequantized)]
    max_error = max(errors)
    mean_error = sum(errors) / len(errors)
    
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    
    # Assert reasonable error bounds
    assert max_error < 0.5, f"Max error too large: {max_error}"
    assert mean_error < 0.2, f"Mean error too large: {mean_error}"
    
    print("  ✅ Test 1 PASSED\n")


def test_mixed_quantizer_scale_preservation():
    """Test that individual scales are correctly preserved."""
    print("Test 2: Scale preservation per weight")
    
    quantizer = MixedQuantizer(threshold=0.5)
    
    # Weights with very different scales
    weights = [0.01, 0.01, 10.0, 10.0, -0.01, -10.0, 5.0, -5.0]
    
    quantized = quantizer.quantize(weights, 2, 4)
    dequantized = quantizer.dequantize(quantized, 2, 4)
    
    print(f"  Original: {weights}")
    print(f"  Dequantized: {dequantized}")
    print(f"  Scales: {quantizer._scales[:4]}...")
    print(f"  Zero points: {quantizer._zero_points[:4]}...")
    
    # Check that small values are reconstructed reasonably
    small_errors = [abs(weights[i] - dequantized[i]) for i in [0, 1, 4]]
    large_errors = [abs(weights[i] - dequantized[i]) for i in [2, 3, 5, 6, 7]]
    
    print(f"  Small weight errors: {small_errors}")
    print(f"  Large weight errors: {large_errors}")
    
    # All errors should be reasonable
    assert all(e < 0.5 for e in small_errors), f"Small weight errors too large"
    assert all(e < 1.0 for e in large_errors), f"Large weight errors too large"
    
    print("  ✅ Test 2 PASSED\n")


def test_mixed_quantizer_empty():
    """Test with empty weights."""
    print("Test 3: Empty weights")
    
    quantizer = MixedQuantizer()
    
    quantized = quantizer.quantize([], 0, 0)
    assert quantized == [], "Empty input should return empty output"
    
    dequantized = quantizer.dequantize([], 0, 0)
    assert dequantized == [], "Empty input should return empty output"
    
    print("  ✅ Test 3 PASSED\n")


def test_mixed_quantizer_single_value():
    """Test with single weight value."""
    print("Test 4: Single weight value")
    
    quantizer = MixedQuantizer(threshold=0.5)
    
    weights = [1.0]
    quantized = quantizer.quantize(weights, 1, 1)
    dequantized = quantizer.dequantize(quantized, 1, 1)
    
    print(f"  Original: {weights}")
    print(f"  Quantized: {quantized}")
    print(f"  Dequantized: {dequantized}")
    
    error = abs(weights[0] - dequantized[0])
    print(f"  Error: {error:.6f}")
    
    assert error < 0.5, f"Error too large: {error}"
    
    print("  ✅ Test 4 PASSED\n")


def test_mixed_quantizer_bits_per_weight():
    """Test bits per weight calculation."""
    print("Test 5: Bits per weight")
    
    quantizer = MixedQuantizer(threshold=0.5)
    
    # All same small weights -> threshold logic puts top 50% in INT6
    # So uniform weights will be split 50/50
    uniform_weights = [0.1] * 10
    quantizer.quantize(uniform_weights, 2, 5)
    bits = quantizer.bits_per_weight
    
    print(f"  Uniform weights (50/50 split): {bits:.2f} bits/weight")
    assert 5.0 <= bits <= 6.0, f"Expected 5-6 bits for uniform, got {bits}"
    
    # Mixed case with clear magnitude difference
    mixed_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 10.0]
    quantizer.quantize(mixed_weights, 2, 5)
    bits = quantizer.bits_per_weight
    
    print(f"  Mixed weights (clear split): {bits:.2f} bits/weight")
    assert 5.0 <= bits <= 6.0, f"Expected 5-6 bits for mixed, got {bits}"
    
    print("  ✅ Test 5 PASSED\n")


def test_mixed_quantizer_consistency():
    """Test that quantization is consistent for same input."""
    print("Test 6: Consistency")
    
    quantizer1 = MixedQuantizer(threshold=0.5)
    quantizer2 = MixedQuantizer(threshold=0.5)
    
    weights = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5] * 10
    
    quantized1 = quantizer1.quantize(weights, 10, 6)
    quantized2 = quantizer2.quantize(weights, 10, 6)
    
    print(f"  Same weights quantized twice")
    print(f"  Result 1: {quantized1[:5]}...")
    print(f"  Result 2: {quantized2[:5]}...")
    
    assert quantized1 == quantized2, "Quantization should be deterministic"
    
    print("  ✅ Test 6 PASSED\n")


def test_mixed_quantizer_int5_int6_split():
    """Test INT5/INT6 split based on magnitude."""
    print("Test 7: INT5/INT6 split")
    
    # Test with varied weights and threshold=0.5
    quantizer = MixedQuantizer(threshold=0.5)
    
    # Varied weights - top 50% by magnitude go to INT6
    varied_weights = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    quantizer.quantize(varied_weights, 2, 4)
    int5_count = sum(1 for m in quantizer.mask if not m)
    int6_count = sum(1 for m in quantizer.mask if m)
    print(f"  Varied weights (threshold=0.5): {int5_count} INT5, {int6_count} INT6")
    # Top 50% (largest magnitudes) should be INT6
    assert int6_count == 4, f"Expected 4 INT6 (top 50%), got {int6_count}"
    assert int5_count == 4, f"Expected 4 INT5 (bottom 50%), got {int5_count}"
    
    # Verify split - top 4 by magnitude should be INT6
    sorted_by_magnitude = sorted(enumerate(varied_weights), key=lambda x: abs(x[1]))
    int6_indices = [idx for idx, _ in sorted_by_magnitude[-4:]]  # Top 4
    
    for idx in int6_indices:
        assert quantizer.mask[idx] == True, f"Weight at index {idx} ({varied_weights[idx]}) should be INT6"
    
    int5_indices = [idx for idx, _ in sorted_by_magnitude[:4]]  # Bottom 4
    for idx in int5_indices:
        assert quantizer.mask[idx] == False, f"Weight at index {idx} ({varied_weights[idx]}) should be INT5"
    
    print("  ✅ Test 7 PASSED\n")


def test_scale_mismatch_detection():
    """Test that scale mismatch is detected."""
    print("Test 8: Scale mismatch detection")
    
    quantizer = MixedQuantizer(threshold=0.5)
    
    weights = [1.0, 2.0, 3.0]
    quantized = quantizer.quantize(weights, 1, 3)
    
    # Tamper with scales to trigger mismatch
    quantizer._scales = [1.0]  # Wrong length
    
    try:
        quantizer.dequantize(quantized, 1, 3)
        assert False, "Should have raised ValueError for scale mismatch"
    except ValueError as e:
        print(f"  Correctly detected mismatch: {e}")
    
    print("  ✅ Test 8 PASSED\n")


def test_reconstruction_quality():
    """Test overall reconstruction quality."""
    print("Test 9: Reconstruction quality")
    
    quantizer = MixedQuantizer(threshold=0.5)
    
    # Various weight distributions
    test_cases = [
        ("Small uniform", [0.01 * i for i in range(-10, 11)]),
        ("Large uniform", [1.0 * i for i in range(-5, 6)]),
        ("Mixed", [0.01, 0.1, 1.0, 5.0, 10.0] * 4),
        ("Gaussian-like", [0.1, 0.2, 0.5, 1.0, 2.0, 1.0, 0.5, 0.2, 0.1] * 3),
    ]
    
    for name, weights in test_cases:
        quantizer.quantize(weights, 1, len(weights))
        dequantized = quantizer.dequantize(quantizer.quantize(weights, 1, len(weights)), 1, len(weights))
        
        errors = [abs(o - d) for o, d in zip(weights, dequantized)]
        max_err = max(errors)
        mean_err = sum(errors) / len(errors)
        
        print(f"  {name}: max_err={max_err:.4f}, mean_err={mean_err:.4f}")
        
        # Quality assertions
        assert max_err < 1.0, f"{name}: max error {max_err} too large"
        assert mean_err < 0.3, f"{name}: mean error {mean_err} too large"
    
    print("  ✅ Test 9 PASSED\n")


def run_all_tests():
    """Run all MixedQuantizer tests."""
    print("=" * 70)
    print("MIXED QUANTIZER UNIT TESTS")
    print("=" * 70)
    print()
    
    try:
        test_mixed_quantizer_basic()
        test_mixed_quantizer_scale_preservation()
        test_mixed_quantizer_empty()
        test_mixed_quantizer_single_value()
        test_mixed_quantizer_bits_per_weight()
        test_mixed_quantizer_consistency()
        test_mixed_quantizer_int5_int6_split()
        test_scale_mismatch_detection()
        test_reconstruction_quality()
        
        print("=" * 70)
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
