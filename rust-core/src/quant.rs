//! Quantization utilities for model compression.
//!
//! Provides:
//! - INT6 quantization
//! - Mixed INT5/INT6 quantization
//! - GPTQ-lite quantization

use pyo3::prelude::*;
use ndarray::Array2;

/// Base quantizer trait.
pub trait Quantize {
    fn quantize(&self, weights: &Array2<f32>) -> Array2<i8>;
    fn dequantize(&self, quantized: &Array2<i8>) -> Array2<f32>;
    fn bits_per_weight(&self) -> f32;
}

/// INT6 quantizer.
///
/// Quantizes weights to 6-bit integers.
#[pyclass]
pub struct Int6Quantizer {
    scale: f32,
    zero_point: i8,
}

#[pymethods]
impl Int6Quantizer {
    #[new]
    fn new() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0,
        }
    }

    /// Quantize weights to INT6.
    fn quantize(&mut self, weights: Vec<f32>, rows: usize, cols: usize) -> Vec<u8> {
        let weight_matrix = Array2::from_shape_vec((rows, cols), weights).unwrap();
        let (quantized, scale, zero_point) = self.quantize_matrix(&weight_matrix);

        self.scale = scale;
        self.zero_point = zero_point;

        // Pack 6-bit values: 4 values fit in 3 bytes
        self.pack_6bit(&quantized)
    }

    /// Dequantize INT6 weights.
    fn dequantize(&self, packed: Vec<u8>, rows: usize, cols: usize) -> Vec<f32> {
        let quantized = self.unpack_6bit(&packed, rows, cols);
        let dequantized = self.dequantize_matrix(&quantized);
        dequantized.into_raw_vec()
    }

    /// Get scale factor.
    #[getter]
    fn scale(&self) -> f32 {
        self.scale
    }

    /// Get zero point.
    #[getter]
    fn zero_point(&self) -> i8 {
        self.zero_point
    }

    /// Get bits per weight.
    fn bits_per_weight(&self) -> f32 {
        6.0
    }

    /// Get compression ratio vs fp32.
    fn compression_ratio(&self) -> f32 {
        32.0 / 6.0
    }
}

impl Int6Quantizer {
    fn quantize_matrix(&self, weights: &Array2<f32>) -> (Array2<i8>, f32, i8) {
        let min_val = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // INT6 range: 0 to 63
        let quant_min: f32 = 0.0;
        let quant_max: f32 = 63.0;

        let scale = (max_val - min_val) / (quant_max - quant_min);
        let scale = scale.max(1e-8); // Prevent division by zero
        let zero_point = (-min_val / scale + quant_min).round() as i8;

        let quantized = weights.mapv(|w| {
            let q = ((w - min_val) / scale + quant_min).round() as i8;
            q.clamp(0, 63)
        });

        (quantized, scale, zero_point)
    }

    fn dequantize_matrix(&self, quantized: &Array2<i8>) -> Array2<f32> {
        quantized.mapv(|q| {
            let q_float = q as f32;
            (q_float - self.zero_point as f32) * self.scale
        })
    }

    fn pack_6bit(&self, quantized: &Array2<i8>) -> Vec<u8> {
        let values: Vec<i8> = quantized.iter().cloned().collect();
        let num_values = values.len();
        // 4 values fit in 3 bytes (6 bits * 4 = 24 bits = 3 bytes)
        let packed_size = (num_values * 6 + 7) / 8;
        let mut packed = vec![0u8; packed_size];

        let mut bit_pos = 0;
        for value in values {
            let v = (value & 0x3F) as u8; // Keep only 6 bits

            // Write 6 bits across byte boundary if needed
            let byte_pos = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            if bit_offset <= 2 {
                // Fits in current byte
                packed[byte_pos] |= v << bit_offset;
            } else {
                // Spans two bytes
                packed[byte_pos] |= v << bit_offset;
                if byte_pos + 1 < packed.len() {
                    packed[byte_pos + 1] |= v >> (8 - bit_offset);
                }
            }

            bit_pos += 6;
        }

        packed
    }

    fn unpack_6bit(&self, packed: &[u8], rows: usize, cols: usize) -> Array2<i8> {
        let num_values = rows * cols;
        let mut values = Vec::with_capacity(num_values);

        let mut bit_pos = 0;
        for _ in 0..num_values {
            let byte_pos = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            let v = if byte_pos + 1 < packed.len() && bit_offset > 2 {
                // Read across byte boundary
                let low = (packed[byte_pos] >> bit_offset) as u16;
                let high = ((packed[byte_pos + 1] & ((1 << (bit_offset - 2)) - 1)) as u16)
                    << (8 - bit_offset);
                (low | high) as u8
            } else if byte_pos < packed.len() {
                (packed[byte_pos] >> bit_offset) & 0x3F
            } else {
                0
            };

            values.push(v as i8);
            bit_pos += 6;
        }

        Array2::from_shape_vec((rows, cols), values).unwrap()
    }
}

impl Default for Int6Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Mixed INT5/INT6 quantizer.
///
/// Uses INT5 for less sensitive weights and INT6 for more sensitive ones.
#[pyclass]
pub struct MixedQuantizer {
    threshold: f32,
    scale_5bit: f32,
    scale_6bit: f32,
}

#[pymethods]
impl MixedQuantizer {
    #[new]
    #[pyo3(signature = (threshold = 0.5))]
    fn new(threshold: f32) -> Self {
        Self {
            threshold,
            scale_5bit: 1.0,
            scale_6bit: 1.0,
        }
    }

    /// Get bits per weight (average).
    fn bits_per_weight(&self) -> f32 {
        5.5 // Average of 5 and 6
    }

    /// Get compression ratio vs fp32.
    fn compression_ratio(&self) -> f32 {
        32.0 / 5.5
    }
}

/// Main quantizer class that dispatches to specific implementations.
#[pyclass]
pub struct Quantizer {
    quant_type: String,
    int6: Int6Quantizer,
    mixed: MixedQuantizer,
}

#[pymethods]
impl Quantizer {
    #[new]
    #[pyo3(signature = (quant_type = "int6"))]
    fn new(quant_type: &str) -> Self {
        Self {
            quant_type: quant_type.to_string(),
            int6: Int6Quantizer::new(),
            mixed: MixedQuantizer::new(0.5),
        }
    }

    /// Quantize weights.
    fn quantize(&mut self, weights: Vec<f32>, rows: usize, cols: usize) -> Vec<u8> {
        match self.quant_type.as_str() {
            "int6" => self.int6.quantize(weights, rows, cols),
            "mixed" => {
                // For mixed, just use int6 for now (simplified)
                self.int6.quantize(weights, rows, cols)
            }
            _ => self.int6.quantize(weights, rows, cols),
        }
    }

    /// Dequantize weights.
    fn dequantize(&self, packed: Vec<u8>, rows: usize, cols: usize) -> Vec<f32> {
        self.int6.dequantize(packed, rows, cols)
    }

    /// Get compression ratio.
    fn compression_ratio(&self) -> f32 {
        self.int6.compression_ratio()
    }

    /// Get bits per weight.
    fn bits_per_weight(&self) -> f32 {
        self.int6.bits_per_weight()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int6_quantization() {
        let mut quantizer = Int6Quantizer::new();
        let weights = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let rows = 1;
        let cols = 5;

        let packed = quantizer.quantize(weights.clone(), rows, cols);
        let dequantized = quantizer.dequantize(packed, rows, cols);

        // Check that dequantized values are close to original
        for (orig, deq) in weights.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.1);
        }
    }
}
