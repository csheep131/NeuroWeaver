//! Evaluation utilities.
//!
//! Provides BPB (Bits Per Byte) computation and other metrics.

use pyo3::prelude::*;

/// BPB (Bits Per Byte) computer.
///
/// Computes the bits per byte metric for model evaluation.
#[pyclass]
pub struct BPBComputer {
    vocab_size: usize,
}

#[pymethods]
impl BPBComputer {
    #[new]
    #[pyo3(signature = (vocab_size = 256))]
    fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    /// Compute BPB from logits and target tokens.
    ///
    /// Args:
    ///     logits: Flattened logits array (batch * seq_len * vocab_size)
    ///     targets: Target token IDs (batch * seq_len)
    ///     batch_size: Batch size
    ///     seq_len: Sequence length
    ///
    /// Returns:
    ///     BPB value (float)
    #[pyo3(signature = (logits, targets, batch_size, seq_len))]
    fn compute_bpb(
        &self,
        logits: Vec<f32>,
        targets: Vec<u32>,
        batch_size: usize,
        seq_len: usize,
    ) -> PyResult<f32> {
        let vocab = self.vocab_size;
        let total_positions = batch_size * seq_len;

        if logits.len() != total_positions * vocab {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!(
                    "Logits length {} doesn't match expected {}",
                    logits.len(),
                    total_positions * vocab
                ),
            ));
        }

        if targets.len() != total_positions {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!(
                    "Targets length {} doesn't match expected {}",
                    targets.len(),
                    total_positions
                ),
            ));
        }

        let mut total_bits = 0.0;
        let mut total_bytes = 0;

        for i in 0..total_positions {
            let target = targets[i] as usize;
            if target >= vocab {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Target token {} exceeds vocab size {}", target, vocab),
                ));
            }

            // Get logits for this position
            let pos_logits = &logits[i * vocab..(i + 1) * vocab];

            // Compute log softmax and get probability of target
            let log_prob = self.log_softmax_target(pos_logits, target);

            // Bits = -log2(p) = -ln(p) / ln(2)
            let bits = -log_prob / std::f32::consts::LN_2;
            total_bits += bits;

            // Each token represents approximately 1 byte at byte level
            total_bytes += 1;
        }

        if total_bytes == 0 {
            return Ok(0.0);
        }

        Ok(total_bits / total_bytes as f32)
    }

    /// Compute BPB from loss value directly.
    ///
    /// If you already have cross-entropy loss (in nats), convert to BPB:
    /// BPB = loss / ln(2)
    #[staticmethod]
    fn bpb_from_loss(loss_nats: f32) -> f32 {
        loss_nats / std::f32::consts::LN_2
    }

    /// Compute theoretical minimum BPB for a given vocabulary.
    #[staticmethod]
    fn theoretical_minimum_bpb(vocab_size: usize) -> f32 {
        // With uniform distribution, each token carries log2(vocab_size) bits
        // If each token represents 1 byte, minimum BPB = 8 / log2(vocab_size)
        if vocab_size <= 1 {
            return f32::INFINITY;
        }
        8.0 / (vocab_size as f32).log2()
    }

    /// Get vocab size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl BPBComputer {
    fn log_softmax_target(&self, logits: &[f32], target: usize) -> f32 {
        // Find max for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute log-sum-exp
        let log_sum_exp: f32 = logits
            .iter()
            .map(|&l| ((l - max_logit).exp()))
            .sum::<f32>()
            .ln();

        // Log probability of target
        let log_prob = logits[target] - max_logit - log_sum_exp;

        log_prob
    }
}

/// Sliding window evaluator.
#[pyclass]
pub struct SlidingWindowEval {
    window_size: usize,
    stride: usize,
}

#[pymethods]
impl SlidingWindowEval {
    #[new]
    #[pyo3(signature = (window_size = 1024, stride = 512))]
    fn new(window_size: usize, stride: usize) -> Self {
        Self { window_size, stride }
    }

    /// Create sliding windows from text.
    fn create_windows(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut windows = Vec::new();

        let mut start = 0;
        while start < chars.len() {
            let end = (start + self.window_size).min(chars.len());
            let window: String = chars[start..end].iter().collect();
            windows.push(window);
            start += self.stride;

            if end >= chars.len() {
                break;
            }
        }

        windows
    }

    /// Get number of windows for a given text length.
    fn num_windows(&self, text_length: usize) -> usize {
        if text_length <= self.window_size {
            return 1;
        }
        (text_length - self.window_size + self.stride - 1) / self.stride + 1
    }

    /// Get window size.
    #[getter]
    fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get stride.
    #[getter]
    fn stride(&self) -> usize {
        self.stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpb_computer() {
        let computer = BPBComputer::new(256);

        // Simple test: uniform logits
        let logits = vec![1.0; 256];
        let targets = vec![0, 1, 2];
        let bpb = computer.compute_bpb(logits, targets, 1, 3).unwrap();

        // With uniform distribution, BPB should be 8.0 (log2(256))
        assert!((bpb - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_sliding_window() {
        let eval = SlidingWindowEval::new(10, 5);
        let text = "Hello, World! This is a test.";
        let windows = eval.create_windows(text);

        assert!(!windows.is_empty());
        assert!(windows.iter().all(|w| w.len() <= 10));
    }
}
