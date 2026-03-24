//! Model components for the ablation machine.
//!
//! Provides:
//! - Backbone architecture
//! - Recurrent blocks
//! - Attention mechanisms
//! - XSA (cross-attention)
//! - FiLM (feature-wise linear modulation)
//! - Activations

use pyo3::prelude::*;

/// Activation functions.
#[pyclass]
pub struct Activations;

#[pymethods]
impl Activations {
    /// Apply GELU activation.
    #[staticmethod]
    fn gelu(x: Vec<f32>) -> Vec<f32> {
        x.iter()
            .map(|&v| 0.5 * v * (1.0 + (v * 0.7978845608028654).tanh()))
            .collect()
    }

    /// Apply LeakyReLU activation.
    #[staticmethod]
    fn leaky_relu(x: Vec<f32>, negative_slope: f32) -> Vec<f32> {
        x.iter()
            .map(|&v| if v >= 0.0 { v } else { v * negative_slope })
            .collect()
    }

    /// Apply ReLU activation.
    #[staticmethod]
    fn relu(x: Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }

    /// Apply SiLU activation.
    #[staticmethod]
    fn silu(x: Vec<f32>) -> Vec<f32> {
        x.iter()
            .map(|&v| v / (1.0 + (-v).exp()))
            .collect()
    }
}

/// Multi-head attention implementation.
#[pyclass]
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

#[pymethods]
impl MultiHeadAttention {
    #[new]
    fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Get number of heads.
    #[getter]
    fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get head dimension.
    #[getter]
    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

/// Grouped Query Attention (GQA) implementation.
#[pyclass]
pub struct GroupedQueryAttention {
    num_query_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

#[pymethods]
impl GroupedQueryAttention {
    #[new]
    fn new(num_query_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_query_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Get number of query heads.
    #[getter]
    fn num_query_heads(&self) -> usize {
        self.num_query_heads
    }

    /// Get number of KV heads.
    #[getter]
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get head dimension.
    #[getter]
    fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get scale factor.
    #[getter]
    fn scale(&self) -> f32 {
        self.scale
    }
}

/// XSA (Cross-Sequence Attention) module.
#[pyclass]
pub struct XSAModule {
    enabled: bool,
    num_heads: usize,
}

#[pymethods]
impl XSAModule {
    #[new]
    #[pyo3(signature = (enabled = true, num_heads = 8))]
    fn new(enabled: bool, num_heads: usize) -> Self {
        Self { enabled, num_heads }
    }

    /// Check if XSA is enabled.
    #[getter]
    fn enabled(&self) -> bool {
        self.enabled
    }

    /// Get number of heads.
    #[getter]
    fn num_heads(&self) -> usize {
        self.num_heads
    }
}

/// FiLM (Feature-wise Linear Modulation) module.
#[pyclass]
pub struct FiLMModule {
    enabled: bool,
}

#[pymethods]
impl FiLMModule {
    #[new]
    #[pyo3(signature = (enabled = true))]
    fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Apply FiLM modulation.
    fn forward(&self, x: Vec<f32>, gamma: Vec<f32>, beta: Vec<f32>) -> Vec<f32> {
        if !self.enabled {
            return x;
        }

        x.iter()
            .zip(gamma.iter().zip(beta.iter()))
            .map(|(&xi, (&g, &b))| xi * g + b)
            .collect()
    }

    /// Check if FiLM is enabled.
    #[getter]
    fn enabled(&self) -> bool {
        self.enabled
    }
}

/// Recurrent block implementation.
#[pyclass]
pub struct RecurrentBlock {
    hidden_size: usize,
    tied: bool,
}

#[pymethods]
impl RecurrentBlock {
    #[new]
    #[pyo3(signature = (hidden_size, tied = false))]
    fn new(hidden_size: usize, tied: bool) -> Self {
        Self { hidden_size, tied }
    }

    /// Get hidden size.
    #[getter]
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Check if weights are tied.
    #[getter]
    fn tied(&self) -> bool {
        self.tied
    }
}

/// Backbone model configuration.
#[pyclass]
pub struct BackboneConfig {
    #[pyo3(get, set)]
    pub d_model: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub num_heads: usize,
    #[pyo3(get, set)]
    pub mlp_ratio: usize,
    #[pyo3(get, set)]
    pub max_seq_len: usize,
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    pub use_rope: bool,
    #[pyo3(get, set)]
    pub use_xsa: bool,
    #[pyo3(get, set)]
    pub use_film: bool,
}

impl Clone for BackboneConfig {
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            mlp_ratio: self.mlp_ratio,
            max_seq_len: self.max_seq_len,
            vocab_size: self.vocab_size,
            use_rope: self.use_rope,
            use_xsa: self.use_xsa,
            use_film: self.use_film,
        }
    }
}

#[pymethods]
impl BackboneConfig {
    #[new]
    #[pyo3(signature = (
        d_model = 512,
        num_layers = 6,
        num_heads = 8,
        mlp_ratio = 4,
        max_seq_len = 1024,
        vocab_size = 256,
        use_rope = true,
        use_xsa = false,
        use_film = false
    ))]
    fn new(
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        mlp_ratio: usize,
        max_seq_len: usize,
        vocab_size: usize,
        use_rope: bool,
        use_xsa: bool,
        use_film: bool,
    ) -> Self {
        Self {
            d_model,
            num_layers,
            num_heads,
            mlp_ratio,
            max_seq_len,
            vocab_size,
            use_rope,
            use_xsa,
            use_film,
        }
    }

    /// Get intermediate dimension (MLP hidden).
    fn d_ff(&self) -> usize {
        self.d_model * self.mlp_ratio
    }

    /// Get head dimension.
    fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Print config summary.
    fn summary(&self) -> String {
        format!(
            "BackboneConfig {{ d_model: {}, layers: {}, heads: {}, mlp_ratio: {}, seq_len: {}, vocab: {} }}",
            self.d_model, self.num_layers, self.num_heads, self.mlp_ratio, self.max_seq_len, self.vocab_size
        )
    }
}

/// Simple backbone model (placeholder for actual implementation).
#[pyclass]
pub struct Backbone {
    #[pyo3(get)]
    config: BackboneConfig,
}

#[pymethods]
impl Backbone {
    #[new]
    fn new(config: BackboneConfig) -> Self {
        Self { config }
    }

    /// Get number of parameters (estimate).
    fn num_parameters(&self) -> usize {
        // Rough estimate
        let embed_params = self.config.vocab_size * self.config.d_model;
        let attention_params = self.config.num_layers * 4 * self.config.d_model * self.config.d_model;
        let mlp_params = self.config.num_layers * 2 * self.config.d_model * self.config.d_ff();
        embed_params + attention_params + mlp_params
    }

    /// Get parameter count in millions.
    fn num_parameters_millions(&self) -> f32 {
        self.num_parameters() as f32 / 1_000_000.0
    }

    /// Print model summary.
    fn summary(&self) -> String {
        format!(
            "Backbone {{ params: {:.2}M, {} }}",
            self.num_parameters_millions(),
            self.config.summary()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu() {
        let input = vec![-1.0, 0.0, 1.0];
        let output = Activations::gelu(input.clone());
        assert_eq!(output.len(), input.len());
        assert!(output[0] < 0.0);
        assert_eq!(output[1], 0.0);
        assert!(output[2] > 0.0);
    }

    #[test]
    fn test_backbone_config() {
        let config = BackboneConfig::new(512, 6, 8, 4, 1024, 256, true, false, false);
        assert_eq!(config.d_ff(), 2048);
        assert_eq!(config.head_dim(), 64);
    }
}
