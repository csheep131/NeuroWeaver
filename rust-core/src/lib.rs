//! Rust core for the ablation machine.
//!
//! This crate provides high-performance implementations of:
//! - Tokenizers (byte, bigram hash, trigram hash)
//! - Quantization (int6, mixed precision)
//! - Model components (backbone, recurrent blocks, attention)

use pyo3::prelude::*;

pub mod tokenizers;
pub mod quant;
pub mod models;
pub mod eval;

pub use tokenizers::*;
pub use quant::*;
pub use models::*;
pub use eval::*;

/// Python module for rust-core
#[pymodule]
fn rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Tokenizers
    m.add_class::<tokenizers::ByteTokenizer>()?;
    m.add_class::<tokenizers::BigramHashTokenizer>()?;
    m.add_class::<tokenizers::TrigramHashTokenizer>()?;

    // Quantization
    m.add_class::<quant::Quantizer>()?;
    m.add_class::<quant::Int6Quantizer>()?;

    // Evaluation
    m.add_class::<eval::BPBComputer>()?;

    Ok(())
}
