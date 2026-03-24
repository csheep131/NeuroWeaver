//! Tokenizer implementations for the ablation machine.
//!
//! Provides fast tokenization with:
//! - Byte-level tokenization
//! - Bigram hash tokenization
//! - Trigram hash tokenization
//! - Fallback mechanisms

use pyo3::prelude::*;
use fxhash::FxHasher32;
use std::hash::Hasher;

/// Byte-level tokenizer.
///
/// This is the simplest tokenizer that works at the byte level.
#[pyclass]
pub struct ByteTokenizer {
    vocab_size: usize,
}

#[pymethods]
impl ByteTokenizer {
    #[new]
    #[pyo3(signature = (vocab_size = 256))]
    fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    /// Encode text to tokens.
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    /// Decode tokens to text.
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        let bytes: Result<Vec<u8>, _> = tokens
            .into_iter()
            .map(|t| {
                if t < 256 {
                    Ok(t as u8)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Invalid byte token: {}", t),
                    ))
                }
            })
            .collect();

        let bytes = bytes?;
        String::from_utf8(bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid UTF-8: {}", e)))
    }

    /// Get vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get token count for text.
    fn get_token_count(&self, text: &str) -> usize {
        text.len()
    }
}

/// Bigram hash tokenizer.
///
/// Uses hashing to represent bigrams in a fixed-size vocabulary.
#[pyclass]
pub struct BigramHashTokenizer {
    vocab_size: usize,
    byte_fallback: bool,
}

#[pymethods]
impl BigramHashTokenizer {
    #[new]
    #[pyo3(signature = (vocab_size = 4096, byte_fallback = true))]
    fn new(vocab_size: usize, byte_fallback: bool) -> Self {
        // Ensure vocab_size is greater than 256 for hash space
        let vocab_size = vocab_size.max(257);
        Self {
            vocab_size,
            byte_fallback,
        }
    }

    /// Encode text to tokens using bigram hashing.
    fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let bytes: Vec<u8> = text.bytes().collect();
        let mut tokens = Vec::with_capacity(bytes.len());

        // First byte
        if self.byte_fallback {
            tokens.push(bytes[0] as u32);
        } else {
            tokens.push(self.hash_bigram(0, bytes[0] as usize));
        }

        // Bigrams
        for i in 1..bytes.len() {
            let prev = bytes[i - 1] as usize;
            let curr = bytes[i] as usize;
            tokens.push(self.hash_bigram(prev, curr));
        }

        tokens
    }

    /// Decode tokens to text (best effort).
    fn decode(&self, tokens: Vec<u32>) -> String {
        // Decoding is approximate for hash-based tokenizers
        let mut result = String::new();
        for token in tokens {
            if token < 256 {
                // Byte fallback
                result.push(token as u8 as char);
            } else {
                // Can't decode hashed bigrams exactly - use replacement character
                result.push('?');
            }
        }
        result
    }

    /// Get vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get token count for text.
    fn get_token_count(&self, text: &str) -> usize {
        text.len()
    }
}

impl BigramHashTokenizer {
    fn hash_bigram(&self, prev: usize, curr: usize) -> u32 {
        let mut hasher = FxHasher32::default();
        hasher.write_u8(prev as u8);
        hasher.write_u8(curr as u8);
        let hash = hasher.finish();
        ((hash % (self.vocab_size - 256) as u64) + 256) as u32
    }
}

/// Trigram hash tokenizer.
///
/// Uses hashing to represent trigrams in a fixed-size vocabulary.
#[pyclass]
pub struct TrigramHashTokenizer {
    vocab_size: usize,
    byte_fallback: bool,
}

#[pymethods]
impl TrigramHashTokenizer {
    #[new]
    #[pyo3(signature = (vocab_size = 8192, byte_fallback = true))]
    fn new(vocab_size: usize, byte_fallback: bool) -> Self {
        Self {
            vocab_size,
            byte_fallback,
        }
    }

    /// Encode text to tokens using trigram hashing.
    fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let bytes: Vec<u8> = text.bytes().collect();
        let mut tokens = Vec::with_capacity(bytes.len());

        // First two bytes (fallback)
        if self.byte_fallback {
            for i in 0..bytes.len().min(2) {
                tokens.push(bytes[i] as u32);
            }
        }

        // Trigrams
        for i in 2..bytes.len() {
            let b1 = bytes[i - 2] as usize;
            let b2 = bytes[i - 1] as usize;
            let b3 = bytes[i] as usize;
            tokens.push(self.hash_trigram(b1, b2, b3));
        }

        tokens
    }

    /// Decode tokens to text (best effort).
    fn decode(&self, tokens: Vec<u32>) -> String {
        let mut result = String::new();
        for token in tokens {
            if token < 256 {
                result.push(token as u8 as char);
            } else {
                // Can't decode hashed trigrams exactly
                result.push('?');
            }
        }
        result
    }

    /// Get vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get token count for text.
    fn get_token_count(&self, text: &str) -> usize {
        text.len().saturating_sub(2)
    }
}

impl TrigramHashTokenizer {
    fn hash_trigram(&self, b1: usize, b2: usize, b3: usize) -> u32 {
        let mut hasher = FxHasher32::default();
        hasher.write_u8(b1 as u8);
        hasher.write_u8(b2 as u8);
        hasher.write_u8(b3 as u8);
        let hash = hasher.finish();
        ((hash % (self.vocab_size - 256) as u64) + 256) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_tokenizer() {
        let tokenizer = ByteTokenizer::new(256);
        let tokens = tokenizer.encode("hello");
        assert_eq!(tokens, vec![104, 101, 108, 108, 111]);
    }

    #[test]
    fn test_bigram_tokenizer() {
        let tokenizer = BigramHashTokenizer::new(4096, true);
        let tokens = tokenizer.encode("hello");
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], 104); // First byte is fallback
    }
}
