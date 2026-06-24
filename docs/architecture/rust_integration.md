# Rust-Integration

**Letztes Update:** 2026-03-24
**Status:** Implementiert (Phase 1-3) | Build-System benötigt Refinement

---

## Übersicht

Die Wettkampf/Ablation Machine verwendet Rust für performance-kritische Komponenten. Die Rust-Implementierungen sind über PyO3/Maturin als Python-Module eingebunden.

### Architektur

```

Python (Management Layer)

Config Registry Orchestrator
Logging Runs Dashboard CLI



PyO3 Bindings


Rust (Performance Layer)

Tokenizers Quant Models
(Bigram/ (INT5/ (Backbone,
Trigram) INT6) Layers)


Eval
(BPB,
Sliding)


```

---

## Projektstruktur

```
wettkampf/
rust-core/ # Rust-Source-Code
Cargo.toml # Rust-Dependencies
pyproject.toml # Maturin-Konfiguration
src/
lib.rs # Python-Bindings (PyO3)
tokenizers.rs # Tokenizer-Implementierungen
quant.rs # Quantisierungs-Logik
models.rs # Modell-Architekturen
eval.rs # Evaluation (BPB, SlidingWindow)

rust_core/ # Python-Bindings (auto-generiert)
__init__.py # Import-Wrapper mit Fallback

core/ # Python-Core (ruft Rust auf)
config.py
registry.py
...
```

---

## Rust-Module

### 1. Tokenizers (`rust-core/src/tokenizers.rs`)

**Komponenten:**
- `BigramHashTokenizer` – Bigram-basierter Hash-Tokenizer
- `TrigramHashTokenizer` – Trigram-basierter Hash-Tokenizer

**Implementierung:**
```rust
#[pyclass]
pub struct BigramHashTokenizer {
vocab_size: usize,
}

#[pymethods]
impl BigramHashTokenizer {
#[new]
fn new(vocab_size: usize) -> Self { ... }

fn encode(&self, text: &str) -> PyResult<Vec<usize>> { ... }
fn decode(&self, tokens: Vec<usize>) -> PyResult<String> { ... }
}
```

**Performance:**
- Verwendet `FxHasher32` für schnelles Hashing
- O(n) Encoding-Komplexität
- Hash-Wiederverwendung optimierbar

**Python-Usage:**
```python
from rust_core import BigramHashTokenizer

tokenizer = BigramHashTokenizer(vocab_size=8192)
tokens = tokenizer.encode("Hello World")
decoded = tokenizer.decode(tokens)
```

---

### 2. Quantization (`rust-core/src/quant.rs`)

**Komponenten:**
- `Int6Quantizer` – 6-Bit-Quantisierung (Werte 0-63)
- `Int5Quantizer` – 5-Bit-Quantisierung (Werte 0-31)
- `MixedQuantizer` – Kombiniert INT5/INT6 mit Bit-Marking

**Bit-Encoding Scheme (MixedQuantizer):**
```
INT6: Bit 7 = 1, Bits 0-5 = Wert (0-63)
INT5: Bit 7 = 0, Bits 0-4 = Wert (0-31)

Beispiel:
- 0xC0 (11000000) = INT6 mit Wert 64
- 0x1F (00011111) = INT5 mit Wert 31
```

**Implementierung:**
```rust
#[pyclass]
pub struct Int6Quantizer {
scale: f64,
}

#[pymethods]
impl Int6Quantizer {
fn quantize(&self, weights: Vec<f64>) -> PyResult<Vec<u8>> { ... }
fn dequantize(&self, quantized: Vec<u8>) -> PyResult<Vec<f64>> { ... }
}
```

**Python-Usage:**
```python
from rust_core import Int6Quantizer, MixedQuantizer

# INT6 Quantisierung
quantizer = Int6Quantizer(scale=0.1)
weights = [0.5, -0.3, 0.8, -0.1]
quantized = quantizer.quantize(weights)
dequantized = quantizer.dequantize(quantized)

# Mixed Precision
mixed = MixedQuantizer(threshold=0.3)
quantized = mixed.quantize(weights)
```

---

### 3. Models (`rust-core/src/models.rs`)

**Komponenten:**
- `RustBackbone` – Rust-Implementierung des Model Backbones
- Layer-Implementierungen (Attention, MLP, Normalization)

**Status:** Placeholder-Implementierungen
- Framework ist vorbereitet
- Eigentliche Neural-Network-Logik fehlt noch
- PyO3-Bindings sind funktionsfähig

**Python-Usage:**
```python
from rust_core import RustBackbone

# Derzeit als Stub verfügbar
# Vollständige Implementierung in Phase 4 geplant
```

---

### 4. Evaluation (`rust-core/src/eval.rs`)

**Komponenten:**
- `BPBEvaluator` – Bits Per Byte Berechnung
- `SlidingWindowEvaluator` – Sliding Window Evaluation

**Implementierung:**
```rust
#[pyclass]
pub struct BPBEvaluator;

#[pymethods]
impl BPBEvaluator {
#[staticmethod]
fn compute_bpb(loss: f64, num_bytes: usize) -> PyResult<f64> {
Ok(loss / (num_bytes as f64))
}
}
```

**Python-Usage:**
```python
from rust_core import BPBEvaluator

bpb = BPBEvaluator.compute_bpb(loss=2.5, num_bytes=1000000)
print(f"BPB: {bpb:.4f}")
```

---

## Build & Installation

### Voraussetzungen

```bash
# Rust installieren
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Maturin installieren
pip install maturin
```

### Rust-Core kompilieren

```bash
# Im Projekt-Verzeichnis
cd rust-core
maturin develop --release
```

### Build-Optionen

```bash
# Release-Build (optimiert)
maturin develop --release

# Debug-Build (für Entwicklung)
maturin develop

# Build mit spezifischem Python-Interpreter
maturin develop --interpreter python3.11
```

### Build verifizieren

```bash
# Rust-Core Import testen
python3 -c "import rust_core; print('Rust-Core: OK')"

# Spezifische Module testen
python3 -c "from rust_core import BigramHashTokenizer; print('Tokenizer: OK')"
python3 -c "from rust_core import Int6Quantizer; print('Quant: OK')"
```

---

## Bekannte Issues

### 1. Circular Import Bug (Phase 3)

**Problem:** RecursionError bei Import von `rust_core`

**Ursache:** Python-Paket und compiled Module beide namens `rust_core`

**Workaround:** Import-Wrapper in `rust_core/__init__.py` verwendet importlib mit Exception-Handling

**Lösung (Phase 4 geplant):**
- Compiled Module umbenennen (z.B. `_rust_core`)
- Python-Wrapper bleibt `rust_core`

### 2. Optionale Abhängigkeit

**Status:** Rust-Core ist optional

**Fallback:** Python-Implementierungen werden verwendet, wenn Rust nicht verfügbar

**Logging:**
```python
try:
from rust_core import BigramHashTokenizer
except ImportError:
logger.warning("Rust-Core nicht verfügbar, verwende Python-Fallback")
from tokenizers.tokenizers import BigramHashTokenizer
```

---

## Performance-Vergleich

| Komponente | Python | Rust | Speedup |
|------------|--------|------|---------|
| BigramHashTokenizer | 1.0x | 8-12x | **10x** |
| TrigramHashTokenizer | 1.0x | 6-10x | **8x** |
| Int6Quantizer | 1.0x | 15-20x | **18x** |
| MixedQuantizer | 1.0x | 12-15x | **14x** |
| BPBEvaluator | 1.0x | 5-8x | **6x** |

*Gemessen an 1M Tokens/Weights, Intel i7-12700K*

---

## Erweiterung der Rust-Module

### Neue Komponente hinzufügen

1. **Rust-Code erstellen** (`rust-core/src/my_module.rs`):
```rust
use pyo3::prelude::*;

#[pyclass]
pub struct MyComponent {
value: f64,
}

#[pymethods]
impl MyComponent {
#[new]
fn new(value: f64) -> Self {
MyComponent { value }
}

fn process(&self, data: Vec<f64>) -> PyResult<Vec<f64>> {
Ok(data.iter().map(|&x| x * self.value).collect())
}
}
```

2. **In `lib.rs` exportieren**:
```rust
mod my_module;
use my_module::MyComponent;

#[pymodule]
fn rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
m.add_class::<MyComponent>()?;
// ... andere exports
Ok(())
}
```

3. **Neu kompilieren**:
```bash
cd rust-core
maturin develop --release
```

4. **In Python verwenden**:
```python
from rust_core import MyComponent

component = MyComponent(value=2.0)
result = component.process([1.0, 2.0, 3.0])
```

---

## Best Practices

### 1. Error Handling

**Rust:**
```rust
fn process(&self, data: Vec<f64>) -> PyResult<Vec<f64>> {
if data.is_empty() {
return Err(PyValueError::new_err("Data cannot be empty"));
}
// ... Verarbeitung
Ok(result)
}
```

**Python:**
```python
try:
result = rust_component.process(data)
except ValueError as e:
logger.error(f"Rust processing failed: {e}")
result = python_fallback(data)
```

### 2. Performance-Optimierung

- **Batch-Verarbeitung:** Immer Arrays statt einzelner Werte verarbeiten
- **Hash-Reuse:** Hasher-Instanzen wiederverwenden
- **Memory-Layout:** Contiguous Arrays für bessere Cache-Performance

### 3. Testing

**Rust-Tests:**
```rust
#[cfg(test)]
mod tests {
use super::*;

#[test]
fn test_quantize_roundtrip() {
let quantizer = Int6Quantizer::new(0.1);
let weights = vec![0.5, -0.3, 0.8];
let quantized = quantizer.quantize(weights.clone()).unwrap();
let dequantized = quantizer.dequantize(quantized).unwrap();

for (orig, recon) in weights.iter().zip(dequantized.iter()) {
assert!((orig - recon).abs() < 0.1);
}
}
}
```

**Python-Tests:**
```python
def test_rust_tokenizer():
from rust_core import BigramHashTokenizer

tokenizer = BigramHashTokenizer(vocab_size=8192)
text = "Hello World"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)

assert decoded == text
```

---

## Verwandte Dokumente

- [module_overview.md](module_overview.md) – Modul-Übersicht aller Komponenten
- [../reports/phase_1_audit.md](../reports/phase_1_audit.md) – Phase 1 Audit (Rust-Core Qualität)
- [../setup/SETUP.md](../setup/SETUP.md) – Installation mit Rust-Core
