# Entwickler-Guide

**Letztes Update:** 2026-03-24  
**Zielgruppe:** Entwickler, die am Wettkampf-Projekt arbeiten

---

## Übersicht

Dieser Guide richtet sich an Entwickler, die:
- Neue Features implementieren möchten
- Die Codebase verstehen wollen
- Tests schreiben und ausführen möchten
- Rust-Erweiterungen entwickeln wollen

---

## Entwicklungsumgebung einrichten

### 1. Repository klonen

```bash
git clone <repository-url>
cd wettkampf
```

### 2. Python-Virtual Environment

```bash
# Python 3.10+ empfohlen
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# oder
.venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt
```

### 3. Rust-Core kompilieren (optional, aber empfohlen)

```bash
# Rust installieren (falls nicht vorhanden)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Maturin installieren
pip install maturin

# Rust-Core im Development-Mode kompilieren
cd rust-core
maturin develop
```

### 4. Installation verifizieren

```bash
# Alle Module testen
python3 -c "from core import Config, RunRegistry; print('Core: OK')"
python3 -c "from orchestrator import SweepRunner, PromotionSystem; print('Orchestrator: OK')"
python3 -c "from research import AblationReporter; print('Research: OK')"
python3 -c "from quant import Int6Quantizer; print('Quant: OK')"

# Rust-Core testen (falls kompiliert)
python3 -c "import rust_core; print('Rust-Core: OK')"
```

---

## Projektstruktur verstehen

```
wettkampf/
├── configs/                   # YAML-Konfigurationen
│   ├── base.yaml              # Base-Konfiguration
│   └── runs/                  # Run-Konfigurationen (19)
│
├── core/                      # Phase 1: Experiment Core
│   ├── config.py              # Config-Loading
│   ├── registry.py            # Run-Registry
│   ├── logging.py             # Logging
│   ├── seed.py                # Seed-Management
│   └── artifacts.py           # Artifact-Tracking
│
├── models/factories/          # Phase 2: Model Factories
│   ├── backbone_factory.py    # BackboneFactory
│   └── feature_gate.py        # FeatureGate, FeatureGateManager
│
├── tokenizers/                # Phase 2: Tokenizer Lab
│   └── tokenizers.py          # Byte, BigramHash, TrigramHash
│
├── quant/                     # Phase 2: Quantization Lab
│   └── quantizers.py          # Int6Quantizer, Int5Quantizer, MixedQuantizer
│
├── train/                     # Phase 1: Training
│   ├── trainer.py             # Training-Loop
│   ├── optimizer_factory.py   # OptimizerFactory
│   ├── scheduler.py           # Learning Rate Scheduler
│   └── ema.py                 # Exponential Moving Average
│
├── eval/                      # Phase 1: Evaluation
│   ├── bpb_eval.py            # BPB-Evaluation
│   ├── sliding_window.py      # Sliding Window Evaluation
│   └── benchmark.py           # Performance Benchmarking
│
├── research/                  # Phase 2: Research Engine
│   ├── ablation_engine.py     # AblationReporter, KillRules
│   ├── phase1_evaluator.py    # Phase1Evaluator
│   ├── phase2_evaluator.py    # Phase2Evaluator
│   └── phase3_evaluator.py    # Phase3Evaluator
│
├── orchestrator/              # Phase 3: Production Pipeline
│   ├── sweep.py               # SweepRunner
│   ├── promote.py             # PromotionSystem
│   ├── submit_bundle.py       # SubmissionBuilder
│   ├── dashboard.py           # Dashboard CLI
│   ├── multi_seed.py          # MultiSeedOrchestrator
│   └── combo_builder.py       # DynamicComboBuilder
│
├── reports/                   # Phase 1: Reporting
│   ├── compare_runs.py        # RunComparator
│   └── leaderboard.py         # LeaderboardGenerator
│
├── runs/                      # Phase 1: Run-System
│   ├── run.py                 # Haupt-Run-Logik
│   └── __main__.py            # CLI-Entry-Point
│
├── rust-core/                 # Rust-Core (Performance-kritisch)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs             # Python-Bindings
│       ├── tokenizers.rs      # Rust-Tokenizer
│       ├── quant.rs           # Rust-Quantisierung
│       ├── models.rs          # Rust-Modelle
│       └── eval.rs            # Rust-Evaluation
│
├── rust_core/                 # Python-Bindings (auto-generiert)
├── results/                   # Ergebnisse (auto-generiert)
└── tests/                     # Tests (in Entwicklung)
```

---

## Entwicklung nach Phase

### Phase 1: Experiment Core

**Zuständige Module:** `core/`, `runs/`, `train/`, `eval/`

**Typische Aufgaben:**
- Neue Config-Parameter hinzufügen
- Logging-Format erweitern
- Run-Registry erweitern
- Training-Loop optimieren

**Beispiel: Neues Config-Parameter**

```yaml
# configs/base.yaml
model:
  d_model: 512
  num_layers: 6
  activation: "gelu"
  use_layer_norm: true  # NEU
```

```python
# core/config.py
class Config:
    @property
    def use_layer_norm(self) -> bool:
        return self._raw.get("model", {}).get("use_layer_norm", True)
```

---

### Phase 2: Research Engine

**Zuständige Module:** `models/factories/`, `tokenizers/`, `quant/`, `research/`

**Typische Aufgaben:**
- Neue Feature-Gates implementieren
- Tokenizer-Varianten hinzufügen
- Quantisierungs-Methoden erweitern
- Kill-Rules anpassen

**Beispiel: Neues Feature-Gate**

```python
# models/factories/feature_gate.py
from models.factories.feature_gate import FeatureGate, FeatureDependency

# Neues Feature-Gate für LayerNorm
layer_norm_gate = FeatureGate(
    name="layer_norm",
    dependencies=[
        FeatureDependency(
            name="use_layer_norm",
            condition=lambda c: c.get("model", {}).get("use_layer_norm", False),
            required=True,
        ),
    ],
    condition=lambda c: c.get("model", {}).get("use_layer_norm", False),
)

# Registrieren
FeatureGateManager.register(layer_norm_gate)
```

**Beispiel: Neuer Tokenizer**

```python
# tokenizers/tokenizers.py
class UnigramHashTokenizer:
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        # Unigram-Hash-Implementierung
        tokens = []
        for char in text:
            token = hash(char) % self.vocab_size
            tokens.append(token)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        # Placeholder-Decoding
        return "?" * len(tokens)
```

---

### Phase 3: Production Pipeline

**Zuständige Module:** `orchestrator/`, `reports/`

**Typische Aufgaben:**
- Sweep-Parameter erweitern
- Promotion-Stages anpassen
- Submission-Bundle formatieren
- Dashboard-Commands hinzufügen

**Beispiel: Neues Dashboard-Command**

```python
# orchestrator/dashboard.py
class DashboardCLI:
    def cmd_stats(self, args: list[str]) -> None:
        """Statistiken über alle Runs anzeigen"""
        runs = self.registry.get_all_runs()
        total = len(runs)
        completed = sum(1 for r in runs if r.status == "completed")
        failed = sum(1 for r in runs if r.status == "failed")

        print(f"Total Runs: {total}")
        print(f"Completed: {completed} ({completed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
```

---

## Rust-Entwicklung

### Neue Rust-Komponente hinzufügen

**1. Rust-Code erstellen** (`rust-core/src/my_module.rs`):

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

**2. In `lib.rs` exportieren**:

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

**3. Neu kompilieren**:

```bash
cd rust-core
maturin develop --release
```

**4. In Python verwenden**:

```python
from rust_core import MyComponent

component = MyComponent(value=2.0)
result = component.process([1.0, 2.0, 3.0])
```

### Rust-Tests schreiben

```rust
// rust-core/src/my_module.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process() {
        let component = MyComponent::new(2.0);
        let data = vec![1.0, 2.0, 3.0];
        let result = component.process(data).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }
}
```

**Tests ausführen**:

```bash
cd rust-core
cargo test
```

---

## Testing

### Tests ausführen

```bash
# Alle Tests
pytest tests/

# Spezifische Test-Suite
pytest tests/test_core.py
pytest tests/test_orchestrator.py
pytest tests/test_research.py

# Mit Coverage
pytest tests/ --cov=. --cov-report=html
```

### Module einzeln testen

```bash
# Core
python3 -c "from core import Config, RunRegistry; print('OK')"

# Orchestrator
python3 -c "from orchestrator import SweepRunner, PromotionSystem; print('OK')"

# Research
python3 -c "from research import AblationReporter; print('OK')"

# Quant
python3 -c "from quant import Int6Quantizer; q = Int6Quantizer(); print('OK')"

# Rust-Core (falls kompiliert)
python3 -c "import rust_core; print(f'Rust-Core: {rust_core.__version__}')"
```

### Beispiel: Unit-Test schreiben

```python
# tests/test_quantizers.py
import pytest
from quant.quantizers import Int6Quantizer, MixedQuantizer


class TestInt6Quantizer:
    def test_quantize_roundtrip(self):
        quantizer = Int6Quantizer(scale=0.1)
        weights = [0.5, -0.3, 0.8, -0.1]
        quantized = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized)

        for orig, recon in zip(weights, dequantized):
            assert abs(orig - recon) < 0.1

    def test_empty_weights(self):
        quantizer = Int6Quantizer(scale=0.1)
        quantized = quantizer.quantize([])
        assert quantized == []


class TestMixedQuantizer:
    def test_bit_encoding(self):
        quantizer = MixedQuantizer(threshold=0.3)
        weights = [0.1, -0.5, 0.9, -0.2, 0.7]
        quantized = quantizer.quantize(weights)

        # Alle Werte sollten im validen Bereich sein
        for q in quantized:
            assert 0 <= q <= 127  # 7-bit encoding
```

---

## Code-Quality

### Type Hints

Verwende moderne Type Hints (Python 3.10+):

```python
# Gut (modern)
def process(value: float) -> list[float]:
    ...

def get_config() -> Config | None:
    ...

# Veraltet (vermeiden)
from typing import Optional, List

def process(value: float) -> List[float]:
    ...

def get_config() -> Optional[Config]:
    ...
```

### Error Handling

```python
# Gut: Spezifische Exceptions
try:
    result = rust_component.process(data)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    return fallback_result
except ImportError as e:
    logger.warning(f"Rust module not available: {e}")
    return python_fallback(data)

# Schlecht: Catch-all
try:
    result = rust_component.process(data)
except Exception:
    pass
```

### Documentation

```python
class RunComparator:
    """Vergleicht Runs anhand von Metriken.

    Attributes:
        registry: RunRegistry-Instanz für Run-Zugriff
        metrics: Liste der zu vergleichenden Metriken

    Example:
        >>> comparator = RunComparator(registry)
        >>> comparison = comparator.compare_runs(["run001", "run002"])
        >>> print(comparator.print_summary(comparison))
    """

    def compare_runs(self, run_ids: list[str]) -> RunComparison:
        """Vergleicht mehrere Runs.

        Args:
            run_ids: Liste der Run-IDs zum Vergleichen

        Returns:
            RunComparison-Objekt mit Vergleichsergebnissen

        Raises:
            ValueError: Wenn eine Run-ID nicht existiert
        """
        ...
```

---

## Debugging

### Logging aktivieren

```python
# In der Config
import logging
logging.basicConfig(level=logging.DEBUG)

# Oder im Code
from core.logging import setup_logging
setup_logging(level="DEBUG")
```

### Rust-Core debuggen

```bash
# Debug-Build (mit Symbols)
cd rust-core
maturin develop

# Rust-Logs anzeigen
RUST_LOG=debug python3 -m runs.run --config configs/runs/run001_control.yaml
```

### Performance profilen

```bash
# Python-Profiler
python3 -m cProfile -o profile.stats -m runs.run --config configs/runs/run001_control.yaml

# Stats anzeigen
python3 -m pstats profile.stats
```

---

## Häufige Probleme

### Rust-Core wird nicht gefunden

```bash
# Lösung: Rust-Core neu kompilieren
cd rust-core
cargo clean
maturin develop --release
```

### Module nicht gefunden

```bash
# Lösung: PYTHONPATH setzen
export PYTHONPATH=/path/to/wettkampf:$PYTHONPATH

# Oder im Development-Mode installieren
pip install -e .
```

### Permission Errors bei results/

```bash
# Lösung: Verzeichnis-Rechte setzen
chmod -R 755 results/
```

### Circular Import Error

```python
# Lösung: Imports am Ende der Datei oder importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from other_module import OtherClass

def my_function(obj: "OtherClass"):
    ...
```

---

## Best Practices

### 1. Config-First Development

- Neue Features über Config-Parameter steuern
- Default-Werte in `configs/base.yaml`
- Overrides in Run-spezifischen Configs

### 2. Reproduzierbarkeit

- Immer Seeds setzen (`seed.py` verwenden)
- Config-Hash für Run-Tracking
- Deterministische Operationen bevorzugen

### 3. Performance

- Rust für performance-kritische Teile
- Batch-Verarbeitung statt Einzeloperationen
- Caching für wiederholte Berechnungen

### 4. Error Handling

- Spezifische Exception-Typen
- Logging mit Kontext-Information
- Graceful Fallbacks (Python statt Rust)

### 5. Testing

- Unit-Tests für neue Features
- Integration-Tests für Pipelines
- Property-based Tests für komplexe Logik

---

## Nächste Schritte

### Für neue Entwickler

1. [SETUP.md](SETUP.md) – Installation abschließen
2. [module_overview.md](../architecture/module_overview.md) – Modul-Struktur verstehen
3. Ersten Run starten: `python3 -m runs.run --config configs/runs/run001_control.yaml`
4. Dashboard ausprobieren: `python3 -m orchestrator.dashboard`
5. Eigene Config erstellen und testen

### Für erfahrene Entwickler

1. [rust_integration.md](../architecture/rust_integration.md) – Rust-Erweiterungen
2. [phase_3_audit.md](../reports/phase_3_audit.md) – Performance-Optimierungen
3. Neue Features für Phase 4 planen

---

## Verwandte Dokumente

- [SETUP.md](SETUP.md) – Installations-Anleitung
- [../architecture/module_overview.md](../architecture/module_overview.md) – Modul-Übersicht
- [../architecture/rust_integration.md](../architecture/rust_integration.md) – Rust-Integration
- [../reports/](../reports/) – Audit-Reports
