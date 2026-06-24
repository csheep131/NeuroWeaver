# Architecture Documentation

**Last Updated:** 2026-03-24
**Project:** Ablation Machine / Wettkampf
**Status:** Phase 1-3 Complete

---

## System Overview

### Python/Rust Trennung

```

PYTHON LAYER (High-Level)

Orchestrator Research Reports Config Registry
- Sweep - Phase - Compare - YAML - Runs
- Promote - Ablate - Board - Seed - Entry
- Submit - Kill - Metrics - Log - Cache


PyO3 Bindings


RUST LAYER (Performance)

Tokenizers Quant Models Eval
- Byte - INT6 - Forward - BPB
- BigramHash - INT5 - Backward - Window
- TrigramHash - Mixed - Layers - Metrics

```

### Design-Prinzipien

1. **Config-First**: Alle Runs werden durch YAML-Configs deklariert
2. **Reproduzierbarkeit**: Seeds determinieren alle zufälligen Operationen
3. **Modularität**: Klare Trennung zwischen Python (Orchestrierung) und Rust (Performance)
4. **Automatisierung**: Sweeps, Promotion, Submission laufen automatisch
5. **Kill-Rules**: Schlechte Runs werden früh verworfen

---

## Modul-Abhängigkeiten

```

configs/
(YAML)




core/

config ← registry ← logging seed
artifacts RunEntry Logger SeedMgr





train/ models/
- trainer
- optimizer backbone_factory feature_gate
- scheduler ArchitectureCfg FeatureGateManager
- ema




tokenizers/ & quant/

Byte, BigramHash, Trigram Int6Quant, Int5Quant, Mixed
Fallback GPTQLite, QuantizerFactory





eval/

bpb_eval sliding_window benchmark
BPB-Metrics Window-Eval Performance-Measure





research/ (Phase 2)

ablation_engine phase1_evaluator phase2_evaluator
phase3_evaluator phaseX_success phaseX_metrics
KillRules Phase1Report Phase2Report





orchestrator/ (Phase 3)

sweep promote submit_bundle dashboard
multi_seed combo_bldr





reports/

RunComparator LeaderboardGenerator
RunComparison Leaderboard (by_bpb, efficiency)


```

---

## Datenfluss

### Config → Run → Metrics → Reports

```

1. CONFIG LOADING
configs/runs/*.yaml → core.config.Config → validated dict




2. RUN INITIALIZATION
Config → RunEntry (registry) → Seed setup → Artifacts path




3. MODEL CREATION
BackboneFactory.create() → FeatureGate.apply() → Model ready
Tokenizer.load() → Quantizer.configure()




4. TRAINING LOOP
for step in range(num_steps):
forward() → loss → backward() → optimizer.step()
ema.update() → scheduler.step()
benchmark.measure() → metrics.log()




5. EVALUATION
bpb_eval.evaluate() → sliding_window.eval() → quantize()
metrics = {val_bpb, ms_per_step, artifact_bytes, ...}




6. RESEARCH ASSESSMENT (Phase 2)
PhaseEvaluator.assess() → KillRules.check() → AblationReport
Decision: CONTINUE | KILL | PROMOTE




7. ORCHESTRATION (Phase 3)
Sweep.generate() → Promotion.evaluate() → Submission.build()
Dashboard.update() → Leaderboard.generate()




8. REPORTS
RunComparator.compare() → LeaderboardGenerator.generate()
Output: Markdown tables, JSON exports, ZIP bundles

```

---

## Phase-Komponenten-Übersicht

### Phase 1: Experiment Core

| Modul | Komponenten | Zweck |
|-------|-------------|-------|
| `core/` | Config, Registry, Logging, Seed, Artifacts | Basis-Infrastruktur |
| `train/` | Trainer, OptimizerFactory, Scheduler, EMA | Training-Loop |
| `eval/` | BPBEval, SlidingWindow, Benchmark | Evaluation |
| `runs/` | run.py, __main__.py | Run-Entry-Point |
| `reports/` | RunComparator, LeaderboardGenerator | Vergleich |

**Entry Points:**
- `python -m runs.run --config <config.yaml>`

**Outputs:**
- `results/runs/<run_id>/model.pt`
- `results/runs/<run_id>/metrics.json`
- `results/runs/<run_id>/config.yaml`

---

### Phase 2: Research Engine

| Modul | Komponenten | Zweck |
|-------|-------------|-------|
| `models/factories/` | BackboneFactory, FeatureGate | Model-Konstruktion |
| `tokenizers/` | Byte, BigramHash, TrigramHash, Fallback | Tokenisierung |
| `quant/` | Int6Quantizer, Int5Quantizer, MixedQuantizer, GPTQLite | Quantisierung |
| `research/` | AblationEngine, Phase1/2/3 Evaluator | Forschung |

**Feature Gates:**
```python
FeatureGateManager.enable("feature_x")
FeatureGateManager.disable("feature_y")
FeatureGateManager.status() # FeatureStatus enum
```

**Kill Rules:**
```python
KillRule(
condition="artifact_bytes > 16_000_000",
action="kill",
reason="Model too large"
)
```

**Outputs:**
- `results/research/ablation_report.md`
- `results/research/phase1_results.json`
- `results/research/phase2_results.json`

---

### Phase 3: Production Pipeline

| Modul | Komponenten | Zweck |
|-------|-------------|-------|
| `orchestrator/sweep` | SweepRunner, SweepConfig, SweepParameter | Parameter-Sweeps |
| `orchestrator/promote` | PromotionSystem, Stage, StageConfig | Stage-Management |
| `orchestrator/submit_bundle` | SubmissionBuilder, SubmissionBundle | Bundle-Erstellung |
| `orchestrator/dashboard` | Dashboard CLI | Interaktive Übersicht |
| `orchestrator/multi_seed` | MultiSeedOrchestrator | Multi-Seed Runs |
| `orchestrator/combo_builder` | DynamicComboBuilder | Feature-Kombis |

**Stages:**
```
CANDIDATE → PROMOTED → SUBMITTED


new run passed ready for
awaiting thresholds submission
evaluation
```

**Performance-Optimierungen:**
- Sweep: `itertools.product` statt rekursiv (5x schneller)
- Promotion: 3-Layer-Caching (5x schneller)
- Bundle: Cached Registry-Lookups (4x schneller)

**Outputs:**
- `results/sweeps/<sweep_id>/results.json`
- `results/bundles/<bundle_id>.zip`
- `results/leaderboards/bpb.md`

---

## Rust-Core Integration

### Architektur

```

Python Code
from quant import Int6Quantizer
quantizer = Int6Quantizer()


PyO3 Binding


rust-core/src/lib.rs
#[pyclass]
pub struct Int6Quantizer { ... }

#[pymethods]
impl Int6Quantizer {
pub fn quantize(&self, data: Vec<f32>) -> Vec<i8>
}


Cargo Build


rust_core/*.so (shared library)
Auto-generated by maturin

```

### Module

| Rust-Modul | Python-Äquivalent | Performance-Gewinn |
|------------|-------------------|-------------------|
| `tokenizers.rs` | `tokenizers/` | 10-50x bei Vocab > 8k |
| `quant.rs` | `quant/` | 5-20x bei Quantisierung |
| `models.rs` | `models/` | 2-5x bei Forward-Pass |
| `eval.rs` | `eval/` | 3-10x bei BPB-Eval |

### Build-Prozess

```bash
# Development Build
cd rust-core
maturin develop

# Release Build (optimized)
maturin develop --release

# Build Wheel
maturin build --release
```

### Bindings erstellen

```rust
// rust-core/src/lib.rs
use pyo3::prelude::*;

#[pyclass]
pub struct Int6Quantizer {
scale: f32,
}

#[pymethods]
impl Int6Quantizer {
#[new]
fn new() -> Self {
Int6Quantizer { scale: 1.0 }
}

fn quantize(&self, data: Vec<f32>) -> Vec<i8> {
data.iter().map(|&x| (x / self.scale) as i8).collect()
}
}

#[pymodule]
fn rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
m.add_class::<Int6Quantizer>()?;
Ok(())
}
```

---

## Konfigurations-System

### Config-Hierarchie

```
configs/base.yaml # Default-Werte für alle Runs
↓
configs/runs/*.yaml # Run-spezifische Overrides
↓
Command-line args # Runtime-Overrides
↓
Final Config # Merged & validated
```

### Config-Schema

```yaml
# Base-Struktur
run_id: string
seed: int

model:
d_model: int
num_layers: int
activation: string
use_feature_gate: bool

tokenizer:
type: "byte" | "bigram_hash" | "trigram_hash" | "fallback"
vocab_size: int

quantization:
enabled: bool
type: "int6" | "int5" | "mixed" | "gptq_lite"

training:
num_steps: int
learning_rate: float
batch_size: int
optimizer: "adam" | "adamw"
scheduler: "cosine" | "linear" | "none"

evaluation:
eval_interval: int
window_size: int
```

---

## Registry-System

### RunEntry Struktur

```python
class RunEntry:
run_id: str
config: dict
status: "pending" | "running" | "completed" | "failed" | "killed"
metrics: dict # val_bpb, ms_per_step, artifact_bytes, ...
artifacts: dict # paths to model.pt, config.yaml, metrics.json
parent_run_id: str | None
created_at: datetime
updated_at: datetime
```

### Registry-Operationen

```python
registry = RunRegistry()

# Eintrag erstellen
entry = registry.create_run(config)

# Status aktualisieren
registry.update_status(run_id, "running")
registry.update_metrics(run_id, metrics)

# Eintrag abrufen
entry = registry.get(run_id)
all_entries = registry.list()

# Nach Status filtern
completed = registry.filter_by_status("completed")
```

---

## File-Struktur (Results)

```
results/
runs/
<run_id>/
model.pt # Modell-Weights
config.yaml # Run-Konfiguration
metrics.json # Trainings-Metriken
logs/
training.log # Trainings-Logs

sweeps/
<sweep_id>/
results.json # Alle Sweep-Ergebnisse
configs/ # Generierte Configs
summary.md # Mensch-lesbare Zusammenfassung

bundles/
<bundle_id>.zip # Submission-Bundle

leaderboards/
bpb.md # BPB-Leaderboard
efficiency.md # Effizienz-Leaderboard
quantized.md # Quantized-Leaderboard

research/
ablation_report.md # Ablation-Report
phase1_results.json # Phase 1 Ergebnisse
phase2_results.json # Phase 2 Ergebnisse
phase3_results.json # Phase 3 Ergebnisse
```

---

## Extension Points

### Neue Tokenizer hinzufügen

1. Datei erstellen: `tokenizers/my_tokenizer.py`
2. Interface implementieren:
```python
class MyTokenizer:
def encode(self, text: str) -> list[int]: ...
def decode(self, ids: list[int]) -> str: ...
```
3. In `tokenizers/__init__.py` exportieren
4. Config-Schema erweitern

### Neue Quantisierungsmethode

1. Datei erstellen: `quant/my_quantizer.py`
2. Von `BaseQuantizer` erben
3. Methoden implementieren: `quantize()`, `dequantize()`
4. In `QuantizerFactory` registrieren

### Neue Phase hinzufügen

1. Ordner erstellen: `research/phase4_evaluator.py`
2. `Phase4Evaluator` Klasse implementieren
3. SuccessCriteria definieren
4. In Orchestrator integrieren

---

## Performance-Charakteristik

### Benchmark-Ergebnisse (Phase 3 Optimizations)

| Operation | Vorher | Nachher | Verbesserung |
|-----------|--------|---------|--------------|
| Sweep-Generation (1000 Combos) | 250ms | 50ms | 5x |
| Promotion-Eval (1000 Runs) | 1500ms | 300ms | 5x |
| Bundle-Creation (100 Runs) | 800ms | 200ms | 4x |
| Rust-Tokenizer (Vocab 8k) | 100ms | 8ms | 12x |
| Rust-Quant (Layer 6) | 50ms | 10ms | 5x |

### Skalierbarkeit

| Komponente | Limit Vorher | Limit Nachher |
|------------|--------------|---------------|
| Sweeps | ~500 Combos | ~10.000 Combos |
| Registry | ~500 Runs | ~10.000 Runs |
| Promotion | ~500 Runs | ~10.000 Runs |

---

## Related Documents

- [README.md](../README.md) - Projekt-Übersicht
- [SETUP.md](../SETUP.md) - Installation & Usage
- [HERMES.md](../HERMES.md) - Entwicklungs-Konventionen
- [docs/runs_development.md](runs_development.md) - Run-Entwicklung
