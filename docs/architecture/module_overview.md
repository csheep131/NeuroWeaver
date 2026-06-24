# Modul-Übersicht

**Letztes Update:** 2026-03-24
**Gesamtanzahl Module:** 11 Hauptmodule | 25+ Submodule

---

## Architektur-Übersicht

```
wettkampf/

core/ # Phase 1: Experiment Core
config.py # Config-Loading, YAML-Parsing
registry.py # Run-Registry, RunEntry
logging.py # Standardisiertes Logging
seed.py # Seed-Management (Reproduzierbarkeit)
artifacts.py # Artifact-Tracking, Size-Limits

models/
factories/ # Phase 2: Model Factories
backbone_factory.py # BackboneFactory, ArchitectureConfig
feature_gate.py # FeatureGate, FeatureGateManager

tokenizers/ # Phase 2: Tokenizer Lab
tokenizers.py # Byte, BigramHash, TrigramHash, Fallback

quant/ # Phase 2: Quantization Lab
quantizers.py # Int6Quantizer, Int5Quantizer, MixedQuantizer, GPTQLiteQuantizer

train/ # Phase 1: Training
trainer.py # Training-Loop
optimizer_factory.py # OptimizerFactory (Adam, AdamW, etc.)
scheduler.py # Learning Rate Scheduler
ema.py # Exponential Moving Average

eval/ # Phase 1: Evaluation
bpb_eval.py # BPB-Evaluation (Bits Per Byte)
sliding_window.py # Sliding Window Evaluation
benchmark.py # Performance Benchmarking

research/ # Phase 2: Research Engine
ablation_engine.py # AblationReporter, KillRules, KillReasons
phase1_evaluator.py # Phase1Evaluator, SuccessCriteria
phase2_evaluator.py # Phase2Evaluator, SuccessCriteria
phase3_evaluator.py # Phase3Evaluator, SuccessCriteria

orchestrator/ # Phase 3: Production Pipeline
sweep.py # SweepRunner, SweepConfig, SweepParameter
promote.py # PromotionSystem, Stage (Candidate/Promoted/Submitted)
submit_bundle.py # SubmissionBuilder, SubmissionBundle
dashboard.py # Dashboard CLI (interaktiv)
multi_seed.py # MultiSeedOrchestrator
combo_builder.py # DynamicComboBuilder, ComboConfig

reports/ # Phase 1: Reporting
compare_runs.py # RunComparator, RunComparison
leaderboard.py # LeaderboardGenerator

runs/ # Phase 1: Run-System
run.py # Haupt-Run-Logik
__main__.py # CLI-Entry-Point

rust-core/ # Rust-Core (Performance-kritisch)
src/lib.rs # Python-Bindings (PyO3)
src/tokenizers.rs # Rust-Tokenizer
src/quant.rs # Rust-Quantisierung
src/models.rs # Rust-Modelle
src/eval.rs # Rust-Evaluation

rust_core/ # Python-Bindings (auto-generiert)
```

---

## Module im Detail

### 1. core/ – Experiment Core (Phase 1)

**Zweck:** Basis-Infrastruktur für Config-Management, Run-Tracking, Logging

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `config.py` | `Config`, `load_config()`, `merge_configs()` | `yaml`, `hashlib`, `json` |
| `registry.py` | `RunRegistry`, `RunEntry`, `RunStatus` | `core.config`, `json`, `pathlib` |
| `logging.py` | `RunLogger`, `setup_logging()` | `json`, `datetime`, `pathlib` |
| `seed.py` | `set_seed()`, `SeedManager` | `random`, `numpy` |
| `artifacts.py` | `ArtifactTracker`, `ArtifactLimit` | `pathlib`, `os` |

**Data Flow:**
```
Config YAML → Config.load() → Config-Objekt
↓
RunRegistry.register()
↓
RunLogger.log_metrics()
↓
ArtifactTracker.save()
```

---

### 2. models/factories/ – Model Factories (Phase 2)

**Zweck:** Dynamische Modell-Erstellung mit Feature-Gates

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `backbone_factory.py` | `BackboneFactory`, `ArchitectureConfig` | `core.config`, `rust_core` (optional) |
| `feature_gate.py` | `FeatureGate`, `FeatureGateManager`, `FeatureDependency` | `core.config`, `typing` |

**Klassen-Hierarchie:**
```
FeatureGate
name: str
dependencies: List[FeatureDependency]
condition: Callable[[Config], bool]
apply(config: Config) → Config

FeatureGateManager
gates: Dict[str, FeatureGate]
register(gate: FeatureGate)
validate_all(config: Config) → bool
get_active_gates(config: Config) → List[str]
```

---

### 3. tokenizers/ – Tokenizer Lab (Phase 2)

**Zweck:** Experimentelle Tokenizer für Ablation-Testing

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `tokenizers.py` | `ByteTokenizer`, `BigramHashTokenizer`, `TrigramHashTokenizer`, `FallbackTokenizer` | `rust_core` (optional), `hashlib` |

**Tokenizer-Typen:**

| Typ | Beschreibung | Vocab Size |
|-----|--------------|------------|
| `ByteTokenizer` | Raw Byte-Encoding | 256 |
| `BigramHashTokenizer` | Hash-basierte Bigramme | Konfigurierbar |
| `TrigramHashTokenizer` | Hash-basierte Trigramme | Konfigurierbar |
| `FallbackTokenizer` | Fallback für OOV-Tokens | - |

**Rust-Integration:**
- `BigramHashTokenizer` → `rust_core.BigramHashTokenizer` (10x schneller)
- `TrigramHashTokenizer` → `rust_core.TrigramHashTokenizer` (8x schneller)

---

### 4. quant/ – Quantization Lab (Phase 2)

**Zweck:** Quantisierungs-Methoden für Modell-Kompression

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `quantizers.py` | `Int6Quantizer`, `Int5Quantizer`, `MixedQuantizer`, `GPTQLiteQuantizer` | `rust_core` (optional), `numpy` |

**Quantisierungs-Typen:**

| Typ | Bits | Werte-Bereich | Use Case |
|-----|------|---------------|----------|
| `Int6Quantizer` | 6 | 0-63 | Standard-Quantisierung |
| `Int5Quantizer` | 5 | 0-31 | Hohe Kompression |
| `MixedQuantizer` | 5/6 | 0-63 (mit Bit-Markierung) | Adaptive Präzision |
| `GPTQLiteQuantizer` | 4-8 | Variabel | GPTQ-ähnlich |

**Bit-Encoding (MixedQuantizer):**
```
INT6: Bit 7 = 1, Bits 0-5 = Wert (0-63)
INT5: Bit 7 = 0, Bits 0-4 = Wert (0-31)
```

---

### 5. train/ – Training (Phase 1)

**Zweck:** Training-Loop, Optimizer, Scheduler

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `trainer.py` | `Trainer`, `TrainingConfig` | `core.config`, `core.logging`, `eval.bpb_eval` |
| `optimizer_factory.py` | `OptimizerFactory`, `create_optimizer()` | `torch` (geplant) |
| `scheduler.py` | `LearningRateScheduler`, `CosineSchedule` | `math` |
| `ema.py` | `ExponentialMovingAverage` | `copy` |

**Training-Loop:**
```python
for step in range(num_steps):
loss = trainer.step(batch)
logger.log_metrics({"loss": loss, "step": step})
scheduler.step()
```

---

### 6. eval/ – Evaluation (Phase 1)

**Zweck:** Metrik-Berechnung, Benchmarking

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `bpb_eval.py` | `BPBEvaluator`, `compute_bpb()` | `math`, `numpy` |
| `sliding_window.py` | `SlidingWindowEvaluator` | `numpy` |
| `benchmark.py` | `Benchmark`, `measure_time()` | `time`, `statistics` |

**Metriken:**
- `val_bpb` – Validation Bits Per Byte
- `ms_per_step` – Millisekunden pro Training-Schritt
- `steps_completed` – Abgeschlossene Schritte
- `artifact_bytes` – Größe der Modell-Artefakte
- `quantized_val_bpb` – BPB nach Quantisierung
- `delta_bpb` – BPB-Änderung vs. Parent-Run

---

### 7. research/ – Research Engine (Phase 2)

**Zweck:** Ablation-Analyse, Kill-Rules, Phase-Evaluatoren

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `ablation_engine.py` | `AblationReporter`, `KillRule`, `KillReason` | `core.registry`, `core.config` |
| `phase1_evaluator.py` | `Phase1Evaluator`, `Phase1Criteria` | `eval.bpb_eval` |
| `phase2_evaluator.py` | `Phase2Evaluator`, `Phase2Criteria` | `quant.quantizers` |
| `phase3_evaluator.py` | `Phase3Evaluator`, `Phase3Criteria` | `orchestrator.promote` |

**Kill-Rules:**
```python
KillRule(
name="artifact_too_large",
condition=lambda m: m.get("artifact_bytes", 0) > 16_000_000,
reason=KillReason.ARTIFACT_SIZE,
)

KillRule(
name="slow_without_gain",
condition=lambda m: (
m.get("delta_ms") is not None
and m["delta_ms"] > 3.0
and m.get("delta_bpb") is not None
and m["delta_bpb"] > -0.05
),
reason=KillReason.PERFORMANCE,
)
```

---

### 8. orchestrator/ – Production Pipeline (Phase 3)

**Zweck:** Automation, Sweep-Execution, Promotion, Submission

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `sweep.py` | `SweepRunner`, `SweepConfig`, `SweepParameter` | `core.config`, `itertools` |
| `promote.py` | `PromotionSystem`, `Stage`, `StageConfig` | `core.registry`, `research.ablation_engine` |
| `submit_bundle.py` | `SubmissionBuilder`, `SubmissionBundle` | `core.registry`, `zipfile` |
| `dashboard.py` | `DashboardCLI`, `DashboardCommand` | `core.registry`, `readline` |
| `multi_seed.py` | `MultiSeedOrchestrator` | `runs.run`, `core.seed` |
| `combo_builder.py` | `DynamicComboBuilder`, `ComboConfig` | `core.config` |

**Stage-Enum:**
```python
class Stage(Enum):
CANDIDATE = "candidate"
PROMOTED = "promoted"
SUBMITTED = "submitted"
```

**Performance-Optimierungen:**
- `SweepRunner`: itertools.product (O(1) Memory)
- `PromotionSystem`: 3-Layer Caching (O(1) Lookups)
- `SubmissionBuilder`: Lazy Loading Cache

---

### 9. reports/ – Reporting (Phase 1)

**Zweck:** Run-Vergleiche, Leaderboards

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `compare_runs.py` | `RunComparator`, `RunComparison` | `core.registry`, `eval.bpb_eval` |
| `leaderboard.py` | `LeaderboardGenerator`, `Leaderboard` | `core.registry`, `tabulate` |

**Vergleichsmetriken:**
- `val_bpb` – Bits Per Byte (niedriger = besser)
- `ms_per_step` – Millisekunden pro Schritt (niedriger = besser)
- `steps_completed` – Abgeschlossene Schritte
- `artifact_bytes` – Modellgröße
- `quantized_val_bpb` – BPB nach Quantisierung
- `delta_bpb` – Änderung vs. Parent

---

### 10. runs/ – Run-System (Phase 1)

**Zweck:** Run-Execution, CLI

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `run.py` | `run_from_config()`, `execute_run()` | `core.config`, `core.registry`, `train.trainer` |
| `__main__.py` | CLI-Entry-Point | `argparse`, `run` |

**CLI-Usage:**
```bash
python -m runs.run --config configs/runs/run001_control.yaml
python -m runs.run --config configs/runs/run001_control.yaml --seed 42
```

---

### 11. rust-core/ – Rust-Core (Performance-kritisch)

**Zweck:** Performance-kritische Komponenten in Rust

| Modul | exports | Dependencies |
|-------|---------|--------------|
| `lib.rs` | Python-Bindings (PyO3) | `pyo3`, `tokenizers`, `quant`, `models`, `eval` |
| `tokenizers.rs` | `BigramHashTokenizer`, `TrigramHashTokenizer` | `pyo3`, `fxhash` |
| `quant.rs` | `Int6Quantizer`, `Int5Quantizer`, `MixedQuantizer` | `pyo3`, `ndarray` |
| `models.rs` | `RustBackbone` (Stub) | `pyo3` |
| `eval.rs` | `BPBEvaluator`, `SlidingWindowEvaluator` | `pyo3` |

**Build:**
```bash
cd rust-core
maturin develop --release
```

---

## Dependency-Graph

```

core/
(Config, Registry, Logging, Seed, Artifacts)




models/ train/ eval/
(Factories) (Trainer) (BPB, Bench)




research/
(Ablation Engine, Phase Evaluators)




orchestrator/
(Sweep, Promotion, Submission, Dashboard)




reports/
(Comparator, Leaderboard)



rust-core/
(Tokenizers, Quant, Models, Eval – Performance Layer)


(wird importiert von)

tokenizers/, quant/, eval/, models/factories/

```

---

## Externe Dependencies

| Paket | Version | Verwendung |
|-------|---------|------------|
| `pyyaml` | ≥6.0 | YAML-Config-Loading |
| `numpy` | ≥1.24 | Array-Operationen |
| `pyo3` | ≥0.18 | Rust/Python-Bindings |
| `maturin` | ≥1.0 | Rust-Build-System |
| `fxhash` | ≥0.2 | Fast Hashing (Rust) |
| `ndarray` | ≥0.15 | Tensor-Operationen (Rust) |
| `tabulate` | ≥0.9 | Leaderboard-Formatting |

---

## Verwandte Dokumente

- [rust_integration.md](rust_integration.md) – Rust-Integration Details
- [../api/modules.md](../api/modules.md) – Modul-API Übersicht
- [../reports/phase_2_audit.md](../reports/phase_2_audit.md) – Phase 2 Audit (Module)
