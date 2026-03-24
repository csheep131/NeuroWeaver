# API-Dokumentation

**Letztes Update:** 2026-03-24
**Status:** In Arbeit

---

## Übersicht

Dieses Verzeichnis enthält die API-Dokumentation der Ablation Machine. Die Dokumentation wird aus den Docstrings im Code generiert und manuell gepflegt.

---

## Module-Übersicht

### Core-Module

| Modul | Pfad | Beschreibung |
|-------|------|--------------|
| `Config` | `core/config.py` | Config-Loading, YAML-Parsing, Config-Validation |
| `RunRegistry` | `core/registry.py` | Run-Registry, Lineage-Tracking, Seed-Statistiken |
| `RunLogger` | `core/logging.py` | Buffered Logging, Metriken-Speicherung |
| `SeedManager` | `core/seed.py` | Seed-Management, Reproduzierbarkeit |
| `ArtifactTracker` | `core/artifacts.py` | Artifact-Tracking, Size-Reporting |

### Model-Module

| Modul | Pfad | Beschreibung |
|-------|------|--------------|
| `BackboneFactory` | `models/factories/backbone_factory.py` | Model-Builder, Recurrent Blocks, Feature-Gates |
| `FeatureGate` | `models/factories/feature_gate.py` | Feature-Gates, Dependency-Validation, Kill-Rules |

### Tokenizer-Module

| Modul | Pfad | Beschreibung |
|-------|------|--------------|
| `ByteTokenizer` | `tokenizers/tokenizers.py` | Byte-basierte Tokenisierung |
| `BigramHash` | `tokenizers/tokenizers.py` | Hash-basierte Bigram-Tokenisierung |
| `TrigramHash` | `tokenizers/tokenizers.py` | Hash-basierte Trigram-Tokenisierung |
| `FallbackTokenizer` | `tokenizers/tokenizers.py` | Byte-Fallback für Hash-Kollisionen |

### Quantisierungs-Module

| Modul | Pfad | Beschreibung |
|-------|------|--------------|
| `Int6Quantizer` | `quant/quantizers.py` | INT6-Quantisierung (6-bit) |
| `Int5Quantizer` | `quant/quantizers.py` | INT5-Quantisierung (5-bit) |
| `MixedQuantizer` | `quant/quantizers.py` | Gemischte Präzision (INT5/INT6) |
| `GPTQLiteQuantizer` | `quant/quantizers.py` | GPTQ-lite Quantisierung |

### Training-Module

| Modul | Pfad | Beschreibung |
|-------|------|--------------|
| `Trainer` | `train/trainer.py` | Training-Loop, Gradient Accumulation |
| `OptimizerFactory` | `train/optimizer_factory.py` | Optimizer-Erstellung (AdamW, Muon) |
| `Scheduler` | `train/scheduler.py` | Learning-Rate-Scheduling |
| `EMA` | `train/ema.py` | Exponential Moving Average |

### Evaluations-Module

| Modul | Pfad | Beschreibung |
|-------|------|--------------|
| `BPBEvaluator` | `eval/bpb_eval.py` | Bits-Per-Byte Evaluation |
| `SlidingWindow` | `eval/sliding_window.py` | Sliding-Window Evaluation |
| `Benchmark` | `eval/benchmark.py` | Performance-Benchmarking |

### Orchestrator-Module

| Modul | Pfad | Beschreibung |
|-------|------|--------------|
| `SweepRunner` | `orchestrator/sweep.py` | Parameter-Sweeps, Kombinationen |
| `PromotionSystem` | `orchestrator/promote.py` | Stage-Management, Promotion |
| `SubmissionBuilder` | `orchestrator/submit_bundle.py` | Submission-Bundle-Erstellung |
| `DashboardCLI` | `orchestrator/dashboard.py` | Interaktive Run-Übersicht |
| `MultiSeedOrchestrator` | `orchestrator/multi_seed.py` | Multi-Seed-Execution |
| `DynamicComboBuilder` | `orchestrator/combo_builder.py` | Dynamische Kombinationen |

### Research-Module

| Modul | Pfad | Beschreibung |
|-------|------|--------------|
| `AblationEngine` | `research/ablation_engine.py` | Ablation-Testing, Kill-Rules |
| `Phase1Evaluator` | `research/phase1_evaluator.py` | Phase-1-Evaluation |
| `Phase2Evaluator` | `research/phase2_evaluator.py` | Phase-2-Evaluation |
| `Phase3Evaluator` | `research/phase3_evaluator.py` | Phase-3-Evaluation |

### Rust-Core-Module

| Modul | Pfad | Beschreibung |
|-------|------|--------------|
| `rust_core.tokenizers` | `rust-core/src/tokenizers.rs` | Rust-Tokenizers (Performance) |
| `rust_core.quant` | `rust-core/src/quant.rs` | Rust-Quantisierung |
| `rust_core.models` | `rust-core/src/models.rs` | Rust-Modelle |
| `rust_core.eval` | `rust-core/src/eval.rs` | Rust-Evaluation |

---

## API-Referenz generieren

### Python Docstrings

Die API-Dokumentation kann mit folgenden Tools generiert werden:

```bash
# Mit pydoc
python -m pydoc orchestrator

# Mit Sphinx (empfohlen)
cd docs/api
sphinx-apidoc -o . ../../orchestrator ../../core ../../models

# Mit pdoc
pdoc orchestrator -o docs/api/
```

### Rust Documentation

```bash
# Rust-Docs generieren
cd rust-core
cargo doc --open

# Docs in Markdown konvertieren
cargo doc --no-deps --document-private-items
```

---

## Wichtige Klassen und Funktionen

### Config-Klasse

```python
class Config:
    """
    Lädt und validiert YAML-Konfigurationen.
    
    Attributes:
        run_id: Eindeutige Run-Identifikation
        seed: Random Seed für Reproduzierbarkeit
        model: Model-Konfiguration
        training: Training-Konfiguration
        tokenizer: Tokenizer-Konfiguration
        
    Example:
        >>> config = Config.from_yaml("configs/runs/run001_control.yaml")
        >>> print(config.run_id)
        'run001_control'
    """
```

### RunRegistry-Klasse

```python
class RunRegistry:
    """
    Zentrales Registry für alle Runs.
    
    Features:
    - O(1) Lookups by run_id
    - Lineage-Tracking (Parent-Child-Beziehungen)
    - Seed-Statistiken (Volatility Detection)
    - Lazy Loading from disk
    
    Example:
        >>> registry = RunRegistry()
        >>> entry = registry.get("run001_control")
        >>> print(entry.status)
        'completed'
    """
```

### SweepRunner-Klasse

```python
class SweepRunner:
    """
    Führt Parameter-Sweeps effizient aus.
    
    Verwendet itertools.product für O(1) Memory-Komplexität.
    
    Attributes:
        config: Sweep-Konfiguration
        registry: Run-Registry für Result-Tracking
        
    Example:
        >>> sweep = SweepRunner(config)
        >>> results = sweep.run_all()
        >>> print(f"Completed: {len(results)}")
    """
```

---

## Verwandte Dokumente

- [docs/README.md](../README.md) – Haupt-Dokumentationsübersicht
- [docs/guides/configuration.md](../guides/configuration.md) – Konfigurations-Handbuch
- [docs/architecture/module_overview.md](../architecture/module_overview.md) – Modul-Übersicht
