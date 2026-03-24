# Setup Guide

## Installation

### 1. Python-Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 2. Rust-Core kompilieren (empfohlen für Performance)

```bash
# Im Projekt-Verzeichnis
maturin develop --release
```

Falls `maturin` nicht installiert ist:

```bash
pip install maturin
```

### 3. Installation verifizieren

```bash
# Python-Module testen
python3 -c "from core import Config, RunRegistry; print('Core: OK')"
python3 -c "from orchestrator import SweepRunner, PromotionSystem; print('Orchestrator: OK')"
python3 -c "from research import AblationReporter; print('Research: OK')"
python3 -c "from quant import Int6Quantizer, MixedQuantizer; print('Quant: OK')"

# Rust-Core testen (falls kompiliert)
python3 -c "import rust_core; print('Rust-Core: OK')"
```

---

## Projektstruktur

```
wettkampf/
├── configs/
│   ├── base.yaml              # Base-Konfiguration
│   └── runs/                  # 19 Run-Konfigurationen
│       ├── run001_control.yaml
│       ├── run002_hash.yaml
│       ├── run003_xsa.yaml
│       ├── run004_leakyrelu.yaml
│       ├── run005_mixed_quant.yaml
│       ├── run006_film.yaml
│       ├── run007_ttt.yaml
│       ├── run008a_star_relu.yaml
│       ├── run009_gqa.yaml
│       ├── run010_recurrence.yaml
│       ├── run016_best_combo_a.yaml
│       ├── run017_best_combo_quantized.yaml
│       ├── run018_control_s1.yaml
│       ├── run019_control_s2.yaml
│       └── run020_control_s3.yaml
│
├── core/                      # Python-Core (Phase 1)
│   ├── __init__.py
│   ├── config.py              # Config-Loading, YAML-Parsing
│   ├── registry.py            # Run-Registry, RunEntry
│   ├── logging.py             # Standardisiertes Logging
│   ├── seed.py                # Seed-Management (Reproduzierbarkeit)
│   └── artifacts.py           # Artifact-Tracking, Size-Limits
│
├── models/
│   └── factories/             # Model Factories (Phase 2)
│       ├── __init__.py
│       ├── backbone_factory.py    # BackboneFactory, ArchitectureConfig
│       └── feature_gate.py        # FeatureGate, FeatureGateManager
│
├── tokenizers/                # Tokenizer (Phase 2)
│   ├── __init__.py
│   └── tokenizers.py          # Byte, BigramHash, TrigramHash, Fallback
│
├── quant/                     # Quantization (Phase 2)
│   ├── __init__.py
│   └── quantizers.py          # Int6Quantizer, Int5Quantizer, MixedQuantizer, GPTQLiteQuantizer
│
├── train/                     # Training (Phase 1)
│   ├── __init__.py
│   ├── trainer.py             # Training-Loop
│   ├── optimizer_factory.py   # OptimizerFactory (Adam, AdamW, etc.)
│   ├── scheduler.py           # Learning Rate Scheduler
│   └── ema.py                 # Exponential Moving Average
│
├── eval/                      # Evaluation (Phase 1)
│   ├── __init__.py
│   ├── bpb_eval.py            # BPB-Evaluation (Bits Per Byte)
│   ├── sliding_window.py      # Sliding Window Evaluation
│   └── benchmark.py           # Performance Benchmarking
│
├── research/                  # Research Engine (Phase 2)
│   ├── __init__.py
│   ├── ablation_engine.py     # AblationReporter, KillRules, KillReasons
│   ├── phase1_evaluator.py    # Phase1Evaluator, SuccessCriteria
│   ├── phase2_evaluator.py    # Phase2Evaluator, SuccessCriteria
│   └── phase3_evaluator.py    # Phase3Evaluator, SuccessCriteria
│
├── orchestrator/              # Production Pipeline (Phase 3)
│   ├── __init__.py
│   ├── sweep.py               # SweepRunner, SweepConfig, SweepParameter
│   ├── promote.py             # PromotionSystem, Stage (Candidate/Promoted/Submitted)
│   ├── submit_bundle.py       # SubmissionBuilder, SubmissionBundle
│   ├── dashboard.py           # Dashboard CLI (interaktiv)
│   ├── multi_seed.py          # MultiSeedOrchestrator
│   └── combo_builder.py       # DynamicComboBuilder, ComboConfig
│
├── reports/                   # Reports (Phase 1)
│   ├── __init__.py
│   ├── compare_runs.py        # RunComparator, RunComparison
│   └── leaderboard.py         # LeaderboardGenerator
│
├── runs/                      # Run-System (Phase 1)
│   ├── __init__.py
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
├── rust_core/                 # Python-Bindings (auto-generiert durch maturin)
│
├── results/                   # Ergebnisse (auto-generiert)
│   ├── runs/                  # Run-Outputs
│   ├── sweeps/                # Sweep-Ergebnisse
│   ├── bundles/               # Submission-Bundles
│   └── leaderboards/          # Leaderboard-Outputs
│
└── tests/                     # Tests
    ├── test_core.py
    ├── test_orchestrator.py
    └── test_research.py
```

---

## Usage

### Einzelnen Run starten

```bash
python3 -m runs.run --config configs/runs/run001_control.yaml
```

### Run mit spezifischem Seed

```bash
python3 -m runs.run --config configs/runs/run001_control.yaml --seed 42
```

### Eigene Run-Config erstellen

1. Kopiere eine existierende Config:
   ```bash
   cp configs/runs/run001_control.yaml configs/runs/my_custom_run.yaml
   ```

2. Bearbeite die Parameter:
   ```yaml
   run_id: "my_custom_run"
   seed: 42

   model:
     d_model: 768
     num_layers: 8
     activation: "gelu"
     use_feature_gate: true

   tokenizer:
     type: "bigram_hash"
     vocab_size: 8192

   quantization:
     enabled: true
     type: "mixed"  # int5/int6 mixed precision

   training:
     num_steps: 50000
     learning_rate: 1e-4
     batch_size: 32
   ```

3. Starte den Run:
   ```bash
   python3 -m runs.run --config configs/runs/my_custom_run.yaml
   ```

---

## Phase 3 Usage-Beispiele

### 1. Sweep Runner

Sweeps ermöglichen das automatische Durchlaufen von Parameter-Kombinationen.

**Config erstellen** (`configs/sweeps/my_sweep.yaml`):

```yaml
sweep_id: "my_first_sweep"
base_config: "configs/runs/run001_control.yaml"

parameters:
  - name: "model.d_model"
    values: [256, 512, 768]
  - name: "model.num_layers"
    values: [4, 6, 8]
  - name: "training.learning_rate"
    values: [1e-4, 3e-4, 1e-3]

execution:
  max_concurrent: 1
  continue_on_failure: true
```

**Sweep ausführen**:

```python
from orchestrator import SweepRunner, create_sweep

# Sweep erstellen
sweep = create_sweep("configs/sweeps/my_sweep.yaml")

# Alle Kombinationen durchlaufen
results = sweep.run_all()

# Ergebnisse analysieren
print(f"Total runs: {len(results)}")
print(f"Successful: {sum(1 for r in results if r.success)}")
```

**Performance-optimiert** (mit Benchmark):

```python
from orchestrator import SweepRunner
from eval import Benchmark

benchmark = Benchmark()
sweep = SweepRunner.from_config("configs/sweeps/my_sweep.yaml")

# Parameter-Generation benchmarken
gen_time = benchmark.measure(sweep.generate_all_parameters)
print(f"Parameter generation: {gen_time:.2f}ms")
```

### 2. Promotion System

Das Promotion System verwaltet Runs durch verschiedene Stages:
- `candidate` → Neuer Run zur Bewertung
- `promoted` → Erfolgreich bewerteter Run
- `submitted` → Zur Submission vorbereiteter Run

**Usage**:

```python
from orchestrator import PromotionSystem, Stage, create_promotion_system

# Promotion System erstellen
promo = create_promotion_system()

# Alle Runs evaluieren
promo.evaluate_all_runs()

# Runs nach Stage filtern
candidates = promo.get_by_stage(Stage.CANDIDATE)
promoted = promo.get_by_stage(Stage.PROMOTED)
submitted = promo.get_by_stage(Stage.SUBMITTED)

# Spezifischen Run promoten
promo.promote_run("run001_control")

# Run zur Submission einreichen
promo.submit_run("run016_best_combo_a")

# Status-Übersicht
print(f"Candidates: {len(candidates)}")
print(f"Promoted: {len(promoted)}")
print(f"Submitted: {len(submitted)}")
```

**Stage-Konfiguration anpassen**:

```python
from orchestrator import PromotionSystem, StageConfig

config = StageConfig(
    candidate_threshold={"val_bpb": 1.5, "ms_per_step": 100},
    promoted_threshold={"val_bpb": 1.2, "ms_per_step": 80},
    artifact_size_limit=16_000_000,
)

promo = PromotionSystem(config=config)
```

### 3. Submission Bundle

Erstellt ein ZIP-Bundle mit allen benötigten Artefakten für eine Submission.

**Usage**:

```python
from orchestrator import SubmissionBuilder, create_submission_bundle

# Builder erstellen
builder = create_submission_bundle()

# Bundle aus Runs erstellen
bundle = builder.build_bundle([
    "run001_control",
    "run016_best_combo_a",
    "run017_best_combo_quantized"
])

# Bundle speichern
bundle.save("results/submission.zip")

# Bundle-Info anzeigen
print(f"Bundle size: {bundle.size_bytes:,} bytes")
print(f"Runs included: {len(bundle.runs)}")
print(f"Configs included: {len(bundle.configs)}")
```

**Bundle-Inhalt**:

```
submission.zip
├── MANIFEST.json          # Bundle-Metadaten
├── runs/
│   ├── run001_control/
│   │   ├── model.pt
│   │   ├── config.yaml
│   │   └── metrics.json
│   └── ...
└── README.md              # Bundle-Beschreibung
```

### 4. Dashboard CLI

Interaktives CLI für Run-Übersicht und Analyse.

**Starten**:

```bash
python3 -m orchestrator.dashboard
```

**Verfügbare Commands**:

```
Dashboard Commands:
  list                      Alle Runs auflisten
  list --stage <stage>      Runs nach Stage filtern
  metrics <run_id>          Metriken für Run anzeigen
  compare <run1> <run2>     Zwei Runs vergleichen
  leaderboard               Leaderboard anzeigen
  leaderboard --by <metric> Nach Metrik sortieren (bpb, ms_per_step, etc.)
  sweep <sweep_id>          Sweep-Ergebnisse anzeigen
  promote <run_id>          Run promoten
  submit <run_id>           Run zur Submission einreichen
  exit                      Dashboard verlassen
```

**Beispiel-Session**:

```bash
$ python3 -m orchestrator.dashboard

Dashboard> list
ID                      Stage       Val_BPB    MS/Step
run001_control          promoted    1.234      45.2
run016_best_combo_a     submitted   1.156      52.1
run017_best_combo_quant submitted   1.189      38.7

Dashboard> metrics run001_control
Run: run001_control
  Val BPB: 1.234
  MS/Step: 45.2
  Steps: 10000
  Artifact Size: 8.5 MB

Dashboard> compare run001_control run016_best_combo_a
Metric          run001_control    run016_best_combo_a    Delta
Val BPB         1.234             1.156                  -6.3%
MS/Step         45.2              52.1                   +15.3%

Dashboard> leaderboard --by bpb
Rank  ID                      Val_BPB
1     run016_best_combo_a     1.156
2     run017_best_combo_quant 1.189
3     run001_control          1.234
```

### 5. Multi-Seed Orchestrator

Führt denselben Run mit mehreren Seeds für statistische Signifikanz aus.

```python
from orchestrator import MultiSeedOrchestrator, create_multi_seed_orchestrator

orchestrator = create_multi_seed_orchestrator(
    config_path="configs/runs/run001_control.yaml",
    seeds=[42, 123, 456],
)

results = orchestrator.run_all_seeds()

# Statistiken berechnen
from statistics import mean, stdev
bpb_values = [r.metrics["val_bpb"] for r in results]
print(f"Mean BPB: {mean(bpb_values):.4f}")
print(f"Std Dev: {stdev(bpb_values):.4f}")
```

### 6. Dynamic Combo Builder

Erstellt automatisch Feature-Kombinationen für systematisches Testing.

```python
from orchestrator import DynamicComboBuilder, generate_phase3_combos

# Phase 3 Combos generieren
combos = generate_phase3_combos()

# Builder erstellen
builder = DynamicComboBuilder(base_config="configs/runs/run001_control.yaml")

# Combos als Runs ausführen
for combo in combos:
    config = builder.build_config(combo)
    # Run mit config starten
```

---

## Runs vergleichen

```python
from reports import RunComparator

comparator = RunComparator()
comparison = comparator.compare_runs()
print(comparator.print_summary(comparison))
```

**Vergleichsmetriken**:
- `val_bpb` - Bits Per Byte (niedriger = besser)
- `ms_per_step` - Millisekunden pro Schritt (niedriger = besser)
- `steps_completed` - Abgeschlossene Schritte
- `artifact_bytes` - Modellgröße
- `quantized_val_bpb` - BPB nach Quantisierung
- `delta_bpb` - Änderung vs. Parent

---

## Leaderboard generieren

```python
from reports import LeaderboardGenerator

gen = LeaderboardGenerator()

# Alle Leaderboards generieren
leaderboards = gen.generate_all()

# BPB-Leaderboard
print(leaderboards["bpb"].print_table())

# Effizienz-Leaderboard
print(leaderboards["efficiency"].print_table())

# Quantized Leaderboard
print(leaderboards["quantized"].print_table())
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

---

## Troubleshooting

### Rust-Core Kompilierung fehlschlägt

```bash
# Rust installieren (falls nicht vorhanden)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# maturin aktualisieren
pip install --upgrade maturin

# Clean rebuild
cd rust-core
cargo clean
maturin develop --release
```

### Module nicht gefunden

```bash
# Paket im Development-Mode installieren
pip install -e .

# PYTHONPATH setzen
export PYTHONPATH=/path/to/wettkampf:$PYTHONPATH
```

### Permission Errors bei results/

```bash
# Verzeichnis-Rechte setzen
chmod -R 755 results/
```

---

## Nächste Schritte

### Für neue Nutzer

1. Installation abschließen
2. Test-Run starten: `python3 -m runs.run --config configs/runs/run001_control.yaml`
3. Dashboard ausprobieren: `python3 -m orchestrator.dashboard`
4. Eigene Config erstellen und testen

### Für Entwickler

1. Code-Struktur in `docs/architecture/ARCHITECTURE.md` lesen
2. Bestehende Tests verstehen
3. Eigene Feature-Branch von `main` erstellen
4. Pull Request mit Tests einreichen

---

## Dokumentation

- `README.md` - Projekt-Übersicht
- `SETUP.md` - Diese Datei (Installation & Usage)
- `HERMES.md` - Entwicklungs-Konventionen
- `docs/architecture/ARCHITECTURE.md` - System-Architektur
- `docs/runs_development.md` - Run-Entwicklung
