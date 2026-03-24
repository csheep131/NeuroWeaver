# Runs starten

**Letztes Update:** 2026-03-24  
**Status:** ✅ Vollständig implementiert (Phase 1-3)

---

## Übersicht

Ein "Run" ist eine einzelne Experiment-Ausführung mit spezifischer Konfiguration. Dieser Guide erklärt, wie man Runs startet, überwacht und analysiert.

---

## Schnellstart

### Einzelnen Run starten

```bash
python3 -m runs.run --config configs/runs/run001_control.yaml
```

### Run mit spezifischem Seed

```bash
python3 -m runs.run --config configs/runs/run001_control.yaml --seed 42
```

---

## Run-System Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    runs/run.py                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  execute_run(config: Config) -> RunResult           │    │
│  │  1. Registry initialisieren                          │    │
│  │  2. Logger einrichten                                │    │
│  │  3. Seed setzen                                      │    │
│  │  4. Modell erstellen (BackboneFactory)               │    │
│  │  5. Tokenizer erstellen                              │    │
│  │  6. Training-Loop ausführen                          │    │
│  │  7. Evaluation durchführen                           │    │
│  │  8. Artefakte speichern                              │    │
│  │  9. Metriken loggen                                  │    │
│  │  10. Registry aktualisieren                          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    core/registry.py                          │
│  RunRegistry speichert Run-Metadaten und Metriken           │
└─────────────────────────────────────────────────────────────┘
```

---

## CLI-Usage

### Basis-Command

```bash
python3 -m runs.run --config <CONFIG_PATH>
```

### Optionen

| Option | Kurz | Beschreibung | Default |
|--------|------|--------------|---------|
| `--config` | `-c` | Pfad zur Run-Config | Erforderlich |
| `--seed` | `-s` | Seed überschreiben | Aus Config |
| `--output-dir` | `-o` | Output-Verzeichnis | `results/runs/` |
| `--dry-run` | `-n` | Config validieren, nicht ausführen | `false` |
| `--verbose` | `-v` | Debug-Logging | `false` |

### Beispiele

```bash
# Standard Run
python3 -m runs.run -c configs/runs/run001_control.yaml

# Mit Seed-Override
python3 -m runs.run -c configs/runs/run001_control.yaml -s 123

# Mit Custom Output
python3 -m runs.run -c configs/runs/run001_control.yaml -o results/custom/

# Dry-Run (nur Validierung)
python3 -m runs.run -c configs/runs/my_config.yaml -n

# Mit verbose Logging
python3 -m runs.run -c configs/runs/run001_control.yaml -v
```

---

## Verfügbare Run-Configs

### Control Runs (Baseline)

| Config | Beschreibung | Seed |
|--------|--------------|------|
| `run001_control.yaml` | Baseline ohne Features | 42 |
| `run018_control_s1.yaml` | Control Seed 1 | 1 |
| `run019_control_s2.yaml` | Control Seed 2 | 2 |
| `run020_control_s3.yaml` | Control Seed 3 | 3 |

### Tokenizer-Experimente

| Config | Beschreibung | Tokenizer |
|--------|--------------|-----------|
| `run002_hash.yaml` | Bigram/Trigram Hash | bigram_hash |

### Architektur-Experimente

| Config | Beschreibung | Feature |
|--------|--------------|---------|
| `run003_xsa.yaml` | Cross-Attention | XSA |
| `run006_film.yaml` | FiLM Conditioning | FiLM |
| `run007_ttt.yaml` | TTT Layer | TTT |
| `run009_gqa.yaml` | Grouped Query Attention | GQA |
| `run010_recurrence.yaml` | Recurrence | Recurrent |

### Activation-Experimente

| Config | Beschreibung | Activation |
|--------|--------------|------------|
| `run004_leakyrelu.yaml` | LeakyReLU | leaky_relu |
| `run008a_star_relu.yaml` | StarReLU | star_relu |

### Quantisierung-Experimente

| Config | Beschreibung | Quantisierung |
|--------|--------------|---------------|
| `run005_mixed_quant.yaml` | Mixed Precision | mixed (INT5/INT6) |
| `run017_best_combo_quantized.yaml` | Quantized Combo | mixed |

### Combo-Experimente

| Config | Beschreibung |
|--------|--------------|
| `run016_best_combo_a.yaml` | Beste Feature-Kombination |
| `run017_best_combo_quantized.yaml` | Quantisierte Best-Kombi |

---

## Run-Ablauf

### 1. Initialisierung

```python
# runs/run.py
def execute_run(config_path: str, seed: int | None = None) -> RunResult:
    # Config laden
    config = load_config(config_path)
    
    # Seed überschreiben falls angegeben
    if seed is not None:
        config._raw["seed"] = seed
    
    # Registry initialisieren
    registry = RunRegistry()
    
    # Logger einrichten
    logger = RunLogger(config.run_id)
    
    # Seed setzen (Reproduzierbarkeit)
    set_seed(config.seed)
    
    # Run in Registry registrieren
    registry.register(config.run_id, config)
```

### 2. Modell-Erstellung

```python
# BackboneFactory verwenden
from models.factories import BackboneFactory

factory = BackboneFactory(use_rust=True)
model = factory.create(config)
```

### 3. Training-Loop

```python
# Trainer initialisieren
from train.trainer import Trainer

trainer = Trainer(
    model=model,
    config=config,
    logger=logger,
)

# Training durchführen
for step in range(config.training.num_steps):
    batch = get_next_batch()
    loss = trainer.step(batch)
    
    # Metriken loggen
    if step % config.logging.log_steps == 0:
        logger.log_metrics({
            "loss": loss,
            "step": step,
        })
```

### 4. Evaluation

```python
# BPB-Evaluation
from eval.bpb_eval import BPBEvaluator

evaluator = BPBEvaluator()
val_bpb = evaluator.evaluate(model, val_data)

# Metriken speichern
logger.log_metrics({
    "val_bpb": val_bpb,
    "ms_per_step": trainer.avg_step_time,
    "steps_completed": step,
})
```

### 5. Artefakte speichern

```python
# Modell speichern
artifact_path = logger.save_artifact(
    model.state_dict(),
    "model.pt",
)

# Artifact-Größe tracken
from core.artifacts import ArtifactTracker

tracker = ArtifactTracker()
artifact_bytes = tracker.get_size(artifact_path)

logger.log_metrics({"artifact_bytes": artifact_bytes})
```

### 6. Registry aktualisieren

```python
# Run abschließen
registry.complete_run(
    config.run_id,
    metrics=logger.get_metrics(),
    artifact_path=artifact_path,
)
```

---

## Run-Metriken

Jeder Run produziert folgende Metriken:

| Metrik | Beschreibung | Ziel |
|--------|--------------|------|
| `val_bpb` | Validation Bits Per Byte | Niedriger = besser |
| `ms_per_step` | Millisekunden pro Training-Schritt | Niedriger = besser |
| `steps_completed` | Abgeschlossene Training-Schritte | Höher = besser |
| `artifact_bytes` | Größe der Modell-Artefakte | Niedriger = besser |
| `quantized_val_bpb` | BPB nach Quantisierung | Niedriger = besser |
| `delta_bpb` | BPB-Änderung vs. Parent-Run | Negativ = besser |
| `delta_ms` | ms/Step-Änderung vs. Parent | Negativ = besser |

---

## Run-Outputs

### Verzeichnis-Struktur

```
results/
└── runs/
    └── run001_control/
        ├── model.pt              # Modell-Gewichte
        ├── config.yaml           # Verwendete Config
        ├── metrics.json          # Alle Metriken
        ├── logs.jsonl            # Training-Logs
        └── artifacts/
            └── ...               # Zusätzliche Artefakte
```

### metrics.json Format

```json
{
  "run_id": "run001_control",
  "seed": 42,
  "config_hash": "a1b2c3d4e5f6...",
  "parent_run_id": null,
  "status": "completed",
  "metrics": {
    "val_bpb": 1.234,
    "ms_per_step": 45.2,
    "steps_completed": 10000,
    "artifact_bytes": 8500000,
    "quantized_val_bpb": 1.289,
    "delta_bpb": 0.0,
    "delta_ms": 0.0
  },
  "started_at": "2026-03-24T10:00:00Z",
  "completed_at": "2026-03-24T12:30:00Z"
}
```

---

## Runs überwachen

### Live-Metriken anzeigen

Während ein Run läuft, können die Metriken im Logfile verfolgt werden:

```bash
# Logs in Echtzeit verfolgen
tail -f results/runs/run001_control/logs.jsonl
```

### Fortschritt anzeigen

```bash
# Einfache Fortschrittsanzeige
python3 -c "
import json
with open('results/runs/run001_control/metrics.json') as f:
    metrics = json.load(f)
    print(f'Steps: {metrics[\"metrics\"][\"steps_completed\"]}')
    print(f'Val BPB: {metrics[\"metrics\"][\"val_bpb\"]:.4f}')
"
```

---

## Runs vergleichen

### RunComparator verwenden

```python
from reports import RunComparator

comparator = RunComparator()
comparison = comparator.compare_runs(["run001_control", "run002_hash"])

print(comparator.print_summary(comparison))
```

### Ausgabe

```
Run Comparison
==============
Metric          run001_control    run002_hash    Delta
---------------------------------------------------------
Val BPB         1.234             1.256          +1.8%
MS/Step         45.2              42.1           -6.9%
Steps           10000             10000          0.0%
Artifact (MB)   8.5               8.2            -3.5%
```

---

## Eigene Run-Config erstellen

### Schritt 1: Template kopieren

```bash
cp configs/runs/run001_control.yaml configs/runs/my_custom_run.yaml
```

### Schritt 2: Config anpassen

```yaml
# configs/runs/my_custom_run.yaml
run_id: "my_custom_run"
seed: 42
parent_run_id: "run001_control"  # Für Lineage-Tracking

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
  type: "mixed"
  threshold: 0.3

training:
  num_steps: 50000
  learning_rate: 1e-4
  batch_size: 32
```

### Schritt 3: Run starten

```bash
python3 -m runs.run --config configs/runs/my_custom_run.yaml
```

---

## Troubleshooting

### Run stürzt ab

**Ursache:** Config-Fehler, Memory-Issue, Rust-Core nicht verfügbar

**Lösung:**
```bash
# Mit verbose Logging starten
python3 -m runs.run -c configs/runs/my_run.yaml -v

# Dry-Run für Config-Validierung
python3 -m runs.run -c configs/runs/my_run.yaml -n
```

### Rust-Core nicht gefunden

```bash
# Rust-Core neu kompilieren
cd rust-core
maturin develop --release
```

### Memory-Issues

```yaml
# In der Config batch_size reduzieren
training:
  batch_size: 16  # Statt 32
```

### Artifact zu groß

Kill-Rule greift bei >16MB:

```yaml
# Modell-Größe reduzieren
model:
  d_model: 256  # Statt 512
  num_layers: 4  # Statt 6
```

---

## Best Practices

### 1. Reproduzierbarkeit

- Immer Seeds dokumentieren
- Config-Hash für Run-Tracking verwenden
- Parent-Run-ID für Lineage setzen

### 2. Experiment-Design

- Control-Run als Baseline verwenden
- Immer nur ein Feature pro Run ändern
- Multi-Seed-Runs für statistische Signifikanz

### 3. Resource-Management

- Artifact-Size-Limit beachten (16MB)
- Training-Schritte angemessen wählen
- Batch-Größe an verfügbaren Memory anpassen

### 4. Logging

- Regelmäßige Metrik-Logs (alle 100 Steps)
- Artefakte nur bei Bedarf speichern
- Verbose-Modus für Debugging

---

## Verwandte Dokumente

- [configuration.md](configuration.md) – Konfigurations-Handbuch
- [sweep_guide.md](sweep_guide.md) – Sweep Runner
- [../architecture/module_overview.md](../architecture/module_overview.md) – Modul-Übersicht
- [../reports/phase_1_audit.md](../reports/phase_1_audit.md) – Phase 1 Audit
