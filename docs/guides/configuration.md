# Konfigurations-Handbuch

**Letztes Update:** 2026-03-24
**Version:** 1.0
**Config-Typen:** Base Config, Run Config, Sweep Config

---

## Übersicht

Das Wettkampf-System verwendet YAML-Konfigurationen für alle Aspekte des Experiment-Managements. Es gibt drei Config-Typen:

1. **Base Config** (`configs/base.yaml`) – Default-Werte für alle Runs
2. **Run Config** (`configs/runs/*.yaml`) – Spezifische Experiment-Konfiguration
3. **Sweep Config** (`configs/sweeps/*.yaml`) – Parameter-Sweep-Definition

---

## Base Config

**Pfad:** `configs/base.yaml`

Die Base Config enthält Default-Werte, die von allen Runs geerbt werden.

### Schema

```yaml
# configs/base.yaml

# Run-Identifikation
run_id: "default"
seed: 42
parent_run_id: null # Für Lineage-Tracking

# Modell-Konfiguration
model:
d_model: 512
num_layers: 6
num_heads: 8
activation: "gelu" # gelu, relu, leaky_relu, star_relu
use_layer_norm: true
use_feature_gate: false
dropout: 0.0

# Tokenizer-Konfiguration
tokenizer:
type: "byte" # byte, bigram_hash, trigram_hash, fallback
vocab_size: 8192

# Quantisierung
quantization:
enabled: false
type: "int6" # int6, int5, mixed, gptq_lite
threshold: 0.3 # Für MixedQuantizer
scale: 0.1

# Training-Konfiguration
training:
num_steps: 10000
learning_rate: 3e-4
batch_size: 32
warmup_steps: 1000
weight_decay: 0.01
use_ema: false
ema_decay: 0.999

# Optimizer
optimizer:
type: "adam" # adam, adamw, sgd
beta1: 0.9
beta2: 0.999
epsilon: 1e-8

# Scheduler
scheduler:
type: "cosine" # cosine, linear, constant
min_lr: 1e-6

# Evaluation
evaluation:
eval_steps: 1000
use_sliding_window: true
window_size: 100

# Logging
logging:
log_steps: 100
save_artifacts: true
artifact_format: "pt" # pt, safetensors

# Kill-Rules (für Ablation)
kill_rules:
artifact_size_limit: 16000000 # 16 MB
ms_per_step_limit: 100
bpb_degradation_limit: 0.1
```

---

## Run Config

**Pfad:** `configs/runs/*.yaml`

Run Configs überschreiben die Base Config für spezifische Experimente.

### Beispiel: Control Run

```yaml
# configs/runs/run001_control.yaml
run_id: "run001_control"
seed: 42

model:
d_model: 512
num_layers: 6
activation: "gelu"

tokenizer:
type: "byte"

quantization:
enabled: false

training:
num_steps: 10000
learning_rate: 3e-4
```

### Beispiel: Feature-Experiment

```yaml
# configs/runs/run006_film.yaml
run_id: "run006_film"
seed: 42
parent_run_id: "run001_control" # Lineage-Tracking

model:
d_model: 512
num_layers: 6
activation: "gelu"
use_feature_gate: true
feature_gates:
- name: "film"
condition: true
params:
conditioning_dim: 256

tokenizer:
type: "byte"

training:
num_steps: 10000
learning_rate: 3e-4
```

### Beispiel: Quantized Run

```yaml
# configs/runs/run005_mixed_quant.yaml
run_id: "run005_mixed_quant"
seed: 42
parent_run_id: "run001_control"

model:
d_model: 512
num_layers: 6
activation: "gelu"

tokenizer:
type: "byte"

quantization:
enabled: true
type: "mixed"
threshold: 0.3

training:
num_steps: 10000
learning_rate: 3e-4
```

### Beispiel: Multi-Seed Run

```yaml
# configs/runs/run018_control_s1.yaml
run_id: "run018_control_s1"
seed: 1 # Variierter Seed
parent_run_id: "run001_control"

# Alle anderen Werte von Base Config
```

---

## Sweep Config

**Pfad:** `configs/sweeps/*.yaml`

Sweep Configs definieren Parameter-Raster für automatisierte Sweeps.

### Schema

```yaml
# configs/sweeps/my_sweep.yaml
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
output_dir: "results/sweeps/my_sweep"
```

### Beispiel: Architecture Sweep

```yaml
# configs/sweeps/architecture_sweep.yaml
sweep_id: "architecture_sweep"
base_config: "configs/runs/run001_control.yaml"

parameters:
- name: "model.d_model"
values: [256, 512, 768, 1024]
- name: "model.num_layers"
values: [4, 6, 8, 12]
- name: "model.activation"
values: ["gelu", "relu", "leaky_relu", "star_relu"]

execution:
max_concurrent: 1
continue_on_failure: true
```

### Beispiel: Quantization Sweep

```yaml
# configs/sweeps/quant_sweep.yaml
sweep_id: "quant_sweep"
base_config: "configs/runs/run001_control.yaml"

parameters:
- name: "quantization.type"
values: ["int6", "int5", "mixed", "gptq_lite"]
- name: "quantization.threshold"
values: [0.2, 0.3, 0.5]

execution:
max_concurrent: 1
continue_on_failure: true
```

---

## Config-Loading

### Python-API

```python
from core.config import Config, load_config, merge_configs

# Base Config laden
base = load_config("configs/base.yaml")

# Run Config laden
run = load_config("configs/runs/run001_control.yaml")

# Merge (Run überschreibt Base)
config = merge_configs(base, run)

# Zugriff auf Werte
print(config.run_id) # "run001_control"
print(config.model.d_model) # 512
print(config.training.learning_rate) # 3e-4

# Config-Hash (für Reproduzierbarkeit)
print(config.config_hash) # "a1b2c3d4e5f6..."
```

### Config-Klasse

```python
class Config:
"""Konfigurations-Wrapper mit Type-Safety."""

def __init__(self, raw: dict):
self._raw = raw

@property
def run_id(self) -> str:
return self._raw.get("run_id", "default")

@property
def seed(self) -> int:
return self._raw.get("seed", 42)

@property
def model(self) -> ModelConfig:
return ModelConfig(self._raw.get("model", {}))

@property
def training(self) -> TrainingConfig:
return TrainingConfig(self._raw.get("training", {}))

@property
def config_hash(self) -> str:
"""SHA256-Hash der Config (cached)."""
if not hasattr(self, '_cached_hash'):
config_str = json.dumps(self._raw, sort_keys=True)
self._cached_hash = hashlib.sha256(
config_str.encode()
).hexdigest()[:16]
return self._cached_hash
```

---

## Verfügbare Optionen

### Modell-Optionen

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `d_model` | int | 512 | Modell-Dimension |
| `num_layers` | int | 6 | Anzahl Layer |
| `num_heads` | int | 8 | Attention-Heads |
| `activation` | str | "gelu" | Activation-Funktion |
| `use_layer_norm` | bool | true | Layer Normalization |
| `use_feature_gate` | bool | false | Feature-Gates aktivieren |
| `dropout` | float | 0.0 | Dropout-Rate |

**Activation-Typen:**
- `gelu` – Gaussian Error Linear Unit
- `relu` – Rectified Linear Unit
- `leaky_relu` – Leaky ReLU
- `star_relu` – StarReLU (Phase 3)

### Tokenizer-Optionen

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `type` | str | "byte" | Tokenizer-Typ |
| `vocab_size` | int | 8192 | Vokabular-Größe |

**Tokenizer-Typen:**
- `byte` – Raw Byte-Encoding (256 Tokens)
- `bigram_hash` – Hash-basierte Bigramme
- `trigram_hash` – Hash-basierte Trigramme
- `fallback` – Fallback für OOV-Tokens

### Quantisierung-Optionen

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `enabled` | bool | false | Quantisierung aktivieren |
| `type` | str | "int6" | Quantisierungs-Typ |
| `threshold` | float | 0.3 | Threshold für Mixed |
| `scale` | float | 0.1 | Quantisierungs-Scale |

**Quantisierungs-Typen:**
- `int6` – 6-Bit (Werte 0-63)
- `int5` – 5-Bit (Werte 0-31)
- `mixed` – Mixed Precision INT5/INT6
- `gptq_lite` – GPTQ-ähnliche Quantisierung

### Training-Optionen

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `num_steps` | int | 10000 | Training-Schritte |
| `learning_rate` | float | 3e-4 | Lernrate |
| `batch_size` | int | 32 | Batch-Größe |
| `warmup_steps` | int | 1000 | Warmup-Schritte |
| `weight_decay` | float | 0.01 | Weight Decay |
| `use_ema` | bool | false | Exponential Moving Average |
| `ema_decay` | float | 0.999 | EMA-Decay |

### Optimizer-Optionen

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `type` | str | "adam" | Optimizer-Typ |
| `beta1` | float | 0.9 | Adam Beta1 |
| `beta2` | float | 0.999 | Adam Beta2 |
| `epsilon` | float | 1e-8 | Adam Epsilon |

**Optimizer-Typen:**
- `adam` – Adam Optimizer
- `adamw` – AdamW Optimizer
- `sgd` – Stochastic Gradient Descent

### Scheduler-Optionen

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `type` | str | "cosine" | Scheduler-Typ |
| `min_lr` | float | 1e-6 | Minimale Lernrate |

**Scheduler-Typen:**
- `cosine` – Cosine Annealing
- `linear` – Linear Decay
- `constant` – Konstante Lernrate

---

## Eigene Config erstellen

### Schritt 1: Base Config kopieren

```bash
cp configs/runs/run001_control.yaml configs/runs/my_custom_run.yaml
```

### Schritt 2: Config bearbeiten

```yaml
# configs/runs/my_custom_run.yaml
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
type: "mixed"

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

## Config-Validierung

### Automatische Validierung

Beim Laden einer Config wird automatisch validiert:

```python
from core.config import Config, ConfigValidationError

try:
config = load_config("configs/runs/invalid.yaml")
config.validate() # Wirft ConfigValidationError bei Fehlern
except ConfigValidationError as e:
print(f"Config-Validierung fehlgeschlagen: {e}")
```

### Validierungs-Regeln

| Regel | Beschreibung |
|-------|--------------|
| `run_id` | Muss eindeutig sein |
| `seed` | Muss positiv sein |
| `d_model` | Muss durch `num_heads` teilbar sein |
| `num_layers` | Muss ≥ 1 sein |
| `learning_rate` | Muss > 0 sein |
| `batch_size` | Muss positiv sein |
| `quantization.threshold` | Muss zwischen 0 und 1 liegen |

---

## Environment-Variablen

Beste Config-Werte können via Environment-Variablen überschrieben werden:

```bash
# Learning Rate überschreiben
TRAINING_LEARNING_RATE=1e-3 python3 -m runs.run --config configs/runs/run001_control.yaml

# Seed überschreiben
SEED=123 python3 -m runs.run --config configs/runs/run001_control.yaml
```

**Format:** `SECTION_KEY=value` (Großbuchstaben, Unterstriche)

---

## Verwandte Dokumente

- [runs_guide.md](runs_guide.md) – Runs starten
- [sweep_guide.md](sweep_guide.md) – Sweep Runner
- [../architecture/module_overview.md](../architecture/module_overview.md) – Modul-Übersicht
- [../setup/SETUP.md](../setup/SETUP.md) – Installation
