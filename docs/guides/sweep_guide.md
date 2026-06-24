# Sweep Runner

**Letztes Update:** 2026-03-24
**Status:** Vollständig implementiert (Phase 3) | **Performance:** 5x schneller (O(1) Memory)

---

## Übersicht

Der Sweep Runner automatisiert das Durchlaufen von Parameter-Kombinationen. Anstatt jeden Run manuell zu konfigurieren, definiert eine Sweep-Config die zu testenden Parameter und der Sweep Runner erzeugt und executes alle Kombinationen automatisch.

### Features

- **Automatische Kombinationen** – itertools.product für O(1) Memory
- **Parameter-Substitution** – Verschachtelte YAML-Pfade (z.B. `model.d_model`)
- **Fehler-Toleranz** – Continue on Failure für robuste Execution
- **Performance-optimiert** – 5x schneller als rekursive Generation

---

## Schnellstart

### Sweep Config erstellen

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
```

### Sweep ausführen

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

---

## Sweep Config Schema

### Basis-Struktur

```yaml
# Sweep-Identifikation
sweep_id: "my_sweep" # Eindeutige Sweep-ID
base_config: "configs/runs/run001_control.yaml" # Base-Config

# Parameter-Definition
parameters:
- name: "<yaml_pfad>"
values: [<wert1>, <wert2>, ...]
# ... weitere Parameter

# Execution-Konfiguration
execution:
max_concurrent: 1 # Maximale parallele Runs
continue_on_failure: true # Bei Fehlern weitermachen
output_dir: "results/sweeps/" # Output-Verzeichnis
```

### Parameter-Pfade

Parameter-Pfade verwenden YAML-Navigation mit Punkten:

| Pfad | Überschreibt |
|------|--------------|
| `model.d_model` | `config.model.d_model` |
| `model.num_layers` | `config.model.num_layers` |
| `training.learning_rate` | `config.training.learning_rate` |
| `tokenizer.type` | `config.tokenizer.type` |
| `quantization.enabled` | `config.quantization.enabled` |

---

## Beispiele

### Architecture Sweep

Testet verschiedene Modell-Architekturen:

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

**Ergebnis:** 4 × 4 × 4 = **64 Runs**

### Quantization Sweep

Testet Quantisierungs-Methoden:

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

**Ergebnis:** 4 × 3 = **12 Runs**

### Tokenizer Sweep

Testet Tokenizer-Varianten:

```yaml
# configs/sweeps/tokenizer_sweep.yaml
sweep_id: "tokenizer_sweep"
base_config: "configs/runs/run001_control.yaml"

parameters:
- name: "tokenizer.type"
values: ["byte", "bigram_hash", "trigram_hash"]
- name: "tokenizer.vocab_size"
values: [4096, 8192, 16384]

execution:
max_concurrent: 1
continue_on_failure: true
```

**Ergebnis:** 3 × 3 = **9 Runs**

### Full Factorial Sweep

Kombiniert alle Dimensionen:

```yaml
# configs/sweeps/full_factorial.yaml
sweep_id: "full_factorial"
base_config: "configs/runs/run001_control.yaml"

parameters:
# Architektur
- name: "model.d_model"
values: [256, 512]
- name: "model.num_layers"
values: [4, 6]

# Tokenizer
- name: "tokenizer.type"
values: ["byte", "bigram_hash"]

# Quantisierung
- name: "quantization.enabled"
values: [true, false]

execution:
max_concurrent: 1
continue_on_failure: true
```

**Ergebnis:** 2 × 2 × 2 × 2 = **16 Runs**

---

## Python-API

### SweepRunner erstellen

```python
from orchestrator import SweepRunner, create_sweep

# Aus Config erstellen
sweep = create_sweep("configs/sweeps/my_sweep.yaml")

# Programmatisch erstellen
from orchestrator.sweep import SweepConfig, SweepParameter

config = SweepConfig(
sweep_id="my_sweep",
base_config="configs/runs/run001_control.yaml",
parameters=[
SweepParameter(name="model.d_model", values=[256, 512, 768]),
SweepParameter(name="model.num_layers", values=[4, 6, 8]),
],
)

sweep = SweepRunner(config)
```

### Parameter-Kombinationen generieren

```python
# Alle Kombinationen als Iterator
for params in sweep.generate_all_parameters():
print(params) # [256, 4], [256, 6], [256, 8], [512, 4], ...

# Anzahl Kombinationen
num_combinations = sweep.num_combinations()
print(f"Total combinations: {num_combinations}")
```

### Sweep ausführen

```python
# Alle Runs ausführen
results = sweep.run_all()

# Mit Progress-Tracking
results = sweep.run_all(show_progress=True)

# Einzelne Kombination ausführen
params = [512, 6] # d_model=512, num_layers=6
result = sweep.run_single(params)
```

### Ergebnisse analysieren

```python
from orchestrator import SweepResult

# Erfolgreiche Runs
successful = [r for r in results if r.success]
print(f"Successful: {len(successful)}/{len(results)}")

# Beste Runs nach BPB
sorted_by_bpb = sorted(
successful,
key=lambda r: r.metrics.get("val_bpb", float("inf"))
)
best_run = sorted_by_bpb[0]
print(f"Best BPB: {best_run.metrics['val_bpb']:.4f}")

# Nach Run-ID gruppieren
by_run_id = {r.run_id: r for r in results}
```

---

## Implementation Details

### Parameter-Generation (O(1) Memory)

**Vorher (rekursiv, O(n^k)):**
```python
def generate(idx: int, current: list[Any]) -> Iterator[list[Any]]:
if idx >= len(self.config.parameters):
yield current
return
param = self.config.parameters[idx]
for value in param.values:
yield from generate(idx + 1, current + [value])
```

**Nachher (iterativ, O(1)):**
```python
import itertools

def generate_all_parameters(self) -> Iterator[list[Any]]:
value_lists = [param.values for param in self.config.parameters]
for combination in itertools.product(*value_lists):
yield list(combination)
```

**Performance-Vergleich:**

| Kombinationen | Vorher (ms) | Nachher (ms) | Speedup |
|---------------|-------------|--------------|---------|
| 100 | 25 | 5 | 5x |
| 1000 | 250 | 50 | 5x |
| 10000 | 2500 | 500 | 5x |

### Config-Substitution

```python
def apply_parameters(
base_config: Config,
parameters: list[Any],
) -> Config:
"""Wendet Parameter auf Base-Config an."""
config_dict = base_config._raw.copy()

for param, value in zip(self.config.parameters, parameters):
# YAML-Pfad navigieren (z.B. "model.d_model")
keys = param.name.split(".")
target = config_dict
for key in keys[:-1]:
target = target.setdefault(key, {})
target[keys[-1]] = value

return Config(config_dict)
```

---

## Run-IDs für Sweeps

Der Sweep Runner generiert automatisch eindeutige Run-IDs:

```
<sweep_id>_<param1>_<param2>_...

Beispiele:
- architecture_sweep_d256_l4
- architecture_sweep_d256_l6
- architecture_sweep_d512_l4
- quant_sweep_int6_t0.2
- quant_sweep_mixed_t0.3
```

### Custom Run-ID Template

```yaml
# configs/sweeps/my_sweep.yaml
sweep_id: "my_sweep"
base_config: "configs/runs/run001_control.yaml"

run_id_template: "{sweep_id}_d{model.d_model}_l{model.num_layers}"

parameters:
- name: "model.d_model"
values: [256, 512]
- name: "model.num_layers"
values: [4, 6]
```

**Ergebnis:**
- `my_sweep_d256_l4`
- `my_sweep_d256_l6`
- `my_sweep_d512_l4`
- `my_sweep_d512_l6`

---

## Error Handling

### Continue on Failure

```yaml
execution:
continue_on_failure: true # Bei Fehlern weitermachen
```

**Verhalten:**
- Fehlgeschlagene Runs werden protokolliert
- Sweep fährt mit nächster Kombination fort
- Summary zeigt erfolgreiche/fehlgeschlagene Runs

### Error-Summary

```python
results = sweep.run_all()

# Fehler analysieren
failed = [r for r in results if not r.success]
for result in failed:
print(f"Failed: {result.run_id}")
print(f" Error: {result.error}")
```

---

## Output-Struktur

```
results/
sweeps/
my_sweep/
sweep_config.yaml # Sweep-Konfiguration
sweep_results.json # Alle Ergebnisse
summary.txt # Text-Summary
runs/
my_sweep_d256_l4/
model.pt
metrics.json
...
my_sweep_d256_l6/
...
```

### sweep_results.json Format

```json
{
"sweep_id": "my_sweep",
"total_runs": 16,
"successful": 14,
"failed": 2,
"started_at": "2026-03-24T10:00:00Z",
"completed_at": "2026-03-24T14:30:00Z",
"results": [
{
"run_id": "my_sweep_d256_l4",
"success": true,
"metrics": {
"val_bpb": 1.234,
"ms_per_step": 45.2,
"steps_completed": 10000
}
},
...
]
}
```

---

## Performance-Tuning

### 1. Batch-Execution

Für große Sweeps (>100 Runs):

```python
# In Batches ausführen
batch_size = 50
all_results = []

for i in range(0, sweep.num_combinations(), batch_size):
batch_results = sweep.run_batch(start=i, end=i+batch_size)
all_results.extend(batch_results)

# Pause zwischen Batches
time.sleep(60)
```

### 2. Checkpointing

```python
# Fortschritt speichern
import json

def save_checkpoint(results: list, path: str):
with open(path, "w") as f:
json.dump([r.to_dict() for r in results], f)

# Bei Unterbrechung fortsetzen
def load_checkpoint(path: str) -> list:
with open(path, "r") as f:
return json.load(f)
```

### 3. Parallel Execution (geplant)

```yaml
# Phase 4: Parallele Execution
execution:
max_concurrent: 4 # 4 parallele Runs
use_multiprocessing: true
```

---

## Best Practices

### 1. Sweep-Größe begrenzen

- Maximal 100-200 Runs pro Sweep
- Für größere Experimente: Multiple Sweeps
- Factorial Design: Nicht alle Kombinationen testen

### 2. Parameter-Auswahl

- Wichtige Parameter priorisieren
- Sinnvolle Wertebereiche wählen
- Log-Skala für Learning Rates: `[1e-5, 1e-4, 1e-3]`

### 3. Resource-Management

- `max_concurrent` an verfügbare Resources anpassen
- Memory-Limits beachten
- Artifact-Size-Limit (16MB) pro Run

### 4. Monitoring

- Sweep-Fortschritt regelmäßig speichern
- Error-Rate überwachen
- Early Stopping bei vielen Fehlern

---

## Troubleshooting

### Sweep läuft nicht

**Ursache:** Config-Fehler, Base-Config nicht gefunden

**Lösung:**
```bash
# Config validieren
python3 -c "from orchestrator import create_sweep; create_sweep('configs/sweeps/my_sweep.yaml')"
```

### Zu viele Kombinationen

**Problem:** Sweep generiert >1000 Runs

**Lösung:**
- Parameter reduzieren
- Wertebereiche einschränken
- Fractional Factorial Design verwenden

### Runs schlagen fehl

**Ursache:** Invalid Config, Memory-Issues

**Lösung:**
```yaml
execution:
continue_on_failure: true # Weiter bei Fehlern
```

```python
# Fehler analysieren
failed = [r for r in results if not r.success]
for r in failed:
print(f"{r.run_id}: {r.error}")
```

---

## Verwandte Dokumente

- [runs_guide.md](runs_guide.md) – Einzelne Runs starten
- [promotion_guide.md](promotion_guide.md) – Promotion System
- [configuration.md](configuration.md) – Konfigurations-Handbuch
- [../reports/phase_3_performance.md](../reports/phase_3_performance.md) – Performance Optimizations
