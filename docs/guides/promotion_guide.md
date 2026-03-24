# Promotion System

**Letztes Update:** 2026-03-24  
**Status:** ✅ Vollständig implementiert (Phase 3) | **Performance:** 5x schneller (3-Layer Caching)

---

## Übersicht

Das Promotion System verwaltet Runs durch verschiedene Stages (Candidate → Promoted → Submitted) und automatisiert die Bewertung von Runs basierend auf definierten Kriterien.

### Features

- **Stage-Management** – 3 Stages: Candidate, Promoted, Submitted
- **Automatisierte Bewertung** – Screening-Kriterien für Promotion
- **3-Layer Caching** – O(1) Lookups für 1000+ Runs
- **Kill-Rules Integration** – Automatische Disqualifikation bei Regelverstößen

---

## Schnellstart

### Promotion System erstellen

```python
from orchestrator import PromotionSystem, create_promotion_system

# Mit Default-Konfiguration
promo = create_promotion_system()

# Alle Runs evaluieren
promo.evaluate_all_runs()

# Runs nach Stage filtern
candidates = promo.get_by_stage("candidate")
promoted = promo.get_by_stage("promoted")
submitted = promo.get_by_stage("submitted")

print(f"Candidates: {len(candidates)}")
print(f"Promoted: {len(promoted)}")
print(f"Submitted: {len(submitted)}")
```

---

## Stage-Übersicht

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  CANDIDATE  │ ───► │  PROMOTED   │ ───► │  SUBMITTED  │
│  (Neu)      │      │ (Bewertet)  │      │ (Eingereicht)│
└─────────────┘      └─────────────┘      └─────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
   Screening           Promotion            Submission
   Evaluation          Evaluation           Bundle
```

### Stage-Enum

```python
from orchestrator import Stage

class Stage(Enum):
    CANDIDATE = "candidate"    # Neuer Run zur Bewertung
    PROMOTED = "promoted"      # Erfolgreich bewerteter Run
    SUBMITTED = "submitted"    # Zur Submission vorbereiteter Run
```

---

## Python-API

### PromotionSystem erstellen

```python
from orchestrator import PromotionSystem, StageConfig, create_promotion_system

# Mit Default-Konfiguration
promo = create_promotion_system()

# Mit benutzerdefinierter Konfiguration
config = StageConfig(
    candidate_threshold={"val_bpb": 1.5, "ms_per_step": 100},
    promoted_threshold={"val_bpb": 1.2, "ms_per_step": 80},
    artifact_size_limit=16_000_000,
)

promo = PromotionSystem(config=config)
```

### Runs evaluieren

```python
# Alle Runs evaluieren
promo.evaluate_all_runs()

# Spezifischen Run evaluieren
promo.evaluate_run("run001_control")

# Screening-Evaluation
screening_results = promo.evaluate_screening()
print(f"Passed screening: {len(screening_results['passed'])}")
```

### Stage-Management

```python
# Run promoten
promo.promote_run("run001_control")

# Run zur Submission einreichen
promo.submit_run("run016_best_combo_a")

# Run zurücksetzen (zu Candidate)
promo.reset_run("run001_control")

# Stage eines Runs abrufen
stage = promo.get_stage("run001_control")
print(f"Stage: {stage.value}")
```

### Runs nach Stage filtern

```python
# Alle Runs einer Stage
candidates = promo.get_by_stage(Stage.CANDIDATE)
promoted = promo.get_by_stage(Stage.PROMOTED)
submitted = promo.get_by_stage(Stage.SUBMITTED)

# Alle Runs mit Stage-Info
all_runs = promo.get_all_runs()
for run in all_runs:
    print(f"{run.run_id}: {promo.get_stage(run.run_id).value}")
```

---

## Screening-Kriterien

### Default-Kriterien

Runs müssen folgende Kriterien erfüllen, um von Candidate zu Promoted zu gelangen:

| Kriterium | Threshold | Beschreibung |
|-----------|-----------|--------------|
| `val_bpb` | < 1.5 | Maximaler BPB-Wert |
| `ms_per_step` | < 100 | Maximale Schrittzeit |
| `artifact_bytes` | < 16MB | Maximale Artefakt-Größe |
| `steps_completed` | ≥ 1000 | Mindestanzahl Schritte |

### Kill-Rules

Runs werden automatisch disqualifiziert bei:

| Regel | Bedingung | Konsequenz |
|-------|-----------|------------|
| `artifact_too_large` | artifact_bytes > 16MB | Disqualifiziert |
| `slow_without_gain` | delta_ms > 3.0 AND delta_bpb > -0.05 | Disqualifiziert |
| `quant_gap_untragbar` | quantized_val_bpb - val_bpb > 0.1 | Disqualifiziert |
| `volatil_over_seeds` | BPB-Varianz > 0.05 über Seeds | Disqualifiziert |

### Eigene Kriterien definieren

```python
from orchestrator import StageConfig

config = StageConfig(
    candidate_threshold={
        "val_bpb": 1.5,
        "ms_per_step": 100,
        "artifact_bytes": 16_000_000,
    },
    promoted_threshold={
        "val_bpb": 1.2,
        "ms_per_step": 80,
        "artifact_bytes": 12_000_000,
    },
    custom_rules=[
        {
            "name": "no_degradation",
            "condition": lambda m: m.get("delta_bpb", 0) <= 0,
            "description": "Keine BPB-Verschlechterung",
        },
    ],
)

promo = PromotionSystem(config=config)
```

---

## Promotion-Workflow

### 1. Candidate Screening

```python
# Alle Candidates screenen
screening_results = promo.evaluate_screening()

# Ergebnisse
passed = screening_results["passed"]      # Bestanden
failed = screening_results["failed"]      # Durchgefallen
disqualified = screening_results["disqualified"]  # Disqualifiziert

print(f"Passed: {len(passed)}")
print(f"Failed: {len(failed)}")
for run in failed:
    print(f"  {run.run_id}: {run.failure_reason}")
```

### 2. Promotion Evaluation

```python
# Candidates zu Promoted befördern
for candidate in promo.get_by_stage(Stage.CANDIDATE):
    metrics = promo.get_metrics(candidate.run_id)
    
    # Promotion-Kriterien prüfen
    if (
        metrics.get("val_bpb", float("inf")) < 1.2
        and metrics.get("ms_per_step", float("inf")) < 80
    ):
        promo.promote_run(candidate.run_id)
        print(f"Promoted: {candidate.run_id}")
```

### 3. Submission Preparation

```python
# Promoted Runs zur Submission einreichen
for run in promo.get_by_stage(Stage.PROMOTED):
    promo.submit_run(run.run_id)
    print(f"Submitted: {run.run_id}")
```

---

## 3-Layer Caching (Performance)

Das Promotion System verwendet 3 Cache-Ebenen für O(1) Lookups:

### Layer 1: Run Entry Cache

```python
# Memoisiert Registry-Lookups
self._run_cache: dict[str, RunEntry] = {}

def _get_run_entry(self, run_id: str) -> RunEntry | None:
    if run_id not in self._run_cache:
        self._run_cache[run_id] = self.registry.get(run_id)
    return self._run_cache[run_id]
```

### Layer 2: Stage Cache

```python
# Mappt run_id → Stage
self._stage_cache: dict[str, Stage] = {}

def get_stage(self, run_id: str) -> Stage:
    if run_id not in self._stage_cache:
        self._stage_cache[run_id] = self._determine_stage(run_id)
    return self._stage_cache[run_id]
```

### Layer 3: Runs by Stage Cache

```python
# Pre-computed Stage-Gruppierungen
self._runs_by_stage_cache: dict[Stage, list[str]] = {}

def get_by_stage(self, stage: Stage) -> list[RunEntry]:
    if stage not in self._runs_by_stage_cache:
        self._rebuild_stage_cache()
    return [self.registry.get(rid) for rid in self._runs_by_stage_cache[stage]]
```

### Performance-Vergleich

| Operation | Ohne Cache | Mit Cache | Speedup |
|-----------|------------|-----------|---------|
| Stage Lookup | O(n) | O(1) | 5x |
| Get by Stage | O(k×n) | O(1) | 5x |
| Evaluate All | O(n²) | O(n) | 5x |

---

## Integration mit anderen Komponenten

### Ablation Engine

```python
from research import AblationReporter
from orchestrator import PromotionSystem

# Ablation Reporter für Kill-Rules
reporter = AblationReporter()
promo = PromotionSystem()

# Kill-Rules anwenden
for run in promo.get_by_stage(Stage.CANDIDATE):
    metrics = promo.get_metrics(run.run_id)
    
    # Kill-Rules prüfen
    kill_reasons = reporter.evaluate_kill_rules(metrics)
    if kill_reasons:
        promo.disqualify_run(run.run_id, reasons=kill_reasons)
```

### Submission Bundle

```python
from orchestrator import SubmissionBuilder, PromotionSystem

promo = PromotionSystem()
builder = SubmissionBuilder()

# Alle submitted Runs für Bundle sammeln
submitted_runs = promo.get_by_stage(Stage.SUBMITTED)
run_ids = [r.run_id for r in submitted_runs]

# Bundle erstellen
bundle = builder.build_bundle(run_ids)
bundle.save("results/submission.zip")
```

### Dashboard CLI

```bash
# Dashboard starten
python3 -m orchestrator.dashboard

# Promotion Commands
Dashboard> list --stage candidate
Dashboard> promote run001_control
Dashboard> submit run016_best_combo_a
Dashboard> list --stage submitted
```

---

## Stage-Konfiguration anpassen

### Beispiel: Strenge Kriterien

```python
from orchestrator import StageConfig, PromotionSystem

strict_config = StageConfig(
    candidate_threshold={
        "val_bpb": 1.3,      # Strenger
        "ms_per_step": 60,   # Strenger
        "artifact_bytes": 10_000_000,  # Strenger
    },
    promoted_threshold={
        "val_bpb": 1.1,      # Sehr streng
        "ms_per_step": 40,   # Sehr streng
        "artifact_bytes": 8_000_000,   # Sehr streng
    },
)

promo = PromotionSystem(config=strict_config)
```

### Beispiel: Lockere Kriterien

```python
lenient_config = StageConfig(
    candidate_threshold={
        "val_bpb": 2.0,      # Locker
        "ms_per_step": 200,  # Locker
        "artifact_bytes": 32_000_000,  # Locker
    },
    promoted_threshold={
        "val_bpb": 1.5,      # Moderat
        "ms_per_step": 100,  # Moderat
        "artifact_bytes": 20_000_000,  # Moderat
    },
)

promo = PromotionSystem(config=lenient_config)
```

---

## Output-Struktur

```
results/
└── promotion/
    ├── promotion_state.json    # Aktueller Stage-Stand
    ├── evaluation_log.jsonl    # Evaluations-Logs
    └── reports/
        ├── screening_report.txt
        └── promotion_report.txt
```

### promotion_state.json Format

```json
{
  "last_updated": "2026-03-24T14:30:00Z",
  "stages": {
    "run001_control": "promoted",
    "run002_hash": "candidate",
    "run016_best_combo_a": "submitted",
    "run017_best_combo_quantized": "submitted"
  },
  "statistics": {
    "total_runs": 19,
    "candidates": 12,
    "promoted": 5,
    "submitted": 2,
    "disqualified": 0
  }
}
```

---

## Best Practices

### 1. Stage-Übergänge dokumentieren

```python
# Logging bei Stage-Änderungen
import logging

def promote_run(self, run_id: str):
    old_stage = self.get_stage(run_id)
    self._stage_cache[run_id] = Stage.PROMOTED
    logging.info(f"Promoted {run_id}: {old_stage} → PROMOTED")
```

### 2. Cache invalidieren bei Updates

```python
def refresh_cache(self):
    """Cache bei Registry-Updates invalidieren."""
    self._run_cache.clear()
    self._stage_cache.clear()
    self._runs_by_stage_cache.clear()
```

### 3. Batch-Operationen für große Run-Zahlen

```python
# Batch-Promotion für viele Runs
def batch_promote(self, run_ids: list[str]):
    for run_id in run_ids:
        self.promote_run(run_id)
    self._rebuild_stage_cache()  # Cache einmal neu bauen
```

---

## Troubleshooting

### Run wird nicht promotet

**Ursache:** Screening-Kriterien nicht erfüllt

**Lösung:**
```python
# Screening-Ergebnisse prüfen
screening = promo.evaluate_screening()
for run in screening["failed"]:
    print(f"{run.run_id}: {run.failure_reason}")
```

### Cache inkonsistent

**Lösung:**
```python
# Cache manuell refreshen
promo.refresh_cache()
```

### Stage nicht gefunden

**Ursache:** Run nicht in Registry

**Lösung:**
```python
# Run in Registry prüfen
entry = promo.registry.get("run001_control")
if entry is None:
    print("Run nicht in Registry")
```

---

## Verwandte Dokumente

- [sweep_guide.md](sweep_guide.md) – Sweep Runner
- [dashboard_guide.md](dashboard_guide.md) – Dashboard CLI
- [../guides/runs_guide.md](../guides/runs_guide.md) – Runs starten
- [../reports/phase_3_audit.md](../reports/phase_3_audit.md) – Phase 3 Audit
