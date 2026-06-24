# Guides Übersicht

**Letztes Update:** 2026-03-24
**Gesamtanzahl Guides:** 12

---

## Übersicht

Dieses Verzeichnis enthält praktische Anleitungen und How-Tos für die Arbeit mit der Ablation Machine. Jeder Guide bietet schrittweise Anleitungen für spezifische Aufgaben.

---

## Guides nach Kategorie

### Konfiguration & Setup

| Guide | Beschreibung |
|-------|--------------|
| [configuration.md](configuration.md) | Konfigurations-Handbuch, YAML-Schema, Base- und Run-Configs |

### Runs & Training

| Guide | Beschreibung |
|-------|--------------|
| [runs_guide.md](runs_guide.md) | Runs starten, Configs erstellen, Run-Typen (smoke, proxy, submission) |
| [runs_development.md](runs_development.md) | Runs Entwicklung, Testing, Debugging |
| [trainings_plan.md](trainings_plan.md) | Trainings-Planung, Hyperparameter, Scheduling |

### Ablation-Testing

| Guide | Beschreibung |
|-------|--------------|
| [ablation_plan.md](ablation_plan.md) | 10-Run Roadmap für systematisches Ablation-Testing |
| [roadmap_runs.md](roadmap_runs.md) | Detaillierte Run-Roadmap mit 28+ Runs, Phasen, Gate-Freeze |
| [run_tabelle.md](run_tabelle.md) | Run-Übersichtstabelle mit Priorisierung und Status |

### Automation

| Guide | Beschreibung |
|-------|--------------|
| [sweep_guide.md](sweep_guide.md) | Sweep Runner, Parameter-Sweeps, Kombinationen generieren |
| [promotion_guide.md](promotion_guide.md) | Promotion System, Stage-Management, Candidate → Promoted → Submitted |

### Phase 4: Autonome Selbstverbesserung

| Guide | Beschreibung |
|-------|--------------|
| [auto_approve.md](auto_approve.md) | Phase 4 Plan: Adaptive Search, Self-Diagnosis, Guided Autonomy |
| [phase4_overview.md](phase4_overview.md) | Implementierungs-Übersicht, Quick Start, Architektur |

### Phase 5: Advanced Features

| Guide | Beschreibung |
|-------|--------------|
| [phase5_overview.md](phase5_overview.md) | Advanced Visualization, Distributed Execution, AutoML |

---

## Schnellstart nach Use-Case

### Ersten Run starten
1. [configuration.md](configuration.md) – Config verstehen
2. [runs_guide.md](runs_guide.md) – Run ausführen
3. [runs_development.md](runs_development.md) – Ergebnisse analysieren

### Ablation-Testing durchführen
1. [ablation_plan.md](ablation_plan.md) – 10-Run Roadmap lesen
2. [roadmap_runs.md](roadmap_runs.md) – Detaillierte Spezifikation
3. [run_tabelle.md](run_tabelle.md) – Quick-Reference

### Sweeps ausführen
1. [sweep_guide.md](sweep_guide.md) – Sweep Runner verwenden
2. [promotion_guide.md](promotion_guide.md) – Beste Runs promoten

### Phase 4 Planung
1. [auto_approve.md](auto_approve.md) – Autonome Selbstverbesserung planen

---

## Run-Typen

| Typ | Zweck | Hardware | Dauer |
|-----|-------|----------|-------|
| **smoke** | Startvalidierung, kein OOM | 8GB lokal | ~1 Min |
| **proxy** | Kurzer Vergleich, relative Tendenzen | 8GB lokal | ~30 Min |
| **submission** | Echte Challenge-Konfiguration | H100 remote | ~4h |

---

## Phasen-Übersicht

### Phase 1: Experiment Core
- Config-first Run-System
- Modultrennung (Python/Rust)
- Run Registry
- Standardisierte Outputs

**Relevante Guides:**
- [configuration.md](configuration.md)
- [runs_guide.md](runs_guide.md)

### Phase 2: Research Engine
- Backbone Factory
- Feature-Gates
- Tokenizer-Lab
- Quant-Lab
- Ablation Engine

**Relevante Guides:**
- [ablation_plan.md](ablation_plan.md)
- [roadmap_runs.md](roadmap_runs.md)

### Phase 3: Production Pipeline
- Sweep Runner
- Promotion System
- Submission Bundle
- Dashboard CLI

**Relevante Guides:**
- [sweep_guide.md](sweep_guide.md)
- [promotion_guide.md](promotion_guide.md)

# Phase 4: Autonome Selbstverbesserung IN PLANUNG → IMPLEMENTIERT
- Adaptive Search Engine (4A)
- Self-Diagnosis & Recovery (4B)
- Guided Autonomy (4C)
- Evaluation & Refinement (4D)

**Relevante Guides:**
- [auto_approve.md](auto_approve.md) – Vollständige Spezifikation
- [phase4_overview.md](phase4_overview.md) – Implementierungs-Übersicht

**Quick Start:**
```bash
# Meta-Features extrahieren
python3 scripts/enrich_historical_runs.py --all

# Dashboard starten
python3 -m orchestrator.meta_dashboard summary

# Run-Vorschläge generieren
python3 -m orchestrator.meta_dashboard hypotheses --top 10

# Autonomer Zyklus
python3 -m orchestrator.phase4_orchestrator run
```

# Phase 5: Advanced Features IMPLEMENTIERT
- Advanced Visualization (Plotly, Dash)
- Real-time Monitoring
- Distributed Execution (Multi-GPU)
- AutoML Integration (HPO, NAS)

**Relevante Guides:**
- [phase5_overview.md](phase5_overview.md) – Implementierungs-Übersicht

**Quick Start:**
```bash
# Advanced Dashboard
python3 -m orchestrator.phase5 advanced-dashboard

# Run Explorer
python3 -m orchestrator.phase5 run-explorer

# Live Monitoring
python3 -m orchestrator.phase5 live-monitor run001

# Hyperparameter-Optimierung
python3 -m orchestrator.phase5 hpo-optimize --trials 50

# Neural Architecture Search
python3 -m orchestrator.phase5 nas-search --budget 100
```

---

## Wichtige Konzepte

### Gate-Freeze vor Phase 3

Bevor Phase 3 (Kombinations-Runs) beginnt, müssen alle Phase-2-Runs abgeschlossen und bewertet werden:

| Gate-Label | Bedeutung | Verwendung |
|------------|-----------|------------|
| PASS | Stabil positiv in ≥2 Runs | Darf kombiniert werden |
| WATCH | Gemischt oder knapp positiv | Nur einzeln weiter testen |
| FAIL | Negativ oder keine Verbesserung | Nicht in Phase 3 verwenden |

**Mehr Details:** [roadmap_runs.md](roadmap_runs.md#gate-freeze-vor-phase-3)

### Kill-Regeln

Runs werden automatisch verworfen bei:

1. Artifact > 16.000.000 bytes
2. ms/step deutlich schlechter ohne BPB-Gewinn
3. Quant-Gap untragbar
4. Feature volatil über Seeds
5. Kombi macht Debugging unmöglich

**Mehr Details:** [roadmap_runs.md](roadmap_runs.md#kill-regeln)

---

## Verwandte Dokumente

- [docs/README.md](../README.md) – Haupt-Dokumentationsübersicht
- [docs/architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md) – System-Architektur
- [docs/reports/README.md](../reports/README.md) – Audit-Reports
- [docs/setup/SETUP.md](../setup/SETUP.md) – Installation
