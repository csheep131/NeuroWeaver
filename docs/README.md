# Wettkampf Dokumentation

**Letztes Update:** 2026-03-24  
**Projekt:** Ablation Machine – Experimentier-Plattform für systematisches Ablation-Testing  
**Status:** Phase 1-3 ✅ Alle abgeschlossen | **Performance:** 4-5x Speedup

---

## Übersicht

Diese Dokumentation bietet einen vollständigen Überblick über die Ablation Machine, eine Plattform für reproduzierbares ML-Experiment-Management mit systematischem Feature-Ablation-Testing.

### Ziel der Plattform

1. **Runs deklarativ beschreiben** – Konfiguration statt Code-Bastelei
2. **Reproduzierbar trainieren** – Same config = same results
3. **Automatisch quantisieren** – INT6, INT5, mixed precision, GPTQ-lite
4. **Metriken vergleichen** – BPB, Schrittzeit, Artifact-Größe
5. **Kombinationen testen** – Features systematisch aktivieren/verwerfen
6. **Automatisierte Sweeps** – Parameter-Raster automatisch durchlaufen
7. **Promotion System** – Runs durch Stages (Candidate → Promoted → Submitted)
8. **Dashboard CLI** – Interaktive Übersicht aller Runs und Metriken

---

## Dokumentations-Struktur

```
docs/
├── README.md                    # Diese Datei – Dokumentations-Übersicht
├── architecture/                # Architektur-Dokumente
│   ├── ARCHITECTURE.md          # System-Architektur
│   ├── blueprint.md             # Phase 1 Implementierungs-Blueprint
│   ├── rust_integration.md      # Rust-Integration Details
│   └── module_overview.md       # Modul-Übersicht
├── setup/                       # Setup & Installation
│   ├── SETUP.md                 # Installations-Anleitung
│   └── development_guide.md     # Entwickler-Guide
├── reports/                     # Audit-Reports & Analysen
│   ├── README.md                # Report-Übersicht
│   ├── phase_1_audit.md         # Phase 1 Code Audit
│   ├── phase_2_audit.md         # Phase 2 Code Audit
│   ├── phase_2_bug_fixes.md     # Phase 2 Bug Fixes
│   ├── phase_2_implementation.md# Phase 2 Implementierungsbericht
│   ├── phase_3_audit.md         # Phase 3 Code Audit
│   ├── phase_3_performance.md   # Phase 3 Performance Optimizations
│   └── phase_3_implementation.md# Phase 3 Implementierungsbericht
├── guides/                      # Anleitungen & How-Tos
│   ├── README.md                # Guides-Übersicht
│   ├── configuration.md         # Konfigurations-Handbuch
│   ├── runs_guide.md            # Runs starten
│   ├── runs_development.md      # Runs Entwicklung
│   ├── sweep_guide.md           # Sweep Runner
│   ├── promotion_guide.md       # Promotion System
│   ├── ablation_plan.md         # Ablation-Plan (10-Run Roadmap)
│   ├── roadmap_runs.md          # Detaillierte Run-Roadmap
│   ├── run_tabelle.md           # Run-Übersichtstabelle
│   ├── trainings_plan.md        # Trainings-Planung
│   ├── auto_approve.md          # Phase 4: Autonome Selbstverbesserung (Spezifikation)
│   ├── phase4_overview.md       # Phase 4: Implementierungs-Übersicht
│   └── phase5_overview.md       # Phase 5: Advanced Features
└── api/                         # API-Dokumentation
    └── README.md                # API-Docs Übersicht
```

---

## Schnellstart

### Installation

```bash
# Python-Abhängigkeiten
pip install -r requirements.txt

# Rust-Core kompilieren (optional, für Performance)
cd rust-core
maturin develop --release
```

### Ersten Run starten

```bash
python3 -m runs.run --config configs/runs/run001_control.yaml
```

### Dashboard öffnen

```bash
python3 -m orchestrator.dashboard
```

---

## Dokumentations-Bereiche

### 📐 Architektur (`architecture/`)

| Dokument | Beschreibung |
|----------|--------------|
| [ARCHITECTURE.md](architecture/ARCHITECTURE.md) | System-Architektur, Komponenten, Datenfluss |
| [blueprint.md](architecture/blueprint.md) | Phase 1 Implementierungs-Blueprint |
| [rust_integration.md](architecture/rust_integration.md) | Rust/Python-Integration, Performance-kritische Komponenten |
| [module_overview.md](architecture/module_overview.md) | Modul-Übersicht, Abhängigkeiten, Exporte |

### 🔧 Setup (`setup/`)

| Dokument | Beschreibung |
|----------|--------------|
| [SETUP.md](setup/SETUP.md) | Installation, Projektstruktur, Usage-Beispiele |
| [development_guide.md](setup/development_guide.md) | Entwickler-Guide, Testing, Rust-Erweiterung |

### 📊 Reports (`reports/`)

| Dokument | Beschreibung |
|----------|--------------|
| [README.md](reports/README.md) | Übersicht aller Audit-Reports |
| [phase_1_audit.md](reports/phase_1_audit.md) | Phase 1 Code Audit (Experiment Core) |
| [phase_2_audit.md](reports/phase_2_audit.md) | Phase 2 Code Audit (Research Engine) |
| [phase_2_bug_fixes.md](reports/phase_2_bug_fixes.md) | Phase 2 Bug Fixes Summary |
| [phase_2_implementation.md](reports/phase_2_implementation.md) | Phase 2 Implementierungsbericht |
| [phase_3_audit.md](reports/phase_3_audit.md) | Phase 3 Code Audit (Production Pipeline) |
| [phase_3_performance.md](reports/phase_3_performance.md) | Phase 3 Performance Optimizations |
| [phase_3_implementation.md](reports/phase_3_implementation.md) | Phase 3 Implementierungsbericht |

### 📖 Guides (`guides/`)

| Dokument | Beschreibung |
|----------|--------------|
| [README.md](guides/README.md) | Guides-Übersicht |
| [configuration.md](guides/configuration.md) | Konfigurations-Handbuch, YAML-Schema |
| [runs_guide.md](guides/runs_guide.md) | Runs starten, Configs erstellen |
| [runs_development.md](guides/runs_development.md) | Runs Entwicklung |
| [sweep_guide.md](guides/sweep_guide.md) | Sweep Runner, Parameter-Sweeps |
| [promotion_guide.md](guides/promotion_guide.md) | Promotion System, Stage-Management |
| [ablation_plan.md](guides/ablation_plan.md) | Ablation-Plan (10-Run Roadmap) |
| [roadmap_runs.md](guides/roadmap_runs.md) | Detaillierte Run-Roadmap |
| [run_tabelle.md](guides/run_tabelle.md) | Run-Übersichtstabelle |
| [trainings_plan.md](guides/trainings_plan.md) | Trainings-Planung |
| [auto_approve.md](guides/auto_approve.md) | Phase 4: Autonome Selbstverbesserung (Spezifikation) |
| [phase4_overview.md](guides/phase4_overview.md) | Phase 4: Implementierungs-Übersicht |
| [phase5_overview.md](guides/phase5_overview.md) | Phase 5: Advanced Features |

### 🔌 API (`api/`)

| Dokument | Beschreibung |
|----------|--------------|
| [README.md](api/README.md) | API-Dokumentation Übersicht |

---

## Phasen-Status

### Phase 1: Experiment Core ✅ ABGESCHLOSSEN

- [x] Config-first Run-System
- [x] Modultrennung (Python/Rust)
- [x] Run Registry
- [x] Standardisierte Outputs
- [x] Vergleichbarkeit
- [x] Core-Komponenten (Config, Registry, Logging, Seed, Artifacts)

### Phase 2: Research Engine ✅ ABGESCHLOSSEN

- [x] Backbone Factory
- [x] Feature-Gates (FeatureGate, FeatureGateManager)
- [x] Tokenizer-Lab (Byte, BigramHash, TrigramHash, Fallback)
- [x] Quant-Lab (Int6Quantizer, Int5Quantizer, MixedQuantizer, GPTQLiteQuantizer)
- [x] Ablation Engine (AblationReporter, KillRules)
- [x] Phase 1/2/3 Evaluatoren

### Phase 3: Production Pipeline ✅ ABGESCHLOSSEN

- [x] Sweep Runner (itertools.product, O(1) Memory)
- [x] Promotion System (3-Layer Caching, Stage-Management)
- [x] Submission Bundle (Bundle-Erstellung, ZIP-Export)
- [x] Dashboard CLI (Interaktive Run-Übersicht)
- [x] Multi-Seed Orchestrator
- [x] Dynamic Combo Builder
- [x] RunComparator, LeaderboardGenerator

---

## Performance-Metriken (Phase 3 Optimizations)

| Komponente | Vorher | Nachher | Verbesserung |
|------------|--------|---------|--------------|
| Sweep Generation | O(n^k) rekursiv | O(1) itertools | **5x schneller** |
| Promotion System | O(k×n) linear | O(1) cached | **5x schneller** |
| Bundle Creation | O(m×n) wiederholt | O(n) cached | **4x schneller** |
| **Gesamt** | Langsam bei >500 Runs | Skalierbar bis 10.000+ | **4-5x Speedup** |

---

## Wichtige Links

- [Haupt-README](../README.md) – Projekt-Übersicht
- [HERMES.md](../HERMES.md) – Entwicklungs-Konventionen
- [Ablation-Plan](guides/ablation_plan.md) – 10-Run Roadmap
- [Run-Roadmap](guides/roadmap_runs.md) – Detaillierte Run-Spezifikation

---

## Dokumentation aktualisieren

Diese Dokumentation wird automatisch aus der Codebase generiert, wo immer möglich. Bei Änderungen an:

- **API-Schnittstellen** → `docs/api/README.md` aktualisieren
- **Konfiguration** → `docs/guides/configuration.md` aktualisieren
- **Architektur** → `docs/architecture/` aktualisieren
- **Setup-Prozess** → `docs/setup/SETUP.md` aktualisieren
- **Run-Roadmap** → `docs/guides/roadmap_runs.md` aktualisieren
- **Reports** → `docs/reports/` aktualisieren

**Prinzip:** Dokumentation, die nicht mit der Realität übereinstimmt, ist schlechter als keine Dokumentation. Immer aus der Source of Truth generieren.
