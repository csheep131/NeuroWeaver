# Phase 5: Advanced Features - Dokumentation

## Übersicht

Phase 5 erweitert NeuroWeave um fortgeschrittene Features für Production-Einsatz:

- **Advanced Visualization & Interactive Dashboards** (Woche 11-12)
- **Real-time Monitoring & Alerting** (Woche 13-14)
- **Distributed Execution** (Woche 15-16)
- **AutoML Integration** (Woche 17-18)

## Installation

### Dependencies installieren

```bash
pip install -r requirements.txt
```

Phase 5 benötigt folgende zusätzliche Pakete:

```txt
plotly>=5.14.0 # Interaktive Plots
dash>=2.9.0 # Dashboard Framework
websockets>=10.0 # WebSocket Support
optuna>=3.0.0 # Hyperparameter Optimization
rich>=13.0.0 # Terminal UI
fuzzywuzzy>=0.18.0 # Fuzzy Search
python-Levenshtein>=0.20.0 # Fuzzy Search Speedup
```

## Komponenten

### 1. Advanced Dashboard (`orchestrator/dashboard_advanced.py`)

Interaktives Dashboard mit Plotly für umfassende Run-Analyse.

#### Features

- **3D Pareto-Frontier**: Rotierbare 3D-Visualisierung der Run-Performance
- **Metrics Time Series**: Zeitreihen aller 5 Success Metrics
- **Feature-Importance Heatmap**: Visualisierung der Feature-Wichtigkeit
- **Run-Lineage Graph**: Interaktiver Abstammungsgraph
- **Guardrail Violation Timeline**: Timeline von Guardrail-Verletzungen

#### Usage

```bash
# Dashboard starten
python3 -m orchestrator.phase5 advanced-dashboard

# Mit custom Output-Pfad
python3 -m orchestrator.phase5 advanced-dashboard --output plots/my_dashboard.html

# Mit custom Results-Verzeichnis
python3 -m orchestrator.phase5 advanced-dashboard --results-dir results
```

#### Programmatische Nutzung

```python
from core.registry import RunRegistry
from research.success_metrics import SuccessMetricsTracker
from orchestrator.dashboard_advanced import AdvancedDashboard

registry = RunRegistry("results")
tracker = SuccessMetricsTracker(registry)
dashboard = AdvancedDashboard(registry, tracker)

# Einzelne Plots
dashboard.create_pareto_3d_plot("plots/pareto_3d.html")
dashboard.create_metrics_timeseries("plots/metrics_over_time.html")
dashboard.create_feature_importance_heatmap("plots/feature_importance.html")
dashboard.create_lineage_graph("run001", "plots/lineage_run001.html")

# Komplettes Dashboard
dashboard.generate_full_dashboard("plots/phase5_dashboard.html")
```

---

### 2. Run Explorer (`orchestrator/run_explorer.py`)

Interaktive Terminal-UI für Run-Analyse mit Rich.

#### Features

- **Fuzzy Search**: Intelligente Suche nach Runs
- **Filtern**: Nach Status, Features, Budget
- **Sortieren**: Nach allen Metriken
- **Detail-Ansicht**: Vollständige Run-Informationen
- **Vergleichs-Modus**: 2-5 Runs vergleichen
- **Export**: CSV, JSON, Markdown

#### Usage

```bash
# Interaktive Session starten
python3 -m orchestrator.phase5 run-explorer

# Mit custom Results-Verzeichnis
python3 -m orchestrator.phase5 run-explorer --results-dir results
```

#### Commands

```
/search <query> - Fuzzy Search nach Runs
/filter <feature> - Nach Feature filtern
/sort <metric> - Nach Metrik sortieren
/compare <ids> - Runs vergleichen
/details <id> - Detail-Ansicht
/list - Alle Runs anzeigen
/clear - Filter zurücksetzen
/export <format> - Export (csv, json, markdown)
/help - Hilfe
/quit - Beenden
```

#### Beispiele

```bash
# Suche nach Runs mit GQA
/search gqa

# Nach completed Runs filtern
/filter completed

# Nach delta_bpb sortieren (beste zuerst)
/sort delta_bpb

# Runs vergleichen
/compare run001 run009 run015

# Als Markdown exportieren
/export markdown
```

---

### 3. Real-time Monitor (`orchestrator/realtime_monitor.py`)

Live-Monitoring von Training Runs mit WebSocket-Unterstützung.

#### Features

- **Live Loss-Curve**: Streaming Loss-Visualisierung
- **Gradient Norm Monitoring**: Live Gradient-Überwachung
- **VRAM-Usage**: Live VRAM-Verbrauch
- **Step-Time**: Rolling Average der Step-Zeiten
- **Anomaly-Erkennung**: Live-Anomalieerkennung

#### Usage

```bash
# Live Monitor für Run starten
python3 -m orchestrator.phase5 live-monitor run001

# Mit custom Port
python3 -m orchestrator.phase5 live-monitor run001 --port 8765
```

#### Programmatische Nutzung

```python
from orchestrator.realtime_monitor import RealtimeMonitor, LiveMetrics

monitor = RealtimeMonitor("run001", websocket_port=8765)

# Monitoring starten
monitor.start_monitoring()

# Metriken aktualisieren (vom Training)
monitor.update_metrics(LiveMetrics(
run_id="run001",
step=100,
loss=1.234,
gradient_norm=50.0,
vram_usage_mb=6000,
step_time_ms=150.0,
))

# Live Dashboard erstellen
monitor.create_live_dashboard("plots/live_monitor_run001.html")

# Aktuelle Metriken
metrics = monitor.get_live_metrics()
```

#### WebSocket Client

```javascript
// Browser Client
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = (event) => {
const message = JSON.parse(event.data);
if (message.type === 'metrics_update') {
console.log('Neue Metriken:', message.data);
}
};
```

---

### 4. Health Checker (`orchestrator/health_checker.py`)

Automatische Gesundheitsprüfung für Training Runs.

#### Checks

1. **Loss Divergence**: Loss > 10x Baseline
2. **Gradient Explosion**: Gradient Norm > 1000
3. **VRAM Leak**: Kontinuierlicher VRAM-Anstieg
4. **Step-Time Anomaly**: >50% langsamer
5. **NaN Detection**: NaN in Weights/Gradients

#### Usage

```bash
# Health Check durchführen
python3 -m orchestrator.phase5 health-check run001

# Demo-Modus
python3 -m orchestrator.phase5 health-check
```

#### Programmatische Nutzung

```python
from orchestrator.health_checker import TrainingHealthChecker

checker = TrainingHealthChecker()

metrics = {
"loss_history": [1.5, 1.4, 1.35, 1.3, 1.28],
"gradient_norm_history": [50, 55, 52, 48, 51],
"vram_history": [6000, 6020, 6040, 6060, 6080],
"step_time_history": [100, 102, 98, 105, 101],
}

report = checker.check_health("run001", metrics)

print(f"Health Score: {report.health_score}/100")
print(f"Status: {report.status.value}")
print(f"Issues: {[i.issue_type.value for i in report.issues]}")

# Frühwarnzeichen
warnings = checker.get_early_warning_signs("run001", metrics)
```

#### Health Report

```python
@dataclass
class HealthReport:
run_id: str
health_score: float # 0-100
status: HealthStatus # healthy, warning, critical
issues: List[HealthIssue]
recommendations: List[str]
```

---

### 5. Distributed Runner (`orchestrator/distributed_runner.py`)

Verteilte Run-Ausführung mit Multi-GPU Support.

#### Features

- **Multi-GPU Support**: Parallele Ausführung auf mehreren GPUs
- **Load Balancing**: Automatische Lastverteilung
- **Fault Tolerance**: Retry bei Failure
- **Progress Tracking**: Live-Fortschrittsanzeige
- **Result Aggregation**: Zentrale Ergebnissammlung

#### Usage

```bash
# Distributed Runner starten
python3 -m orchestrator.phase5 distributed-runner

# Mit 4 Workern
python3 -m orchestrator.phase5 distributed-runner --workers 4

# Max 2 Runs pro Worker
python3 -m orchestrator.phase5 distributed-runner --max-concurrent 2
```

#### Programmatische Nutzung

```python
from orchestrator.distributed_runner import DistributedRunner, WorkerConfig

workers = [
WorkerConfig("worker_0", gpu_id=0, max_concurrent_runs=2),
WorkerConfig("worker_1", gpu_id=1, max_concurrent_runs=2),
]

runner = DistributedRunner(workers)

# Runs einreichen
batch_id = runner.submit_runs([
{"depth": 12, "width": 512, "run_id": "run_001"},
{"depth": 14, "width": 640, "run_id": "run_002"},
])

# Status prüfen
status = runner.get_batch_status(batch_id)
print(f"Completed: {status['completed']}/{status['total']}")

# Worker Load
load = runner.get_worker_load()

# Auto-Scale
scaling = runner.auto_scale(target_gpu_util=0.8)

# Starten
runner.start()

# Stoppen
runner.stop()
```

---

### 6. Run Queue Manager (`orchestrator/run_queue.py`)

Priority-basierte Queue für Run-Scheduling.

#### Features

- **Priority Scoring**: Basierend auf Hypothesis Confidence
- **Preemption**: Wichtige Runs können vorziehen
- **Fair Scheduling**: Verhindert Starvation
- **Deadline Awareness**: Berücksichtigt Deadlines

#### Usage

```bash
# Queue Manager starten
python3 -m orchestrator.phase5 queue-manager

# Mit Test-Runs
python3 -m orchestrator.phase5 queue-manager --num-runs 20
```

#### Programmatische Nutzung

```python
from orchestrator.run_queue import RunQueueManager
from datetime import datetime, timedelta

manager = RunQueueManager()

# Runs einreihen
run_id = manager.enqueue(
run_config={"depth": 12, "width": 512},
priority=0.8,
deadline=datetime.utcnow() + timedelta(hours=2),
hypothesis_confidence=0.75,
tags=["gqa", "film"],
)

# Nächsten Run für Worker holen
config = manager.dequeue("worker_0")

# Position in Queue
position = manager.get_position(run_id)

# Priorität ändern
manager.reprioritize(run_id, 0.9)

# Queue-Statistiken
stats = manager.get_queue_stats()
print(f"Queue Length: {stats['queue_length']}")
print(f"Avg Wait Time: {stats['avg_wait_time']}")
```

---

### 7. HPO Integration (`research/hpo_integration.py`)

Hyperparameter-Optimierung mit Optuna.

#### Features

- **Bayesian Optimization**: Effiziente Hyperparameter-Suche
- **Multi-Objective**: Optimierung für BPB, Efficiency, Size
- **Pruning**: Unpromising Runs früh stoppen
- **Transfer Learning**: Von ähnlichen Runs lernen

#### Usage

```bash
# HPO starten
python3 -m orchestrator.phase5 hpo-optimize

# Mit 100 Trials
python3 -m orchestrator.phase5 hpo-optimize --trials 100

# Mit JSON-Export
python3 -m orchestrator.phase5 hpo-optimize --trials 50 --output hpo_results.json
```

#### Programmatische Nutzung

```python
from research.hpo_integration import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(registry)

# Config vorschlagen
config = optimizer.suggest_config(trial_number=0)
print(f"Vorgeschlagene Config: {config}")

# Training ausführen und Ergebnis reporten
metrics = run_training(config)
optimizer.report_result(trial_number=0, metrics=metrics)

# Beste Configs
best_configs = optimizer.get_best_configs(top_k=5)

# Visualisierung
optimizer.create_study_visualization("plots/hpo_study.html")

# Export
optimizer.export_study("hpo_results.json")
```

#### Search Space

```python
optimizer._search_space = {
"depth": (8, 16),
"width": (256, 1024),
"mlp_ratio": (2.0, 5.0),
"learning_rate": (1e-5, 1e-2),
"weight_decay": (0.01, 0.1),
"attention_type": ["standard", "gqa", "xsa"],
"activation": ["gelu", "swiglu", "leaky_relu"],
}
```

---

### 8. NAS Integration (`research/nas_integration.py`)

Neural Architecture Search.

#### Features

- **Search Space**: Depth, Width, Attention-Typen
- **Search Strategy**: Evolutionary / Reinforcement Learning
- **Performance Prediction**: Via Surrogate Scorer
- **Constraint Handling**: VRAM, Time Budget

#### Usage

```bash
# NAS Search starten
python3 -m orchestrator.phase5 nas-search

# Mit Budget 100
python3 -m orchestrator.phase5 nas-search --budget 100

# Mit custom Constraints
python3 -m orchestrator.phase5 nas-search \
--budget 100 \
--max-vram 8000 \
--max-size 500 \
--min-depth 8 \
--max-depth 16 \
--output nas_results.json
```

#### Programmatische Nutzung

```python
from research.nas_integration import NASIntegration

nas = NASIntegration(registry)

# Search Space definieren
nas.define_search_space(
max_vram_mb=8000,
max_size_mb=500,
depth_range=(8, 16),
width_range=(256, 1024),
)

# Suche durchführen
pareto_frontier = nas.search(budget=100)

# Tradeoffs analysieren
report = nas.get_architecture_tradeoffs()
print(report)

# Export
nas.export_architectures("nas_architectures.json")
```

#### Architecture

```python
@dataclass
class Architecture:
arch_id: str
depth: int
width: int
mlp_ratio: float
attention_type: Literal["standard", "gqa", "xsa"]
activation: Literal["gelu", "swiglu", "leaky_relu"]
num_heads: int = 8
head_dim: Optional[int] = None

# Metriken
delta_bpb: Optional[float] = None
efficiency_gain: Optional[float] = None
size_mb: Optional[float] = None
fitness: float = 0.0
```

---

## Phase 5 CLI

Alle Phase 5 Features sind über eine zentrale CLI verfügbar:

```bash
python3 -m orchestrator.phase5 <command> [options]
```

### Verfügbare Commands

| Command | Beschreibung |
|---------|-------------|
| `advanced-dashboard` | Interaktives Dashboard mit Plotly |
| `run-explorer` | Interaktive Run-Analyse CLI |
| `live-monitor <run_id>` | Live Monitoring für Run |
| `health-check <run_id>` | Health Check für Run |
| `distributed-runner` | Distributed Execution |
| `queue-manager` | Queue Management |
| `hpo-optimize` | Hyperparameter-Optimierung |
| `nas-search` | Neural Architecture Search |

---

## Tests

Phase 5 umfasst ~165 Tests für alle Komponenten:

```bash
# Alle Phase 5 Tests
pytest tests/test_phase5.py -v

# Spezifische Test-Klassen
pytest tests/test_phase5.py::TestAdvancedDashboard -v
pytest tests/test_phase5.py::TestRunExplorer -v
pytest tests/test_phase5.py::TestHealthChecker -v
pytest tests/test_phase5.py::TestDistributedRunner -v
pytest tests/test_phase5.py::TestRunQueue -v
pytest tests/test_phase5.py::TestHPOIntegration -v
pytest tests/test_phase5.py::TestNASIntegration -v
```

---

## File Structure

```
NeuroWeave/
orchestrator/
dashboard_advanced.py # Advanced Dashboard
run_explorer.py # Run Explorer CLI
realtime_monitor.py # Real-time Monitor
health_checker.py # Health Checker
distributed_runner.py # Distributed Runner
run_queue.py # Run Queue Manager
phase5.py # Phase 5 CLI
research/
hpo_integration.py # HPO Integration
nas_integration.py # NAS Integration
tests/
test_phase5.py # Phase 5 Tests
plots/ # Generated Plots
phase5_dashboard.html
pareto_3d.html
metrics_timeseries.html
feature_importance.html
live_monitor_*.html
hpo_study.html
nas_architectures.json
requirements.txt
```

---

## Migration von Phase 4

Phase 5 ist vollständig abwärtskompatibel zu Phase 4:

```bash
# Phase 4 Commands funktionieren weiterhin
python3 -m orchestrator.phase4_orchestrator run
python3 -m orchestrator.phase4_orchestrator status
python3 -m orchestrator.phase4_orchestrator report

# Phase 5 Commands parallel nutzbar
python3 -m orchestrator.phase5 advanced-dashboard
python3 -m orchestrator.phase5 run-explorer
```

---

## Performance-Empfehlungen

### Dashboard

- Bei vielen Runs (>1000): Filter vor Dashboard-Generierung
- 3D Plots: Browser mit Hardware-Beschleunigung verwenden

### Distributed Runner

- Worker-Anzahl an verfügbare GPUs anpassen
- Memory-Limits basierend auf VRAM-Größe setzen
- Auto-Scale für variable Workloads aktivieren

### HPO

- Multi-Objective für komplexe Tradeoffs
- Pruning für effiziente Suche aktivieren
- Transfer Learning bei ähnlichen Tasks

### NAS

- Budget basierend auf verfügbaren Ressourcen
- Constraints für VRAM-Limits setzen
- Evolutionary Strategy für große Search Spaces

---

## Troubleshooting

### Plotly nicht installiert

```bash
pip install plotly dash
```

### Rich nicht installiert

```bash
pip install rich
```

### Optuna nicht installiert

```bash
pip install optuna
```

### WebSocket Connection Failed

- Prüfe ob Port 8765 verfügbar ist
- Firewall-Einstellungen prüfen
- Custom Port mit `--port` verwenden

### Queue Starvation

- Priority zu niedrig: `priority > 0.7` setzen
- Deadline zu weit: `deadline < 2h` setzen
- Fair Scheduling temporär deaktivieren

---

## Changelog Phase 5

### v5.0.0 (2026-03-25)

- Advanced Dashboard mit Plotly
- Run Explorer CLI mit Rich
- Real-time Monitor mit WebSocket
- Health Checker mit Anomaly Detection
- Distributed Runner mit Multi-GPU
- Run Queue Manager mit Priority
- HPO Integration mit Optuna
- NAS Integration mit Evolutionary Search
- ~165 Tests für alle Komponenten

---

## Support

Bei Fragen oder Problemen:

1. Dokumentation prüfen
2. Tests als Referenz verwenden
3. Issues mit detaillierter Beschreibung erstellen
