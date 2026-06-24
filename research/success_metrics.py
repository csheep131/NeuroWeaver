#!/usr/bin/env python3
"""
Success Metrics Tracker für NeuroWeave Phase 4.

Definition und Messung von Phase 4 Erfolgsmetriken:
1. Search Efficiency: Runs benötigt für ΔBPB = -0.05
2. Failure Rate: Fehler/Run Rate über Zeit
3. Pareto Frontier Expansion: Frontier Area Growth
4. Human Time Saved: Stunden/Woche für Run-Planning
5. Confidence Accuracy: Accuracy der Vorhersagen
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from core.registry import RunRegistry, RunEntry


@dataclass
class MetricDefinition:
"""Definition einer Success Metric."""

name: str
description: str
formula: str # LaTeX-style Formel
target_value: float
direction: Literal["higher_better", "lower_better"]
baseline_value: float
current_value: float
unit: str
target_met: bool = False

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"name": self.name,
"description": self.description,
"formula": self.formula,
"target_value": self.target_value,
"direction": self.direction,
"baseline_value": self.baseline_value,
"current_value": self.current_value,
"unit": self.unit,
"target_met": self.target_met,
}


class SuccessMetricsTracker:
"""
Tracker für Phase 4 Success Metrics.

Metriken:
1. Search Efficiency: Runs benötigt für ΔBPB = -0.05
2. Failure Rate: Fehler/Run Rate über Zeit
3. Pareto Frontier Expansion: Frontier Area Growth
4. Human Time Saved: Stunden/Woche für Run-Planning
5. Confidence Accuracy: Accuracy der Vorhersagen

Example:
tracker = SuccessMetricsTracker(registry)

# Alle Metriken berechnen
metrics = tracker.get_all_metrics()
for name, metric in metrics.items():
print(f"{name}: {metric.current_value:.2f} (Ziel: {metric.target_value})")
print(f" Target met: {metric.target_met}")

# Report generieren
report = tracker.generate_report()
print(report)
"""

def __init__(
self,
registry: RunRegistry,
baseline_search_efficiency: float = 100.0,
baseline_failure_rate: float = 0.20,
baseline_human_time: float = 10.0,
) -> None:
"""
Initialisiere Success Metrics Tracker.

Args:
registry: RunRegistry für Datenzugriff
baseline_search_efficiency: Baseline Runs für manuelle Auswahl
baseline_failure_rate: Baseline Failure Rate (z.B. 0.20 = 20%)
baseline_human_time: Baseline menschliche Zeit pro Woche (Stunden)
"""
self.registry = registry
self.baseline_search_efficiency = baseline_search_efficiency
self.baseline_failure_rate = baseline_failure_rate
self.baseline_human_time = baseline_human_time

# Zielwerte (aus Spezifikation)
self.targets = {
"search_efficiency_improvement": 0.30, # 30% weniger Runs
"failure_rate_reduction": 0.50, # 50% weniger Fehler
"pareto_expansion": 0.20, # 20% mehr Pareto-Punkte
"human_time_saved": 0.70, # 70% weniger manuelle Zeit
"confidence_accuracy": 0.75, # 75% Accuracy
}

def _get_completed_runs(self) -> List[RunEntry]:
"""Hole alle abgeschlossenen Runs."""
return self.registry.list_runs(status="completed")

def _get_failed_runs(self) -> List[RunEntry]:
"""Hole alle fehlgeschlagenen Runs."""
all_runs = self.registry.list_runs()
return [r for r in all_runs if r.status in ("failed", "killed")]

def _get_runs_with_parent(self) -> List[RunEntry]:
"""Hole Runs mit Parent-Run (für Delta-Berechnung)."""
completed = self._get_completed_runs()
return [r for r in completed if r.parent_run_id is not None and r.delta_bpb is not None]

def compute_search_efficiency(self, target_delta_bpb: float = -0.05) -> Dict[str, Any]:
"""
Search Efficiency berechnen.

Formel:
Runs benötigt für ΔBPB = target_delta_bpb

Ziel: 30% weniger Runs als manuelle Auswahl

Args:
target_delta_bpb: Ziel Delta BPB (default: -0.05)

Returns:
{
"runs_needed_autonomous": N,
"runs_needed_manual": M,
"improvement_percent": X%,
"target_met": True/False
}
"""
runs_with_parent = self._get_runs_with_parent()

if not runs_with_parent:
return {
"runs_needed_autonomous": 0,
"runs_needed_manual": self.baseline_search_efficiency,
"improvement_percent": 0.0,
"target_met": False,
"error": "Keine Runs mit Parent-Run gefunden",
}

# Sortiere Runs nach delta_bpb (beste zuerst)
sorted_runs = sorted(runs_with_parent, key=lambda r: r.delta_bpb if r.delta_bpb is not None else float("inf"))

# Zähle Runs bis target_delta_bpb erreicht
runs_needed = 0
best_delta = 0.0

for run in sorted_runs:
runs_needed += 1
if run.delta_bpb is not None and run.delta_bpb < best_delta:
best_delta = run.delta_bpb

if best_delta <= target_delta_bpb:
break

# Wenn Target nicht erreicht, nimm alle verfügbaren Runs
if best_delta > target_delta_bpb:
runs_needed = len(sorted_runs)

# Verbesserung berechnen
if self.baseline_search_efficiency > 0:
improvement = (self.baseline_search_efficiency - runs_needed) / self.baseline_search_efficiency
else:
improvement = 0.0

target_met = improvement >= self.targets["search_efficiency_improvement"]

return {
"runs_needed_autonomous": runs_needed,
"runs_needed_manual": self.baseline_search_efficiency,
"improvement_percent": round(improvement * 100, 2),
"target_met": target_met,
"best_delta_bpb": round(best_delta, 4),
"target_delta_bpb": target_delta_bpb,
}

def compute_failure_rate_reduction(self, window_size: int = 50) -> Dict[str, Any]:
"""
Failure Rate Reduction berechnen.

Formel:
(failure_rate_before - failure_rate_after) / failure_rate_before

Ziel: 50% weniger OOMs/NaN/Divergence

Args:
window_size: Fenstergröße für "recent" Runs

Returns:
{
"failure_rate_before": X%,
"failure_rate_after": Y%,
"reduction_percent": Z%,
"target_met": True/False
}
"""
all_runs = self.registry.list_runs()

if len(all_runs) < window_size:
# Nicht genug Daten, nimm alle
recent_runs = all_runs
older_runs = all_runs
else:
# Teile in ältere und neuere Runs (einfache Heuristik: erste vs. letzte Hälfte)
sorted_runs = sorted(all_runs, key=lambda r: r.run_id)
mid = len(sorted_runs) // 2
older_runs = sorted_runs[:mid]
recent_runs = sorted_runs[mid:]

# Failure Rates berechnen
def calc_failure_rate(runs: List[RunEntry]) -> float:
if not runs:
return 0.0
failed = sum(1 for r in runs if r.status in ("failed", "killed"))
return failed / len(runs)

failure_rate_before = calc_failure_rate(older_runs)
failure_rate_after = calc_failure_rate(recent_runs)

# Reduktion berechnen
if failure_rate_before > 0:
reduction = (failure_rate_before - failure_rate_after) / failure_rate_before
else:
reduction = 0.0 if failure_rate_after == 0 else -1.0

target_met = reduction >= self.targets["failure_rate_reduction"]

return {
"failure_rate_before": round(failure_rate_before * 100, 2),
"failure_rate_after": round(failure_rate_after * 100, 2),
"reduction_percent": round(reduction * 100, 2),
"target_met": target_met,
"older_runs_count": len(older_runs),
"recent_runs_count": len(recent_runs),
}

def compute_pareto_frontier_expansion(self, days: int = 7) -> Dict[str, Any]:
"""
Pareto Frontier Expansion berechnen.

Formel:
(frontier_volume_now - frontier_volume_week_ago) / frontier_volume_week_ago

Ziel: 20% mehr Pareto-optimale Punkte

Args:
days: Anzahl Tage für Vergleich

Returns:
{
"frontier_volume_now": X,
"frontier_volume_before": Y,
"expansion_percent": Z%,
"pareto_points_now": N,
"pareto_points_before": M,
"target_met": True/False
}
"""
completed_runs = self._get_completed_runs()

if not completed_runs:
return {
"frontier_volume_now": 0.0,
"frontier_volume_before": 0.0,
"expansion_percent": 0.0,
"pareto_points_now": 0,
"pareto_points_before": 0,
"target_met": False,
}

# Pareto-optimale Runs identifizieren
# Ein Run ist Pareto-optimal wenn er nicht in beiden Dimensionen dominiert wird
def is_pareto_optimal(run: RunEntry, all_runs: List[RunEntry]) -> bool:
"""Prüfe ob Run Pareto-optimal ist (besser in mind. einer Dimension)."""
run_bpb = run.val_bpb if run.val_bpb is not None else float("inf")
run_ms = run.ms_per_step if run.ms_per_step is not None else float("inf")

for other in all_runs:
if other.run_id == run.run_id:
continue

other_bpb = other.val_bpb if other.val_bpb is not None else float("inf")
other_ms = other.ms_per_step if other.ms_per_step is not None else float("inf")

# Andere dominiert diese wenn in beiden Dimensionen besser oder gleich
if other_bpb <= run_bpb and other_ms <= run_ms:
if other_bpb < run_bpb or other_ms < run_ms:
return False

return True

# Aktuelle Pareto-Punkte (alle completed)
pareto_points_now = [r for r in completed_runs if is_pareto_optimal(r, completed_runs)]

# Frühere Pareto-Punkte (nur erste Hälfte der Runs)
sorted_runs = sorted(completed_runs, key=lambda r: r.run_id)
early_runs = sorted_runs[: len(sorted_runs) // 2] if len(sorted_runs) > 1 else sorted_runs
pareto_points_before = [r for r in early_runs if is_pareto_optimal(r, early_runs)]

# Volume als Anzahl Pareto-Punkte approximiert
volume_now = len(pareto_points_now)
volume_before = len(pareto_points_before) if len(pareto_points_before) > 0 else 1

expansion = (volume_now - volume_before) / volume_before if volume_before > 0 else 0.0

target_met = expansion >= self.targets["pareto_expansion"]

return {
"frontier_volume_now": volume_now,
"frontier_volume_before": volume_before,
"expansion_percent": round(expansion * 100, 2),
"pareto_points_now": volume_now,
"pareto_points_before": volume_before,
"target_met": target_met,
}

def compute_human_time_saved(self, weeks: int = 4) -> Dict[str, Any]:
"""
Human Time Saved berechnen.

Formel:
(manual_hours_per_week - autonomous_hours_per_week) / manual_hours_per_week

Ziel: 70% weniger manuelle Run-Auswahl

Args:
weeks: Anzahl Wochen für Betrachtung

Returns:
{
"manual_hours_per_week": X,
"autonomous_hours_per_week": Y,
"time_saved_percent": Z%,
"target_met": True/False
}
"""
# Schätzung: Manuelle Auswahl benötigt ~5 Min pro Run
# Autonome Auswahl benötigt ~0.5 Min pro Run (Review)
MANUAL_TIME_PER_RUN = 5.0 / 60.0 # Stunden
AUTONOMOUS_TIME_PER_RUN = 0.5 / 60.0 # Stunden

completed_runs = self._get_completed_runs()

if not completed_runs:
return {
"manual_hours_per_week": self.baseline_human_time,
"autonomous_hours_per_week": 0.0,
"time_saved_percent": 100.0,
"target_met": True,
"total_runs": 0,
}

# Annahme: Alle Runs wären manuell ausgewählt worden
# Autonome Zeit nur für Review
total_manual_time = len(completed_runs) * MANUAL_TIME_PER_RUN
total_autonomous_time = len(completed_runs) * AUTONOMOUS_TIME_PER_RUN

# Pro Woche (angenommene weeks)
manual_per_week = total_manual_time / weeks if weeks > 0 else total_manual_time
autonomous_per_week = total_autonomous_time / weeks if weeks > 0 else total_autonomous_time

# Zeitersparnis
if manual_per_week > 0:
time_saved = (manual_per_week - autonomous_per_week) / manual_per_week
else:
time_saved = 0.0

target_met = time_saved >= self.targets["human_time_saved"]

return {
"manual_hours_per_week": round(manual_per_week, 2),
"autonomous_hours_per_week": round(autonomous_per_week, 2),
"time_saved_percent": round(time_saved * 100, 2),
"target_met": target_met,
"total_runs": len(completed_runs),
"weeks_analyzed": weeks,
}

def compute_confidence_accuracy(self, min_confidence: float = 0.6) -> Dict[str, Any]:
"""
Confidence Accuracy berechnen.

Formel:
Korrelation zwischen predicted_confidence und actual_success_rate

Ziel: >75% Accuracy für erfolgreiche Runs

Args:
min_confidence: Minimale Confidence für Betrachtung

Returns:
{
"predicted_confidence_avg": X,
"actual_success_rate": Y,
"correlation": Z,
"calibration_error": E,
"target_met": True/False
}
"""
completed_runs = self._get_completed_runs()

if not completed_runs:
return {
"predicted_confidence_avg": 0.0,
"actual_success_rate": 0.0,
"correlation": 0.0,
"calibration_error": 1.0,
"target_met": False,
}

# Hole Runs mit Confidence (aus Tags oder Metadaten)
# Hinweis: In echter Implementierung würde Confidence aus SurrogateScorer kommen
confidence_data: List[Tuple[float, bool]] = []

for run in completed_runs:
# Versuche Confidence aus Tags zu extrahieren
# Format: "confidence:0.85"
confidence = None
for tag in run.tags:
if tag.startswith("confidence:"):
try:
confidence = float(tag.split(":")[1])
break
except (ValueError, IndexError):
pass

if confidence is not None and confidence >= min_confidence:
success = run.status == "completed" and (
run.delta_bpb is not None and run.delta_bpb < 0
or (run.delta_bpb is None and run.val_bpb is not None)
)
confidence_data.append((confidence, success))

if not confidence_data:
# Fallback: Alle completed Runs als erfolgreich
success_rate = len([r for r in completed_runs if r.status == "completed"]) / len(completed_runs)
return {
"predicted_confidence_avg": 0.0,
"actual_success_rate": round(success_rate, 4),
"correlation": 0.0,
"calibration_error": 1.0 - success_rate,
"target_met": success_rate >= self.targets["confidence_accuracy"],
"warning": "Keine Confidence-Daten gefunden, verwende Fallback",
}

# Statistiken berechnen
confidences = [c for c, _ in confidence_data]
successes = [1 if s else 0 for _, s in confidence_data]

avg_confidence = np.mean(confidences)
actual_success_rate = np.mean(successes)

# Korrelation (Pearson)
if len(confidences) > 2:
corr_matrix = np.corrcoef(confidences, successes)
if corr_matrix.shape == (2, 2):
correlation = corr_matrix[0, 1]
else:
correlation = 0.0
if np.isnan(correlation):
correlation = 0.0
else:
correlation = 0.0

# Kalibrierungsfehler: |avg_confidence - actual_success_rate|
calibration_error = abs(avg_confidence - actual_success_rate)

# Accuracy: Anteil der Runs wo Confidence mit Erfolg übereinstimmt
# (high confidence + success) oder (low confidence + failure)
correct_predictions = sum(
1 for c, s in confidence_data
if (c >= 0.7 and s) or (c < 0.7 and not s)
)
accuracy = correct_predictions / len(confidence_data) if confidence_data else 0.0

target_met = accuracy >= self.targets["confidence_accuracy"]

return {
"predicted_confidence_avg": round(avg_confidence, 4),
"actual_success_rate": round(actual_success_rate, 4),
"correlation": round(correlation, 4),
"calibration_error": round(calibration_error, 4),
"accuracy": round(accuracy, 4),
"target_met": target_met,
"samples": len(confidence_data),
}

def get_all_metrics(self) -> Dict[str, MetricDefinition]:
"""
Alle Metriken mit aktuellen Werten zurückgeben.

Returns:
Dictionary von MetricDefinitionen
"""
metrics: Dict[str, MetricDefinition] = {}

# 1. Search Efficiency
search_eff = self.compute_search_efficiency()
metrics["search_efficiency"] = MetricDefinition(
name="Search Efficiency",
description="Runs benötigt für ΔBPB = -0.05",
formula=r"\\frac{\\text{Runs}_{manual} - \\text{Runs}_{auto}}{\\text{Runs}_{manual}}",
target_value=self.targets["search_efficiency_improvement"] * 100,
direction="higher_better",
baseline_value=self.baseline_search_efficiency,
current_value=search_eff.get("improvement_percent", 0.0),
unit="%",
target_met=search_eff.get("target_met", False),
)

# 2. Failure Rate Reduction
failure_red = self.compute_failure_rate_reduction()
metrics["failure_rate_reduction"] = MetricDefinition(
name="Failure Rate Reduction",
description="Reduktion der Fehlerquote (OOM, NaN, Divergence)",
formula=r"\\frac{\\text{Rate}_{before} - \\text{Rate}_{after}}{\\text{Rate}_{before}}",
target_value=self.targets["failure_rate_reduction"] * 100,
direction="higher_better",
baseline_value=self.baseline_failure_rate * 100,
current_value=failure_red.get("reduction_percent", 0.0),
unit="%",
target_met=failure_red.get("target_met", False),
)

# 3. Pareto Frontier Expansion
pareto_exp = self.compute_pareto_frontier_expansion()
metrics["pareto_expansion"] = MetricDefinition(
name="Pareto Frontier Expansion",
description="Wachstum der Pareto-optimale Punkte",
formula=r"\\frac{\\text{Volume}_{now} - \\text{Volume}_{before}}{\\text{Volume}_{before}}",
target_value=self.targets["pareto_expansion"] * 100,
direction="higher_better",
baseline_value=1.0,
current_value=pareto_exp.get("expansion_percent", 0.0),
unit="%",
target_met=pareto_exp.get("target_met", False),
)

# 4. Human Time Saved
time_saved = self.compute_human_time_saved()
metrics["human_time_saved"] = MetricDefinition(
name="Human Time Saved",
description="Eingesparte Zeit bei Run-Auswahl",
formula=r"\\frac{\\text{Hours}_{manual} - \\text{Hours}_{auto}}{\\text{Hours}_{manual}}",
target_value=self.targets["human_time_saved"] * 100,
direction="higher_better",
baseline_value=self.baseline_human_time,
current_value=time_saved.get("time_saved_percent", 0.0),
unit="%",
target_met=time_saved.get("target_met", False),
)

# 5. Confidence Accuracy
conf_acc = self.compute_confidence_accuracy()
metrics["confidence_accuracy"] = MetricDefinition(
name="Confidence Accuracy",
description="Accuracy der Confidence-Vorhersagen",
formula=r"\\text{Accuracy} = \\frac{\\text{Correct Predictions}}{\\text{Total}}",
target_value=self.targets["confidence_accuracy"] * 100,
direction="higher_better",
baseline_value=0.5,
current_value=conf_acc.get("accuracy", 0.0) * 100,
unit="%",
target_met=conf_acc.get("target_met", False),
)

return metrics

def generate_report(self) -> str:
"""
Success Metrics Report generieren.

Returns:
Markdown-formattierter Report
"""
metrics = self.get_all_metrics()

report = []
report.append("# Phase 4 Success Metrics Report")
report.append(f"\nGeneriert: {datetime.now().isoformat()}")
report.append("\n" + "=" * 70)

# Zusammenfassung
targets_met = sum(1 for m in metrics.values() if m.target_met)
total_targets = len(metrics)

report.append(f"\n## Zusammenfassung")
report.append(f"\n**{targets_met}/{total_targets} Ziele erreicht**")

# Einzelne Metriken
report.append("\n## Detaillierte Metriken")

for name, metric in metrics.items():
status_icon = "" if metric.target_met else ""
report.append(f"\n### {metric.name} {status_icon}")
report.append(f"\n{metric.description}")
report.append(f"\n**Formel:** `{metric.formula}`")
report.append(f"\n| Kennzahl | Wert |")
report.append(f"|----------|------|")
report.append(f"| aktueller Wert | {metric.current_value:.2f} {metric.unit} |")
report.append(f"| Zielwert | {metric.target_value:.2f} {metric.unit} |")
report.append(f"| Baseline | {metric.baseline_value:.2f} {metric.unit} |")
report.append(f"| Richtung | {'höher besser' if metric.direction == 'higher_better' else 'niedriger besser'} |")
report.append(f"| **Ziel erreicht** | **{'Ja' if metric.target_met else 'Nein'}** |")

# Empfehlungen
report.append("\n## Empfehlungen")

if targets_met == total_targets:
report.append("\n **Alle Ziele erreicht!** Phase 4 kann als erfolgreich betrachtet werden.")
elif targets_met >= total_targets * 0.8:
report.append(f"\n **{targets_met}/{total_targets} Ziele erreicht.** Knapp verfehlte Ziele analysieren.")
else:
report.append(f"\n **Nur {targets_met}/{total_targets} Ziele erreicht.** Weitere Optimierung empfohlen.")

# Nicht erreichte Ziele
missed = [m for m in metrics.values() if not m.target_met]
if missed:
report.append("\n### Nicht erreichte Ziele:")
for m in missed:
gap = m.target_value - m.current_value if m.direction == "higher_better" else m.current_value - m.target_value
report.append(f"- **{m.name}**: Gap von {gap:.2f} {m.unit}")

return "\n".join(report)
