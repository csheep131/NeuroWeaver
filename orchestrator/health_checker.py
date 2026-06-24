#!/usr/bin/env python3
"""
Training Health Checker für NeuroWeave Phase 5.

Automatische Gesundheitsprüfung für Training Runs.

Checks:
1. Loss Divergence (loss > 10x baseline)
2. Gradient Explosion (norm > 1000)
3. VRAM Leak (usage steigt kontinuierlich)
4. Step-Time Anomaly (>50% langsamer)
5. NaN Detection (in weights/gradients)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
import numpy as np
NUMPY_AVAILABLE = True
except ImportError:
NUMPY_AVAILABLE = False
print("Warnung: numpy nicht installiert. Installiere mit: pip install numpy")


class HealthStatus(Enum):
"""Gesundheitsstatus."""

HEALTHY = "healthy"
WARNING = "warning"
CRITICAL = "critical"


class IssueType(Enum):
"""Typ des Gesundheitsproblems."""

LOSS_DIVERGENCE = "loss_divergence"
GRADIENT_EXPLOSION = "gradient_explosion"
VRAM_LEAK = "vram_leak"
STEP_TIME_ANOMALY = "step_time_anomaly"
NaN_DETECTION = "nan_detection"
LOSS_OSCILLATION = "loss_oscillation"
GRADIENT_VANISHING = "gradient_vanishing"
CONVERGENCE_ISSUE = "convergence_issue"


@dataclass
class HealthIssue:
"""Ein einzelnes Gesundheitsproblem."""

issue_type: IssueType
severity: Literal["info", "warning", "critical"]
message: str
value: float
threshold: float
recommendation: str

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"issue_type": self.issue_type.value,
"severity": self.severity,
"message": self.message,
"value": self.value,
"threshold": self.threshold,
"recommendation": self.recommendation,
}


@dataclass
class HealthReport:
"""Gesundheitsbericht für Run."""

run_id: str
health_score: float # 0-100
status: HealthStatus
issues: List[HealthIssue] = field(default_factory=list)
recommendations: List[str] = field(default_factory=list)
timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
metrics_analyzed: Dict[str, Any] = field(default_factory=dict)

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"run_id": self.run_id,
"health_score": self.health_score,
"status": self.status.value,
"issues": [i.to_dict() for i in self.issues],
"recommendations": self.recommendations,
"timestamp": self.timestamp,
"metrics_analyzed": self.metrics_analyzed,
}

def is_healthy(self) -> bool:
"""Prüfe ob Run gesund ist."""
return self.status == HealthStatus.HEALTHY

def has_critical_issues(self) -> bool:
"""Prüfe ob kritische Probleme vorhanden sind."""
return any(issue.severity == "critical" for issue in self.issues)

def has_warning_issues(self) -> bool:
"""Prüfe ob Warnungen vorhanden sind."""
return any(issue.severity == "warning" for issue in self.issues)


class TrainingHealthChecker:
"""
Automatische Gesundheitsprüfung für Training Runs.

Checks:
1. Loss Divergence (loss > 10x baseline)
2. Gradient Explosion (norm > 1000)
3. VRAM Leak (usage steigt kontinuierlich)
4. Step-Time Anomaly (>50% langsamer)
5. NaN Detection (in weights/gradients)

Example:
checker = TrainingHealthChecker()

# Health Check durchführen
report = checker.check_health("run001", metrics={
"loss_history": [1.5, 1.4, 1.3, 10.5],
"gradient_norm_history": [50, 60, 55, 1500],
"vram_history": [6000, 6100, 6200, 6300],
"step_time_history": [100, 105, 110, 200],
})

print(f"Health Score: {report.health_score}")
print(f"Status: {report.status.value}")

# Frühwarnzeichen erkennen
warnings = checker.get_early_warning_signs("run001")
"""

def __init__(
self,
loss_divergence_threshold: float = 10.0,
gradient_explosion_threshold: float = 1000.0,
vram_leak_threshold_mb_per_step: float = 50.0,
step_time_anomaly_threshold: float = 1.5,
nan_check_enabled: bool = True,
window_size: int = 20,
) -> None:
"""
Initialisiere Training Health Checker.

Args:
loss_divergence_threshold: Faktor für Loss Divergence
gradient_explosion_threshold: Absoluter Threshold für Gradient Norm
vram_leak_threshold_mb_per_step: MB Anstieg pro Step für VRAM Leak
step_time_anomaly_threshold: Faktor für Step-Time Anomalie
nan_check_enabled: NaN-Check aktivieren
window_size: Fenstergröße für rolling statistics
"""
self._loss_divergence_threshold = loss_divergence_threshold
self._gradient_explosion_threshold = gradient_explosion_threshold
self._vram_leak_threshold = vram_leak_threshold_mb_per_step
self._step_time_anomaly_threshold = step_time_anomaly_threshold
self._nan_check_enabled = nan_check_enabled
self._window_size = window_size

# Thresholds für Frühwarnzeichen
self._early_warning_thresholds = {
"loss_oscillation": 0.3, # 30% Oszillation
"gradient_increase_rate": 0.1, # 10% pro Step
"step_time_variance": 0.2, # 20% Varianz
}

def check_health(
self,
run_id: str,
metrics: Dict[str, Any],
) -> HealthReport:
"""
Gesundheitscheck durchführen.

Args:
run_id: Run-ID
metrics: Dictionary mit Metriken:
- loss_history: Liste von Loss-Werten
- gradient_norm_history: Liste von Gradient Norms
- vram_history: Liste von VRAM-Werten (MB)
- step_time_history: Liste von Step-Zeiten (ms)
- current_loss: Aktueller Loss (optional)
- baseline_loss: Baseline Loss (optional)

Returns:
HealthReport mit Score und Empfehlungen
"""
issues: List[HealthIssue] = []
recommendations: List[str] = []

# Metriken extrahieren
loss_history = metrics.get("loss_history", [])
gradient_history = metrics.get("gradient_norm_history", [])
vram_history = metrics.get("vram_history", [])
step_time_history = metrics.get("step_time_history", [])
current_loss = metrics.get("current_loss")
baseline_loss = metrics.get("baseline_loss")

# 1. Loss Divergence Check
loss_issue = self._check_loss_divergence(
loss_history, current_loss, baseline_loss
)
if loss_issue:
issues.append(loss_issue)
recommendations.append(loss_issue.recommendation)

# 2. Gradient Explosion Check
gradient_issue = self._check_gradient_explosion(gradient_history)
if gradient_issue:
issues.append(gradient_issue)
recommendations.append(gradient_issue.recommendation)

# 3. VRAM Leak Check
vram_issue = self._check_vram_leak(vram_history)
if vram_issue:
issues.append(vram_issue)
recommendations.append(vram_issue.recommendation)

# 4. Step-Time Anomaly Check
step_time_issue = self._check_step_time_anomaly(step_time_history)
if step_time_issue:
issues.append(step_time_issue)
recommendations.append(step_time_issue.recommendation)

# 5. NaN Detection (wenn enabled)
if self._nan_check_enabled:
nan_issue = self._check_nan_detection(metrics)
if nan_issue:
issues.append(nan_issue)
recommendations.append(nan_issue.recommendation)

# 6. Additional Checks (Oszillation, Vanishing, etc.)
oscillation_issue = self._check_loss_oscillation(loss_history)
if oscillation_issue:
issues.append(oscillation_issue)
recommendations.append(oscillation_issue.recommendation)

gradient_vanishing_issue = self._check_gradient_vanishing(gradient_history)
if gradient_vanishing_issue:
issues.append(gradient_vanishing_issue)
recommendations.append(gradient_vanishing_issue.recommendation)

# Health Score berechnen (0-100)
health_score = self._calculate_health_score(issues)

# Status bestimmen
status = self._determine_status(issues, health_score)

# Allgemeine Empfehlungen hinzufügen
if not recommendations:
recommendations.append("Training läuft gesund. Weiterhin überwachen.")

return HealthReport(
run_id=run_id,
health_score=health_score,
status=status,
issues=issues,
recommendations=recommendations,
metrics_analyzed={
"loss_points": len(loss_history),
"gradient_points": len(gradient_history),
"vram_points": len(vram_history),
"step_time_points": len(step_time_history),
},
)

def _check_loss_divergence(
self,
loss_history: List[float],
current_loss: Optional[float] = None,
baseline_loss: Optional[float] = None,
) -> Optional[HealthIssue]:
"""Prüfe auf Loss Divergence."""
if not loss_history:
return None

# Aktuelle Loss verwenden oder letzte bekannte
loss = current_loss if current_loss is not None else loss_history[-1]

# Baseline bestimmen
if baseline_loss is not None:
baseline = baseline_loss
elif len(loss_history) >= 5:
# Durchschnitt der ersten 5 Steps als Baseline
baseline = sum(loss_history[:5]) / 5
else:
baseline = loss_history[0]

if baseline <= 0:
return None

# Prüfe ob Loss > threshold * baseline
ratio = loss / baseline
if ratio > self._loss_divergence_threshold:
return HealthIssue(
issue_type=IssueType.LOSS_DIVERGENCE,
severity="critical",
message=f"Loss Divergence erkannt: {loss:.4f} ist {ratio:.1f}x höher als Baseline {baseline:.4f}",
value=ratio,
threshold=self._loss_divergence_threshold,
recommendation="Training sofort stoppen. Learning Rate reduzieren oder Gradient Clipping erhöhen.",
)

# Warnung bei moderatem Anstieg
if ratio > self._loss_divergence_threshold / 2:
return HealthIssue(
issue_type=IssueType.LOSS_DIVERGENCE,
severity="warning",
message=f"Loss Anstieg erkannt: {loss:.4f} ist {ratio:.1f}x höher als Baseline {baseline:.4f}",
value=ratio,
threshold=self._loss_divergence_threshold / 2,
recommendation="Training überwachen. Bei weiterem Anstieg Learning Rate reduzieren.",
)

return None

def _check_gradient_explosion(
self,
gradient_history: List[float],
) -> Optional[HealthIssue]:
"""Prüfe auf Gradient Explosion."""
if not gradient_history:
return None

latest_gradient = gradient_history[-1]

if latest_gradient > self._gradient_explosion_threshold:
return HealthIssue(
issue_type=IssueType.GRADIENT_EXPLOSION,
severity="critical",
message=f"Gradient Explosion: Norm {latest_gradient:.1f} überschreitet Threshold {self._gradient_explosion_threshold}",
value=latest_gradient,
threshold=self._gradient_explosion_threshold,
recommendation="Gradient Clipping aktivieren/reduzieren. Learning Rate stark reduzieren.",
)

# Trend-Analyse
if len(gradient_history) >= 10:
recent_avg = sum(gradient_history[-10:]) / 10
# earlier_avg nur berechnen wenn genug Daten vorhanden
if len(gradient_history) >= 20:
earlier_avg = sum(gradient_history[-20:-10]) / 10
else:
earlier_avg = sum(gradient_history[:10]) / 10 if len(gradient_history) >= 10 else recent_avg

if earlier_avg > 0 and recent_avg > earlier_avg * 2:
return HealthIssue(
issue_type=IssueType.GRADIENT_EXPLOSION,
severity="warning",
message=f"Gradient Norm steigt kontinuierlich: {earlier_avg:.1f} → {recent_avg:.1f}",
value=recent_avg / earlier_avg,
threshold=2.0,
recommendation="Gradient Clipping in Betracht ziehen. Training überwachen.",
)

return None

def _check_vram_leak(
self,
vram_history: List[float],
) -> Optional[HealthIssue]:
"""Prüfe auf VRAM Leak."""
if len(vram_history) < 5:
return None

# Linearer Fit für Trend
n = len(vram_history)
x_mean = (n - 1) / 2
y_mean = sum(vram_history) / n

numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vram_history))
denominator = sum((i - x_mean) ** 2 for i in range(n))

if denominator == 0:
return None

slope = numerator / denominator # MB pro Step

if slope > self._vram_leak_threshold:
total_increase = vram_history[-1] - vram_history[0]
return HealthIssue(
issue_type=IssueType.VRAM_LEAK,
severity="critical",
message=f"VRAM Leak erkannt: +{slope:.1f} MB/Step, insgesamt +{total_increase:.0f} MB",
value=slope,
threshold=self._vram_leak_threshold,
recommendation="Auf Memory Leaks prüfen. Batch Size reduzieren. Checkpointing prüfen.",
)

# Warnung bei moderatem Anstieg
if slope > self._vram_leak_threshold / 2:
return HealthIssue(
issue_type=IssueType.VRAM_LEAK,
severity="warning",
message=f"VRAM steigt moderat: +{slope:.1f} MB/Step",
value=slope,
threshold=self._vram_leak_threshold / 2,
recommendation="VRAM-Entwicklung überwachen. Bei weiterem Anstieg eingreifen.",
)

return None

def _check_step_time_anomaly(
self,
step_time_history: List[float],
) -> Optional[HealthIssue]:
"""Prüfe auf Step-Time Anomalie."""
if len(step_time_history) < 5:
return None

latest_time = step_time_history[-1]
recent_avg = sum(step_time_history[-10:-1]) / min(9, len(step_time_history) - 1)

if recent_avg <= 0:
return None

ratio = latest_time / recent_avg

if ratio > self._step_time_anomaly_threshold:
return HealthIssue(
issue_type=IssueType.STEP_TIME_ANOMALY,
severity="warning",
message=f"Step-Time Anomalie: {latest_time:.1f}ms ist {ratio:.1f}x langsamer als Durchschnitt {recent_avg:.1f}ms",
value=ratio,
threshold=self._step_time_anomaly_threshold,
recommendation="System-Last prüfen. I/O Bottlenecks suchen. Data Loading optimieren.",
)

return None

def _check_nan_detection(
self,
metrics: Dict[str, Any],
) -> Optional[HealthIssue]:
"""Prüfe auf NaN in Metriken."""
if not NUMPY_AVAILABLE:
return None

# Verschiedene Metriken auf NaN prüfen
for key, value in metrics.items():
if isinstance(value, (list, np.ndarray)):
if NUMPY_AVAILABLE and np.any(np.isnan(value)):
return HealthIssue(
issue_type=IssueType.NaN_DETECTION,
severity="critical",
message=f"NaN erkannt in {key}",
value=float("nan"),
threshold=0.0,
recommendation="Training sofort stoppen. Numerical Stability prüfen. Learning Rate stark reduzieren.",
)
elif isinstance(value, float) and np.isnan(value):
return HealthIssue(
issue_type=IssueType.NaN_DETECTION,
severity="critical",
message=f"NaN erkannt in {key}: {value}",
value=float("nan"),
threshold=0.0,
recommendation="Training sofort stoppen. Numerical Stability prüfen.",
)

return None

def _check_loss_oscillation(
self,
loss_history: List[float],
) -> Optional[HealthIssue]:
"""Prüfe auf Loss Oszillation."""
if len(loss_history) < 10:
return None

# Berechne Oszillation (Differenz zwischen aufeinanderfolgenden Werten)
diffs = [abs(loss_history[i] - loss_history[i-1]) for i in range(1, len(loss_history))]

if not diffs:
return None

avg_loss = sum(loss_history[-10:]) / 10
if avg_loss <= 0:
return None

avg_diff = sum(diffs[-10:]) / 10
oscillation_ratio = avg_diff / avg_loss

if oscillation_ratio > self._early_warning_thresholds["loss_oscillation"]:
return HealthIssue(
issue_type=IssueType.LOSS_OSCILLATION,
severity="warning",
message=f"Loss oszilliert stark: Ø{avg_diff:.4f} bei ØLoss {avg_loss:.4f} ({oscillation_ratio:.1%})",
value=oscillation_ratio,
threshold=self._early_warning_thresholds["loss_oscillation"],
recommendation="Learning Rate reduzieren. Batch Size erhöhen. Gradient Accumulation erwägen.",
)

return None

def _check_gradient_vanishing(
self,
gradient_history: List[float],
) -> Optional[HealthIssue]:
"""Prüfe auf Gradient Vanishing."""
if len(gradient_history) < 10:
return None

latest = gradient_history[-1]
earlier_avg = sum(gradient_history[:10]) / 10 if len(gradient_history) >= 10 else gradient_history[0]

if earlier_avg <= 0:
return None

ratio = latest / earlier_avg

# Gradient Vanishing wenn < 1% der ursprünglichen Größe
if ratio < 0.01 and latest < 1e-6:
return HealthIssue(
issue_type=IssueType.GRADIENT_VANISHING,
severity="warning",
message=f"Gradient Vanishing: {latest:.2e} ist nur {ratio:.2%} der ursprünglichen Norm",
value=ratio,
threshold=0.01,
recommendation="Activation Functions prüfen (ReLU statt Tanh/Sigmoid). Residual Connections verwenden.",
)

return None

def _calculate_health_score(self, issues: List[HealthIssue]) -> float:
"""
Berechne Health Score (0-100).

Args:
issues: Liste von Gesundheitsproblemen

Returns:
Health Score von 0 (kritisch) bis 100 (gesund)
"""
if not issues:
return 100.0

# Strafpunkte berechnen
penalty = 0.0
for issue in issues:
if issue.severity == "critical":
penalty += 40.0
elif issue.severity == "warning":
penalty += 15.0
else: # info
penalty += 5.0

# Score berechnen (mindestens 0)
score = max(0.0, 100.0 - penalty)
return round(score, 1)

def _determine_status(
self,
issues: List[HealthIssue],
health_score: float,
) -> HealthStatus:
"""
Bestimme Gesundheitsstatus.

Args:
issues: Liste von Gesundheitsproblemen
health_score: Berechneter Health Score

Returns:
HealthStatus (healthy, warning, critical)
"""
# Critical wenn kritische Issues vorhanden oder Score < 30
if any(issue.severity == "critical" for issue in issues) or health_score < 30:
return HealthStatus.CRITICAL

# Warning wenn Warnungen vorhanden oder Score < 70
if any(issue.severity == "warning" for issue in issues) or health_score < 70:
return HealthStatus.WARNING

# Healthy
return HealthStatus.HEALTHY

def get_early_warning_signs(self, run_id: str, metrics: Dict[str, Any]) -> List[str]:
"""
Frühwarnzeichen erkennen.

Beispiele:
- Loss oszilliert stärker als Parent
- Gradient norm steigt über Zeit
- Step-Time wird inkonsistent

Args:
run_id: Run-ID
metrics: Dictionary mit Metriken

Returns:
Liste von Warnzeichen als Strings
"""
warnings: List[str] = []

loss_history = metrics.get("loss_history", [])
gradient_history = metrics.get("gradient_norm_history", [])
step_time_history = metrics.get("step_time_history", [])

# 1. Loss Oszillation Trend
if len(loss_history) >= 20:
recent_oscillation = self._calculate_oscillation(loss_history[-10:])
earlier_oscillation = self._calculate_oscillation(loss_history[:10])

if recent_oscillation > earlier_oscillation * 1.5:
warnings.append(
f" Loss-Oszillation nimmt zu: {earlier_oscillation:.4f} → {recent_oscillation:.4f}"
)

# 2. Gradient Norm Trend
if len(gradient_history) >= 20:
recent_avg = sum(gradient_history[-10:]) / 10
earlier_avg = sum(gradient_history[:10]) / 10

if earlier_avg > 0 and recent_avg > earlier_avg * 1.3:
warnings.append(
f" Gradient Norm steigt: {earlier_avg:.1f} → {recent_avg:.1f} (+{(recent_avg/earlier_avg-1)*100:.0f}%)"
)

# 3. Step-Time Inkonsistenz
if len(step_time_history) >= 20:
recent_variance = np.var(step_time_history[-10:]) if NUMPY_AVAILABLE else self._calc_variance(step_time_history[-10:])
earlier_variance = np.var(step_time_history[:10]) if NUMPY_AVAILABLE else self._calc_variance(step_time_history[:10])

if earlier_variance > 0 and recent_variance > earlier_variance * 2:
warnings.append(
f" Step-Time wird inkonsistent: Varianz {earlier_variance:.1f} → {recent_variance:.1f}"
)

# 4. VRAM kontinuierlicher Anstieg
vram_history = metrics.get("vram_history", [])
if len(vram_history) >= 20:
n = len(vram_history)
x_mean = (n - 1) / 2
y_mean = sum(vram_history) / n
numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vram_history))
denominator = sum((i - x_mean) ** 2 for i in range(n))

if denominator > 0:
slope = numerator / denominator
if slope > 10: # >10 MB/Step
warnings.append(
f" VRAM steigt kontinuierlich: +{slope:.1f} MB/Step"
)

return warnings

def _calculate_oscillation(self, values: List[float]) -> float:
"""Berechne Oszillation (durchschnittliche absolute Differenz)."""
if len(values) < 2:
return 0.0

diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
return sum(diffs) / len(diffs)

def _calc_variance(self, values: List[float]) -> float:
"""Berechne Varianz (Fallback ohne numpy)."""
if len(values) < 2:
return 0.0

mean = sum(values) / len(values)
return sum((x - mean) ** 2 for x in values) / len(values)


def cmd_health_check(args: argparse.Namespace) -> int:
"""Health Check Command."""
print(" Training Health Checker")
print("=" * 60)

checker = TrainingHealthChecker()

# Beispiel-Metriken für Demo
demo_metrics = {
"loss_history": [1.5, 1.4, 1.35, 1.3, 1.28, 1.25, 1.23, 1.22, 1.20, 1.19],
"gradient_norm_history": [50, 55, 52, 48, 51, 49, 53, 50, 48, 52],
"vram_history": [6000, 6020, 6040, 6060, 6080, 6100, 6120, 6140, 6160, 6180],
"step_time_history": [100, 102, 98, 105, 101, 99, 103, 100, 102, 101],
}

# Health Check durchführen
report = checker.check_health(args.run_id, demo_metrics)

# Report ausgeben
status_icon = {
HealthStatus.HEALTHY: "",
HealthStatus.WARNING: "",
HealthStatus.CRITICAL: "",
}.get(report.status, "")

print(f"\n{status_icon} Run: {report.run_id}")
print(f" Health Score: {report.health_score}/100")
print(f" Status: {report.status.value.upper()}")

if report.issues:
print(f"\n Gefundene Probleme ({len(report.issues)}):")
for issue in report.issues:
severity_icon = {
"critical": "",
"warning": "🟡",
"info": "",
}.get(issue.severity, "")
print(f" {severity_icon} [{issue.issue_type.value}]")
print(f" {issue.message}")
print(f" → {issue.recommendation}")

print(f"\n Empfehlungen:")
for rec in report.recommendations:
print(f" • {rec}")

# Frühwarnzeichen
warnings = checker.get_early_warning_signs(args.run_id, demo_metrics)
if warnings:
print(f"\n Frühwarnzeichen:")
for warning in warnings:
print(f" {warning}")

print("\n" + "=" * 60)

# Exit Code basierend auf Status
if report.status == HealthStatus.CRITICAL:
return 2
elif report.status == HealthStatus.WARNING:
return 1
return 0


def create_parser() -> argparse.ArgumentParser:
"""Erstelle Argument Parser."""
parser = argparse.ArgumentParser(
prog="health-check",
description="Training Health Checker",
)
parser.add_argument(
"run_id",
type=str,
default="demo_run",
nargs="?",
help="Run-ID für Health Check",
)
parser.set_defaults(func=cmd_health_check)
return parser


def main() -> int:
"""Hauptfunktion."""
parser = create_parser()
args = parser.parse_args()
return args.func(args)


if __name__ == "__main__":
sys.exit(main())
