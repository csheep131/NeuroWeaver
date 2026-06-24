#!/usr/bin/env python3
"""
A/B-Testing Framework für NeuroWeave Phase 4.

Vergleich von Autonomie-Leveln:
- Autonomous vs Manual: Voll-auto vs. menschliche Auswahl
- Assisted vs Manual: Vorschläge mit Review vs. manuell
- High-Confidence vs Low-Confidence: Confidence-Threshold-Test
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy import stats

from core.registry import RunRegistry, RunEntry


@dataclass
class ABTestConfig:
"""Konfiguration für A/B-Test."""

test_name: str
start_date: datetime
end_date: datetime
treatment_group: Literal["autonomous", "assisted", "manual"]
control_group: Literal["autonomous", "assisted", "manual"]
success_metrics: List[str] # ["bpb_gain", "efficiency", "human_time"]
min_sample_size: int = 30 # Minimale Runs pro Gruppe

def __post_init__(self) -> None:
"""Validiere Konfiguration."""
if self.treatment_group == self.control_group:
raise ValueError(
f"treatment_group und control_group müssen unterschiedlich sein. "
f"Beide sind: {self.treatment_group}"
)

if not self.success_metrics:
raise ValueError("success_metrics darf nicht leer sein")

if self.min_sample_size < 5:
raise ValueError(f"min_sample_size muss mindestens 5 sein, ist: {self.min_sample_size}")


@dataclass
class ABTestResult:
"""Ergebnis eines A/B-Tests für eine einzelne Metrik."""

test_name: str
metric: str
treatment_mean: float
control_mean: float
treatment_std: float
control_std: float
t_statistic: float
p_value: float
confidence_interval: Tuple[float, float]
effect_size: float # Cohen's d
is_significant: bool
significance_level: float = 0.05

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"test_name": self.test_name,
"metric": self.metric,
"treatment_mean": round(self.treatment_mean, 4),
"control_mean": round(self.control_mean, 4),
"treatment_std": round(self.treatment_std, 4),
"control_std": round(self.control_std, 4),
"t_statistic": round(self.t_statistic, 4),
"p_value": round(self.p_value, 6),
"confidence_interval": [round(self.confidence_interval[0], 4), round(self.confidence_interval[1], 4)],
"effect_size": round(self.effect_size, 4),
"is_significant": self.is_significant,
"significance_level": self.significance_level,
}


@dataclass
class ABTestOutcome:
"""Gespeichertes Outcome für einen einzelnen Run."""

run_id: str
group: Literal["treatment", "control"]
metrics: Dict[str, float]
timestamp: datetime = field(default_factory=datetime.now)

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"run_id": self.run_id,
"group": self.group,
"metrics": self.metrics,
"timestamp": self.timestamp.isoformat(),
}

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> ABTestOutcome:
"""Erstelle aus Dictionary."""
return cls(
run_id=data["run_id"],
group=data["group"],
metrics=data["metrics"],
timestamp=datetime.fromisoformat(data["timestamp"]),
)


@dataclass
class ABTestState:
"""Zustand eines A/B-Tests."""

test_id: str
config: ABTestConfig
status: Literal["running", "completed", "paused"] = "running"
treatment_outcomes: List[ABTestOutcome] = field(default_factory=list)
control_outcomes: List[ABTestOutcome] = field(default_factory=list)
created_at: datetime = field(default_factory=datetime.now)
started_at: Optional[datetime] = None
completed_at: Optional[datetime] = None

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"test_id": self.test_id,
"config": {
"test_name": self.config.test_name,
"start_date": self.config.start_date.isoformat(),
"end_date": self.config.end_date.isoformat(),
"treatment_group": self.config.treatment_group,
"control_group": self.config.control_group,
"success_metrics": self.config.success_metrics,
"min_sample_size": self.config.min_sample_size,
},
"status": self.status,
"treatment_outcomes": [o.to_dict() for o in self.treatment_outcomes],
"control_outcomes": [o.to_dict() for o in self.control_outcomes],
"created_at": self.created_at.isoformat(),
"started_at": self.started_at.isoformat() if self.started_at else None,
"completed_at": self.completed_at.isoformat() if self.completed_at else None,
}

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> ABTestState:
"""Erstelle aus Dictionary."""
config = ABTestConfig(
test_name=data["config"]["test_name"],
start_date=datetime.fromisoformat(data["config"]["start_date"]),
end_date=datetime.fromisoformat(data["config"]["end_date"]),
treatment_group=data["config"]["treatment_group"],
control_group=data["config"]["control_group"],
success_metrics=data["config"]["success_metrics"],
min_sample_size=data["config"]["min_sample_size"],
)
return cls(
test_id=data["test_id"],
config=config,
status=data["status"],
treatment_outcomes=[ABTestOutcome.from_dict(o) for o in data["treatment_outcomes"]],
control_outcomes=[ABTestOutcome.from_dict(o) for o in data["control_outcomes"]],
created_at=datetime.fromisoformat(data["created_at"]),
started_at=datetime.fromisoformat(data["started_at"]) if data["started_at"] else None,
completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None,
)


class ABTestFramework:
"""
Framework für A/B-Testing von Autonomie-Leveln.

Test-Szenarien:
1. Autonomous vs Manual: Voll-auto vs. menschliche Auswahl
2. Assisted vs Manual: Vorschläge mit Review vs. manuell
3. High-Confidence vs Low-Confidence: Confidence-Threshold-Test

Example:
framework = ABTestFramework(registry)

# Test erstellen
config = ABTestConfig(
test_name="autonomous_vs_manual",
start_date=datetime.now(),
end_date=datetime.now() + timedelta(days=14),
treatment_group="autonomous",
control_group="manual",
success_metrics=["delta_bpb", "efficiency_gain", "human_time_minutes"],
min_sample_size=30
)
test_id = framework.create_test(config)

# Run zuweisen
group = framework.assign_run_to_group(test_id)
if group == "treatment":
orchestrator.run_autonomous()
else:
human.select_run()

# Outcome
framework.record_outcome(test_id, group, run_id, {
"delta_bpb": -0.02,
"efficiency_gain": 0.15,
"human_time_minutes": 5
})

# Analysieren
results = framework.analyze_test(test_id)
for r in results:
print(f"{r.metric}: p={r.p_value:.4f}, effect={r.effect_size:.2f}")
"""

def __init__(self, registry: RunRegistry, storage_path: str = "results/ab_tests.json") -> None:
"""
Initialisiere A/B-Testing Framework.

Args:
registry: RunRegistry für Datenzugriff
storage_path: Pfad zur Speicherung der Test-Daten
"""
self.registry = registry
self.storage_path = Path(storage_path)
self._tests: Dict[str, ABTestState] = {}
self._load()

def _load(self) -> None:
"""Lade gespeicherte Tests von Disk."""
if self.storage_path.exists():
with open(self.storage_path, "r", encoding="utf-8") as f:
data = json.load(f)
for test_id, test_data in data.items():
self._tests[test_id] = ABTestState.from_dict(test_data)

def _save(self) -> None:
"""Speichere Tests auf Disk."""
self.storage_path.parent.mkdir(parents=True, exist_ok=True)
data = {test_id: test.to_dict() for test_id, test in self._tests.items()}
with open(self.storage_path, "w", encoding="utf-8") as f:
json.dump(data, f, indent=2)

def create_test(self, config: ABTestConfig) -> str:
"""
A/B-Test erstellen.

Args:
config: Test-Konfiguration

Returns:
test_id: Eindeutige Test-ID
"""
test_id = str(uuid.uuid4())[:8]
state = ABTestState(
test_id=test_id,
config=config,
status="running",
started_at=datetime.now(),
)
self._tests[test_id] = state
self._save()
return test_id

def assign_run_to_group(self, test_id: str) -> Literal["treatment", "control"]:
"""
Run zufällig einer Gruppe zuweisen (50/50 Randomisierung).

Args:
test_id: Test-ID

Returns:
"treatment" oder "control"
"""
if test_id not in self._tests:
raise ValueError(f"Test mit ID '{test_id}' existiert nicht")

test = self._tests[test_id]
if test.status != "running":
raise ValueError(f"Test '{test_id}' ist nicht aktiv (Status: {test.status})")

# Einfache 50/50 Randomisierung
return "treatment" if np.random.random() < 0.5 else "control"

def record_outcome(
self,
test_id: str,
group: Literal["treatment", "control"],
run_id: str,
metrics: Dict[str, float],
) -> None:
"""
Outcome für Run.

Args:
test_id: Test-ID
group: "treatment" oder "control"
run_id: Run-ID
metrics: {"delta_bpb": ..., "efficiency_gain": ..., "human_time_minutes": ...}
"""
if test_id not in self._tests:
raise ValueError(f"Test mit ID '{test_id}' existiert nicht")

test = self._tests[test_id]
if test.status != "running":
raise ValueError(f"Test '{test_id}' ist nicht aktiv (Status: {test.status})")

outcome = ABTestOutcome(
run_id=run_id,
group=group,
metrics=metrics.copy(), # Immutable: Kopie erstellen
)

if group == "treatment":
test.treatment_outcomes.append(outcome)
else:
test.control_outcomes.append(outcome)

# Prüfen ob Test abgeschlossen werden kann
self._check_test_completion(test)
self._save()

def _check_test_completion(self, test: ABTestState) -> None:
"""Prüfen ob Test abgeschlossen werden kann."""
min_samples = test.config.min_sample_size
treatment_count = len(test.treatment_outcomes)
control_count = len(test.control_outcomes)

# Test ist completed wenn beide Gruppen genug Samples haben
if treatment_count >= min_samples and control_count >= min_samples:
# Prüfen ob End-Datum erreicht
if datetime.now() >= test.config.end_date:
test.status = "completed"
test.completed_at = datetime.now()

def _compute_cohens_d(
self,
treatment_values: np.ndarray,
control_values: np.ndarray,
) -> float:
"""
Berechne Cohen's d Effektstärke.

Formel:
d = (mean_treatment - mean_control) / pooled_std

wobei:
pooled_std = sqrt(((n1-1)*std1² + (n2-1)*std2²) / (n1+n2-2))
"""
n1 = len(treatment_values)
n2 = len(control_values)

if n1 < 2 or n2 < 2:
return 0.0

mean1 = np.mean(treatment_values)
mean2 = np.mean(control_values)
std1 = np.std(treatment_values, ddof=1)
std2 = np.std(control_values, ddof=1)

# Pooled standard deviation
pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

if pooled_std == 0:
return 0.0

return (mean1 - mean2) / pooled_std

def _compute_confidence_interval(
self,
treatment_values: np.ndarray,
control_values: np.ndarray,
confidence_level: float = 0.95,
) -> Tuple[float, float]:
"""
Berechne Konfidenzintervall für Differenz der Mittelwerte.

Verwendet Welch's t-Intervall für ungleiche Varianzen.
"""
n1 = len(treatment_values)
n2 = len(control_values)

if n1 < 2 or n2 < 2:
return (0.0, 0.0)

mean1 = np.mean(treatment_values)
mean2 = np.mean(control_values)
var1 = np.var(treatment_values, ddof=1)
var2 = np.var(control_values, ddof=1)

# Standardfehler der Differenz
se = np.sqrt(var1 / n1 + var2 / n2)

# Freiheitsgrade (Welch-Satterthwaite Gleichung)
if se == 0:
return (0.0, 0.0)

df = (var1 / n1 + var2 / n2) ** 2 / (
(var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
)

# Kritischer t-Wert
alpha = 1 - confidence_level
t_crit = stats.t.ppf(1 - alpha / 2, df)

# Konfidenzintervall
diff = mean1 - mean2
margin = t_crit * se

return (diff - margin, diff + margin)

def analyze_test(self, test_id: str, significance_level: float = 0.05) -> List[ABTestResult]:
"""
A/B-Test analysieren.

Berechnet für jede Success Metric:
- Mittelwert-Vergleich (t-Test)
- Effektstärke (Cohen's d)
- Konfidenzintervall
- Statistische Signifikanz

Args:
test_id: Test-ID
significance_level: Signifikanzniveau (default: 0.05)

Returns:
Liste der ABTestResult pro Metric
"""
if test_id not in self._tests:
raise ValueError(f"Test mit ID '{test_id}' existiert nicht")

test = self._tests[test_id]
results: List[ABTestResult] = []

# Prüfen ob genug Daten vorhanden
if len(test.treatment_outcomes) < 2 or len(test.control_outcomes) < 2:
return results

# Extrahiere Metrik-Werte
treatment_metrics: Dict[str, List[float]] = defaultdict(list)
control_metrics: Dict[str, List[float]] = defaultdict(list)

for outcome in test.treatment_outcomes:
for metric in test.config.success_metrics:
if metric in outcome.metrics:
treatment_metrics[metric].append(outcome.metrics[metric])

for outcome in test.control_outcomes:
for metric in test.config.success_metrics:
if metric in outcome.metrics:
control_metrics[metric].append(outcome.metrics[metric])

# Analysiere jede Metrik
for metric in test.config.success_metrics:
treatment_vals = treatment_metrics.get(metric, [])
control_vals = control_metrics.get(metric, [])

if len(treatment_vals) < 2 or len(control_vals) < 2:
continue

treatment_arr = np.array(treatment_vals)
control_arr = np.array(control_vals)

# Statistiken
treatment_mean = float(np.mean(treatment_arr))
control_mean = float(np.mean(control_arr))
treatment_std = float(np.std(treatment_arr, ddof=1))
control_std = float(np.std(control_arr, ddof=1))

# Welch's t-Test (für ungleiche Varianzen)
t_stat, p_value = stats.ttest_ind(
treatment_arr,
control_arr,
equal_var=False, # Welch's t-test
)

# Effektstärke
effect_size = self._compute_cohens_d(treatment_arr, control_arr)

# Konfidenzintervall
ci = self._compute_confidence_interval(treatment_arr, control_arr)

# Signifikanz
is_significant = p_value < significance_level

result = ABTestResult(
test_name=test.config.test_name,
metric=metric,
treatment_mean=treatment_mean,
control_mean=control_mean,
treatment_std=treatment_std,
control_std=control_std,
t_statistic=float(t_stat),
p_value=float(p_value),
confidence_interval=ci,
effect_size=effect_size,
is_significant=is_significant,
significance_level=significance_level,
)
results.append(result)

return results

def get_test_summary(self, test_id: str) -> Dict[str, Any]:
"""
Test-Zusammenfassung.

Args:
test_id: Test-ID

Returns:
{
"test_name": "...",
"status": "running" | "completed",
"treatment_runs": N,
"control_runs": M,
"significant_wins": [...],
"recommendation": "..."
}
"""
if test_id not in self._tests:
raise ValueError(f"Test mit ID '{test_id}' existiert nicht")

test = self._tests[test_id]

# Analysiere wenn möglich
results = self.analyze_test(test_id) if len(test.treatment_outcomes) >= 2 and len(test.control_outcomes) >= 2 else []

# Signifikante Wins extrahieren
significant_wins = []
for r in results:
if r.is_significant:
direction = "treatment" if r.treatment_mean > r.control_mean else "control"
significant_wins.append({
"metric": r.metric,
"winner": direction,
"p_value": r.p_value,
"effect_size": r.effect_size,
})

# Empfehlung generieren
recommendation = self._generate_recommendation(test, results)

return {
"test_name": test.config.test_name,
"test_id": test_id,
"status": test.status,
"treatment_runs": len(test.treatment_outcomes),
"control_runs": len(test.control_outcomes),
"treatment_group": test.config.treatment_group,
"control_group": test.config.control_group,
"min_sample_size": test.config.min_sample_size,
"significant_wins": significant_wins,
"recommendation": recommendation,
"created_at": test.created_at.isoformat(),
"started_at": test.started_at.isoformat() if test.started_at else None,
"completed_at": test.completed_at.isoformat() if test.completed_at else None,
}

def _generate_recommendation(self, test: ABTestState, results: List[ABTestResult]) -> str:
"""Generiere Empfehlung basierend auf Test-Ergebnissen."""
if not results:
return "Unzureichende Daten für Empfehlung"

if test.status != "completed":
samples_needed_t = max(0, test.config.min_sample_size - len(test.treatment_outcomes))
samples_needed_c = max(0, test.config.min_sample_size - len(test.control_outcomes))
if samples_needed_t > 0 or samples_needed_c > 0:
return (
f"Test läuft noch. Benötige {samples_needed_t} weitere Treatment- "
f"und {samples_needed_c} Control-Runs für statistische Aussagekraft."
)

# Zähle signifikante Ergebnisse
significant_results = [r for r in results if r.is_significant]

if not significant_results:
return "Keine signifikanten Unterschiede zwischen Treatment und Control"

# Bestimme Gewinner
treatment_wins = sum(1 for r in significant_results if r.treatment_mean > r.control_mean)
control_wins = len(significant_results) - treatment_wins

# Effektstärken bewerten
avg_effect = np.mean([abs(r.effect_size) for r in significant_results])
effect_interpretation = (
"großer Effekt" if avg_effect > 0.8
else "mittlerer Effekt" if avg_effect > 0.5
else "kleiner Effekt"
)

if treatment_wins > control_wins:
return (
f"Treatment ({test.config.treatment_group}) zeigt signifikant bessere Ergebnisse "
f"in {treatment_wins}/{len(significant_results)} Metriken ({effect_interpretation}: d={avg_effect:.2f}). "
f"Empfehlung: {test.config.treatment_group.upper()} bevorzugen."
)
elif control_wins > treatment_wins:
return (
f"Control ({test.config.control_group}) zeigt signifikant bessere Ergebnisse "
f"in {control_wins}/{len(significant_results)} Metriken ({effect_interpretation}: d={avg_effect:.2f}). "
f"Empfehlung: {test.config.control_group.upper()} bevorzugen."
)
else:
return (
f"Ausgewogene Ergebnisse: {treatment_wins} signifikante Metriken pro Gruppe. "
f"Weitere Analyse empfohlen."
)

def list_tests(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
"""
Liste alle Tests.

Args:
status: Optionaler Filter ("running", "completed", "paused")

Returns:
Liste von Test-Zusammenfassungen
"""
tests = list(self._tests.values())
if status:
tests = [t for t in tests if t.status == status]

return [
{
"test_id": t.test_id,
"test_name": t.config.test_name,
"status": t.status,
"treatment_runs": len(t.treatment_outcomes),
"control_runs": len(t.control_outcomes),
"created_at": t.created_at.isoformat(),
}
for t in tests
]

def delete_test(self, test_id: str) -> bool:
"""
Lösche einen Test.

Args:
test_id: Test-ID

Returns:
True wenn gelöscht, False wenn nicht existiert
"""
if test_id in self._tests:
del self._tests[test_id]
self._save()
return True
return False


# Import für Path benötigt
from pathlib import Path
import json
