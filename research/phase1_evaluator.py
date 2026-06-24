"""Phase 1 Erfolgskriterien und Metriken.

Dieses Modul implementiert die Erfolgskriterien und Kill-Regeln
für Phase 1 Runs laut roadmap_runs.md.

Phase 1 Fokus: Baseline und erste Ablationen
- run001_control: Control Baseline
- run001b_frontierish_control: Stärkere Baseline
- run002a_bigram_4k: Bigram Hash 4K
- run002b_bigram_8k: Bigram Hash 8K
- run002c_trigram_small: Trigram Small
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunType(Enum):
"""Typen von Runs für Phase 1."""
CONTROL = "control"
FRONTIERISH = "frontierish"
BIGRAM_4K = "bigram_4k"
BIGRAM_8K = "bigram_8k"
TRIGRAM_SMALL = "trigram_small"


@dataclass
class Phase1SuccessCriteria:
"""Erfolgskriterien für Phase 1 Runs."""

# Challenge (H100, submission) Kriterien
challenge_val_bpb_threshold: float = 1.50 # Für Control
challenge_ms_per_step_threshold: float = 50.0 # ms auf H100
challenge_artifact_bytes_max: int = 10_000_000
challenge_sigma_threshold: float = 0.02 # Reproduzierbarkeit

# Lokal (8GB, proxy) Kriterien
local_peak_vram_mb_max: float = 7500.0
local_steps_in_budget_min: int = 10 # In 5 Min
local_oom_count_max: int = 0

# Run-spezifische Kriterien
run_type: RunType = RunType.CONTROL

def get_bpb_threshold(self) -> float:
"""Get BPB threshold für Run-Typ."""
if self.run_type == RunType.CONTROL:
return 1.50
elif self.run_type == RunType.FRONTIERISH:
return 1.48 # Besser als Control
elif self.run_type in [RunType.BIGRAM_4K, RunType.BIGRAM_8K, RunType.TRIGRAM_SMALL]:
return 1.50 # Gleiche Baseline wie Control
return 1.50

def get_artifact_max(self, run_type: RunType | None = None) -> int:
"""Get Artifact-Größen-Limit."""
rt = run_type or self.run_type
if rt == RunType.FRONTIERISH:
return 9_000_000 # Kleineres Vocab
return 10_000_000

def check_challenge_criteria(self, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
"""Prüfe Challenge-Erfolgskriterien.

Args:
metrics: Dictionary mit Run-Metriken

Returns:
Tuple of (success, list of failed criteria)
"""
failed = []

# BPB-Threshold
val_bpb = metrics.get("val_bpb")
if val_bpb is not None:
threshold = self.get_bpb_threshold()
if val_bpb > threshold:
failed.append(f"val_bpb={val_bpb:.4f} > {threshold} (Threshold)")

# Step-Zeit
ms_per_step = metrics.get("ms_per_step")
if ms_per_step is not None and ms_per_step > self.challenge_ms_per_step_threshold:
failed.append(f"ms_per_step={ms_per_step:.2f} > {self.challenge_ms_per_step_threshold}ms")

# Artifact-Größe
artifact_bytes = metrics.get("artifact_bytes", 0)
artifact_max = self.get_artifact_max()
if artifact_bytes > artifact_max:
failed.append(f"artifact_bytes={artifact_bytes:,} > {artifact_max:,} bytes")

return len(failed) == 0, failed

def check_local_criteria(self, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
"""Prüfe lokale Erfolgskriterien.

Args:
metrics: Dictionary mit Run-Metriken

Returns:
Tuple of (success, list of failed criteria)
"""
failed = []

# VRAM-Limit
peak_vram_mb = metrics.get("peak_vram_mb", 0)
if peak_vram_mb > self.local_peak_vram_mb_max:
failed.append(f"peak_vram_mb={peak_vram_mb:.0f} > {self.local_peak_vram_mb_max:.0f} MB")

# OOM-Fehler
oom_count = metrics.get("oom_count", 0)
if oom_count > self.local_oom_count_max:
failed.append(f"oom_count={oom_count} > {self.local_oom_count_max}")

# Steps im Budget
steps_completed = metrics.get("steps_completed", 0)
if steps_completed < self.local_steps_in_budget_min:
failed.append(f"steps_completed={steps_completed} < {self.local_steps_in_budget_min}")

return len(failed) == 0, failed

def check_kill_criteria(self, metrics: dict[str, Any]) -> tuple[bool, str | None]:
"""Prüfe Kill-Kriterien für Phase 1.

Kill-Kriterien laut Roadmap:
- val_bpb > 1.60 → Architektur zu schwach
- ms_per_step > 75ms → Ineffiziente Implementierung
- artifact_bytes > 12.000.000 → Zu wenig Headroom
- Lokal: OOM bei Default-Konfiguration

Returns:
Tuple of (should_kill, reason)
"""
# val_bpb > 1.60
val_bpb = metrics.get("val_bpb")
if val_bpb is not None and val_bpb > 1.60:
return True, f"val_bpb={val_bpb:.4f} > 1.60 (Architektur zu schwach)"

# ms_per_step > 75ms
ms_per_step = metrics.get("ms_per_step")
if ms_per_step is not None and ms_per_step > 75.0:
return True, f"ms_per_step={ms_per_step:.2f}ms > 75ms (Ineffiziente Implementierung)"

# artifact_bytes > 12.000.000
artifact_bytes = metrics.get("artifact_bytes", 0)
if artifact_bytes > 12_000_000:
return True, f"artifact_bytes={artifact_bytes:,} > 12.000.000 (Zu wenig Headroom)"

# Lokal: OOM
oom_count = metrics.get("oom_count", 0)
if oom_count > 0:
return True, f"oom_count={oom_count} > 0 (OOM bei Default-Konfiguration)"

return False, None


@dataclass
class Phase1Metrics:
"""Metriken für Phase 1 Runs."""

# Core-Metriken
val_bpb: float | None = None
ms_per_step: float | None = None
steps_completed: int = 0
artifact_bytes: int = 0

# Quantisiert (optional)
quantized_val_bpb: float | None = None

# Lokal-spezifisch
peak_vram_mb: float | None = None
avg_vram_mb: float | None = None
tokens_per_sec: float | None = None
oom_count: int = 0
compile_time_s: float | None = None

# Relative Metriken
delta_bpb_vs_parent: float | None = None
delta_ms_vs_parent: float | None = None
bpb_per_mb: float | None = None
bpb_per_ms: float | None = None

# Metadata
run_id: str = ""
run_type: RunType = RunType.CONTROL
is_local_proxy: bool = False
is_smoke_test: bool = False

def to_dict(self) -> dict[str, Any]:
"""Convert to dictionary."""
return {
"val_bpb": self.val_bpb,
"ms_per_step": self.ms_per_step,
"steps_completed": self.steps_completed,
"artifact_bytes": self.artifact_bytes,
"quantized_val_bpb": self.quantized_val_bpb,
"peak_vram_mb": self.peak_vram_mb,
"avg_vram_mb": self.avg_vram_mb,
"tokens_per_sec": self.tokens_per_sec,
"oom_count": self.oom_count,
"compile_time_s": self.compile_time_s,
"delta_bpb_vs_parent": self.delta_bpb_vs_parent,
"delta_ms_vs_parent": self.delta_ms_vs_parent,
"bpb_per_mb": self.bpb_per_mb,
"bpb_per_ms": self.bpb_per_ms,
"run_id": self.run_id,
"run_type": self.run_type.value,
"is_local_proxy": self.is_local_proxy,
"is_smoke_test": self.is_smoke_test,
}

def compute_derived_metrics(self) -> None:
"""Berechne abgeleitete Metriken."""
# BPB per MB
if self.val_bpb is not None and self.artifact_bytes > 0:
artifact_mb = self.artifact_bytes / 1_000_000
self.bpb_per_mb = self.val_bpb / artifact_mb

# BPB per ms
if self.val_bpb is not None and self.ms_per_step is not None and self.ms_per_step > 0:
self.bpb_per_ms = self.val_bpb / self.ms_per_step


@dataclass
class Phase1Report:
"""Bericht für Phase 1 Runs."""

generated_at: str = ""
run_id: str = ""
run_type: str = ""
status: str = "" # success, warning, failed, killed

# Metriken
metrics: dict[str, Any] = field(default_factory=dict)

# Erfolgskriterien
challenge_success: bool = False
challenge_failures: list[str] = field(default_factory=list)
local_success: bool = False
local_failures: list[str] = field(default_factory=list)

# Kill-Status
should_kill: bool = False
kill_reason: str | None = None

# Empfehlungen
recommendation: str = ""
next_steps: list[str] = field(default_factory=list)

def to_dict(self) -> dict[str, Any]:
"""Convert to dictionary."""
return {
"generated_at": self.generated_at,
"run_id": self.run_id,
"run_type": self.run_type,
"status": self.status,
"metrics": self.metrics,
"challenge_success": self.challenge_success,
"challenge_failures": self.challenge_failures,
"local_success": self.local_success,
"local_failures": self.local_failures,
"should_kill": self.should_kill,
"kill_reason": self.kill_reason,
"recommendation": self.recommendation,
"next_steps": self.next_steps,
}

def print_summary(self) -> str:
"""Drucke menschlich-lesbare Zusammenfassung."""
from datetime import datetime

status_icon = {
"success": "",
"warning": "",
"failed": "",
"killed": "",
}.get(self.status, "")

lines = [
"=" * 70,
f"PHASE 1 REPORT: {self.run_id}",
f"Status: {status_icon} {self.status.upper()}",
f"Generated: {self.generated_at or datetime.now().isoformat()}",
"=" * 70,
"",
"METRIKEN",
]

# Core-Metriken
if self.metrics.get("val_bpb") is not None:
lines.append(f" val_bpb: {self.metrics['val_bpb']:.4f}")
if self.metrics.get("ms_per_step") is not None:
lines.append(f" ms_per_step: {self.metrics['ms_per_step']:.2f} ms")
if self.metrics.get("steps_completed") is not None:
lines.append(f" steps_completed: {self.metrics['steps_completed']}")
if self.metrics.get("artifact_bytes") is not None:
lines.append(f" artifact_size: {self.metrics['artifact_bytes'] / 1_000_000:.2f} MB")

# Lokal-spezifisch
if self.is_local_proxy:
lines.append("")
lines.append("LOKALE METRIKEN")
if self.metrics.get("peak_vram_mb") is not None:
lines.append(f" peak_vram_mb: {self.metrics['peak_vram_mb']:.0f} MB")
if self.metrics.get("tokens_per_sec") is not None:
lines.append(f" tokens_per_sec: {self.metrics['tokens_per_sec']:.1f}")
if self.metrics.get("oom_count") is not None:
lines.append(f" oom_count: {self.metrics['oom_count']}")

# Erfolgskriterien
lines.append("")
lines.append("ERFOLGSKRITERIEN")
lines.append(f" Challenge: {' PASS' if self.challenge_success else ' FAIL'}")
for failure in self.challenge_failures:
lines.append(f" - {failure}")

lines.append(f" Lokal: {' PASS' if self.local_success else ' FAIL'}")
for failure in self.local_failures:
lines.append(f" - {failure}")

# Kill-Status
if self.should_kill:
lines.append("")
lines.append(f"KILL-STATUS: {self.kill_reason}")

# Empfehlung
if self.recommendation:
lines.append("")
lines.append(f"EMPFEHLUNG: {self.recommendation}")

lines.append("")
lines.append("=" * 70)
return "\n".join(lines)

@property
def is_local_proxy(self) -> bool:
"""Check if this is a local proxy run."""
return self.metrics.get("is_local_proxy", False)


class Phase1Evaluator:
"""Evaluator für Phase 1 Runs."""

def __init__(self, run_id: str, run_type: RunType | None = None):
self.run_id = run_id
self.run_type = run_type or self._infer_run_type(run_id)
self.criteria = Phase1SuccessCriteria(run_type=self.run_type)

def _infer_run_type(self, run_id: str) -> RunType:
"""Infer RunType aus run_id."""
if "control" in run_id and "frontierish" not in run_id:
return RunType.CONTROL
elif "frontierish" in run_id:
return RunType.FRONTIERISH
elif "bigram_4k" in run_id or "bigram_4k" in run_id:
return RunType.BIGRAM_4K
elif "bigram_8k" in run_id:
return RunType.BIGRAM_8K
elif "trigram" in run_id:
return RunType.TRIGRAM_SMALL
return RunType.CONTROL

def evaluate(
self,
metrics: dict[str, Any],
parent_metrics: dict[str, Any] | None = None,
) -> Phase1Report:
"""Evaluiere einen Phase 1 Run.

Args:
metrics: Run-Metriken
parent_metrics: Optional Parent-Run-Metriken für Deltas

Returns:
Phase1Report
"""
from datetime import datetime

# Erweiterte Metriken berechnen
extended_metrics = metrics.copy()
extended_metrics["run_type"] = self.run_type.value
extended_metrics["is_local_proxy"] = metrics.get("is_local_proxy", False)
extended_metrics["is_smoke_test"] = metrics.get("is_smoke_test", False)

# Relative Deltas berechnen
if parent_metrics:
if parent_metrics.get("val_bpb") and metrics.get("val_bpb"):
extended_metrics["delta_bpb_vs_parent"] = (
metrics["val_bpb"] - parent_metrics["val_bpb"]
)
if parent_metrics.get("ms_per_step") and metrics.get("ms_per_step"):
extended_metrics["delta_ms_vs_parent"] = (
metrics["ms_per_step"] - parent_metrics["ms_per_step"]
)

# Erfolgskriterien prüfen
challenge_success, challenge_failures = self.criteria.check_challenge_criteria(extended_metrics)
local_success, local_failures = self.criteria.check_local_criteria(extended_metrics)
should_kill, kill_reason = self.criteria.check_kill_criteria(extended_metrics)

# Status bestimmen
if should_kill:
status = "killed"
elif challenge_success and local_success:
status = "success"
elif challenge_failures or local_failures:
status = "failed"
else:
status = "warning"

# Empfehlung generieren
recommendation = self._generate_recommendation(
status, challenge_success, local_success, kill_reason
)

# Nächste Schritte
next_steps = self._generate_next_steps(status, self.run_type)

report = Phase1Report(
generated_at=datetime.now().isoformat(),
run_id=self.run_id,
run_type=self.run_type.value,
status=status,
metrics=extended_metrics,
challenge_success=challenge_success,
challenge_failures=challenge_failures,
local_success=local_success,
local_failures=local_failures,
should_kill=should_kill,
kill_reason=kill_reason,
recommendation=recommendation,
next_steps=next_steps,
)

return report

def _generate_recommendation(
self,
status: str,
challenge_success: bool,
local_success: bool,
kill_reason: str | None,
) -> str:
"""Generiere Empfehlung basierend auf Ergebnissen."""
if kill_reason:
return f"Run stoppen: {kill_reason}"

if status == "success":
return "Run erfolgreich - bereit für nächste Phase oder Kombination"

if status == "failed":
failures = []
if not challenge_success:
failures.append("Challenge-Kriterien")
if not local_success:
failures.append("Lokale Kriterien")
return f"Run gescheitert ({', '.join(failures)}) - Konfiguration überprüfen"

if status == "warning":
return "Run mit Warnungen - weitere Evaluation empfohlen"

return "Keine Empfehlung verfügbar"

def _generate_next_steps(self, status: str, run_type: RunType) -> list[str]:
"""Generiere nächste Schritte basierend auf Status."""
steps = []

if status == "success":
if run_type == RunType.CONTROL:
steps = [
"Control-Baseline etabliert - proceed to run001b_frontierish",
"Tokenizer-Varianten testen (run002a/b/c)",
]
elif run_type in [RunType.BIGRAM_4K, RunType.BIGRAM_8K, RunType.TRIGRAM_SMALL]:
steps = [
"Tokenizer-Performance dokumentieren",
"Besten Tokenizer für Phase 2 auswählen",
]
elif run_type == RunType.FRONTIERISH:
steps = [
"Frontier-ish als alternative Baseline verwenden",
"Feature-Gates in Phase 2 testen",
]

elif status == "killed":
steps = [
"Kill-Grund analysieren",
"Konfiguration anpassen oder Feature entfernen",
]

elif status == "failed":
steps = [
"Fehlerursache untersuchen",
"Lokale Proxy-Konfiguration anpassen (seq_len, microbatch)",
"Smoke-Test zuerst ausführen",
]

return steps


def create_phase1_evaluator(run_id: str) -> Phase1Evaluator:
"""Convenience function to create a Phase1Evaluator."""
return Phase1Evaluator(run_id)
