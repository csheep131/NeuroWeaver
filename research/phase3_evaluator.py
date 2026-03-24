"""Phase 3 Erfolgskriterien und Eval-Logik.

Dieses Modul implementiert die Erfolgskriterien für Phase 3 Runs:
- run016_best_combo_a: Beste nicht-quantisierte Kombination
- run017_best_combo_quantized: Beste quantisierte Kombination

Phase 3 Fokus: Finale Kandidaten und Submission
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RunType(Enum):
    """Typen von Phase 3 Runs."""
    BEST_COMBO_A = "best_combo_a"
    BEST_COMBO_QUANTIZED = "best_combo_quantized"
    MULTI_SEED_CONTROL = "multi_seed_control"
    MULTI_SEED_COMBO = "multi_seed_combo"
    MULTI_SEED_QUANT_COMBO = "multi_seed_quant_combo"


@dataclass
class Phase3SuccessCriteria:
    """Erfolgskriterien für Phase 3 Runs."""

    # Challenge Submission Limits
    max_artifact_bytes: int = 16_000_000  # 16 MB hard limit
    max_val_bpb: float = 1.50
    max_quantized_val_bpb: float = 1.50
    max_ms_per_step: float = 75.0  # Für 10min Limit

    # Quant-Gap
    max_quant_gap: float = 0.05  # Für quantisierte Runs

    # Seed-Stabilität
    max_bpb_std: float = 0.03  # Über 3 Seeds

    # Run-spezifisch
    run_type: RunType = RunType.BEST_COMBO_A

    def check_submission_criteria(
        self,
        metrics: dict[str, Any],
        seed_metrics: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, list[str]]:
        """Prüfe Submission-Kriterien.

        Args:
            metrics: Dictionary mit Run-Metriken
            seed_metrics: Optional Metriken für mehrere Seeds

        Returns:
            Tuple of (meets_criteria, list of failures)
        """
        failures = []

        # Artifact-Größe (hard limit)
        artifact_bytes = metrics.get("artifact_bytes", 0)
        if artifact_bytes > self.max_artifact_bytes:
            failures.append(
                f"artifact_bytes={artifact_bytes:,} > {self.max_artifact_bytes:,} "
                "(disqualifiziert)"
            )

        # BPB-Threshold
        val_bpb = metrics.get("val_bpb")
        if val_bpb is not None and val_bpb > self.max_val_bpb:
            failures.append(f"val_bpb={val_bpb:.4f} > {self.max_val_bpb}")

        # Quantized BPB (für quantisierte Runs)
        if self.run_type == RunType.BEST_COMBO_QUANTIZED:
            quant_bpb = metrics.get("quantized_val_bpb")
            if quant_bpb is not None and quant_bpb > self.max_quantized_val_bpb:
                failures.append(f"quantized_val_bpb={quant_bpb:.4f} > {self.max_quantized_val_bpb}")

            # Quant-Gap
            if quant_bpb is not None and val_bpb is not None:
                gap = quant_bpb - val_bpb
                if gap > self.max_quant_gap:
                    failures.append(f"Quant-Gap {gap:.4f} > {self.max_quant_gap}")

        # Step-Zeit
        ms_per_step = metrics.get("ms_per_step")
        if ms_per_step is not None and ms_per_step > self.max_ms_per_step:
            failures.append(f"ms_per_step={ms_per_step:.2f}ms > {self.max_ms_per_step}ms")

        # Seed-Stabilität (wenn mehrere Seeds)
        if seed_metrics and len(seed_metrics) >= 3:
            bpb_values = [m.get("val_bpb") for m in seed_metrics if m.get("val_bpb") is not None]
            if len(bpb_values) >= 3:
                mean_bpb = sum(bpb_values) / len(bpb_values)
                variance = sum((v - mean_bpb) ** 2 for v in bpb_values) / len(bpb_values)
                std_bpb = variance ** 0.5

                if std_bpb > self.max_bpb_std:
                    failures.append(f"bpb_std={std_bpb:.4f} > {self.max_bpb_std} (zu volatil)")

        return len(failures) == 0, failures

    def check_combo_synergy(
        self,
        metrics: dict[str, Any],
        parent_metrics: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        """Prüfe ob Kombination Synergie zeigt.

        Args:
            metrics: Combo-Run-Metriken
            parent_metrics: Parent-Run-Metriken (beste Einzel-Features)

        Returns:
            Tuple of (has_synergy, reason if not)
        """
        if not parent_metrics:
            return True, None  # Can't check without parent

        # Check if combo is better than best single feature
        combo_bpb = metrics.get("val_bpb")
        parent_bpb = parent_metrics.get("val_bpb")

        if combo_bpb is not None and parent_bpb is not None:
            if combo_bpb > parent_bpb + 0.02:  # Combo worse than best single
                return False, f"Combo BPB {combo_bpb:.4f} > single feature BPB {parent_bpb:.4f}"

        # Check artifact size
        combo_artifact = metrics.get("artifact_bytes", 0)
        parent_artifact = parent_metrics.get("artifact_bytes", 0)

        if combo_artifact > parent_artifact * 1.5:  # 50% increase
            return False, f"Combo artifact size {combo_artifact:,} > 1.5x single feature {parent_artifact:,}"

        return True, None


@dataclass
class Phase3Metrics:
    """Metriken für Phase 3 Runs."""

    # Core-Metriken
    val_bpb: float | None = None
    ms_per_step: float | None = None
    steps_completed: int = 0
    artifact_bytes: int = 0

    # Quantisiert
    quantized_val_bpb: float | None = None
    quant_gap: float | None = None

    # Seed-Statistiken
    num_seeds: int = 1
    bpb_mean: float | None = None
    bpb_std: float | None = None
    bpb_min: float | None = None
    bpb_max: float | None = None

    # Combo-spezifisch
    synergy_score: float | None = None  # Wie viel besser als Einzel-Features
    features_combined: list[str] = field(default_factory=list)

    # Metadata
    run_id: str = ""
    run_type: RunType = RunType.BEST_COMBO_A
    is_submission_ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "val_bpb": self.val_bpb,
            "ms_per_step": self.ms_per_step,
            "steps_completed": self.steps_completed,
            "artifact_bytes": self.artifact_bytes,
            "quantized_val_bpb": self.quantized_val_bpb,
            "quant_gap": self.quant_gap,
            "num_seeds": self.num_seeds,
            "bpb_mean": self.bpb_mean,
            "bpb_std": self.bpb_std,
            "bpb_min": self.bpb_min,
            "bpb_max": self.bpb_max,
            "synergy_score": self.synergy_score,
            "features_combined": self.features_combined,
            "run_id": self.run_id,
            "run_type": self.run_type.value,
            "is_submission_ready": self.is_submission_ready,
        }


@dataclass
class Phase3Report:
    """Bericht für Phase 3 Runs."""

    generated_at: str = ""
    run_id: str = ""
    run_type: str = ""
    status: str = ""  # success, warning, failed, killed

    # Metriken
    metrics: dict[str, Any] = field(default_factory=dict)

    # Submission-Kriterien
    submission_ready: bool = False
    submission_failures: list[str] = field(default_factory=list)

    # Combo-Synergie
    has_synergy: bool = False
    synergy_reason: str | None = None

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
            "submission_ready": self.submission_ready,
            "submission_failures": self.submission_failures,
            "has_synergy": self.has_synergy,
            "synergy_reason": self.synergy_reason,
            "should_kill": self.should_kill,
            "kill_reason": self.kill_reason,
            "recommendation": self.recommendation,
            "next_steps": self.next_steps,
        }

    def print_summary(self) -> str:
        """Drucke menschlich-lesbare Zusammenfassung."""
        status_icon = {
            "success": "✅",
            "warning": "⚠️",
            "failed": "❌",
            "killed": "🔴",
        }.get(self.status, "❓")

        submission_icon = "✅" if self.submission_ready else "❌"

        lines = [
            "=" * 70,
            f"PHASE 3 REPORT: {self.run_id}",
            f"Status: {status_icon} {self.status.upper()}",
            f"Submission Ready: {submission_icon}",
            f"Generated: {self.generated_at or datetime.now().isoformat()}",
            "=" * 70,
            "",
            "METRIKEN",
        ]

        # Core-Metriken
        if self.metrics.get("val_bpb") is not None:
            lines.append(f"  val_bpb:          {self.metrics['val_bpb']:.4f}")
        if self.metrics.get("quantized_val_bpb") is not None:
            lines.append(f"  quantized_val_bpb: {self.metrics['quantized_val_bpb']:.4f}")
        if self.metrics.get("artifact_bytes") is not None:
            lines.append(f"  artifact_size:    {self.metrics['artifact_bytes'] / 1_000_000:.2f} MB")
        if self.metrics.get("ms_per_step") is not None:
            lines.append(f"  ms_per_step:      {self.metrics['ms_per_step']:.2f} ms")

        # Seed-Statistiken
        if self.metrics.get("num_seeds", 1) > 1:
            lines.append("")
            lines.append("SEED-STATISTIKEN")
            lines.append(f"  num_seeds:  {self.metrics.get('num_seeds')}")
            if self.metrics.get("bpb_mean") is not None:
                lines.append(f"  bpb_mean:   {self.metrics['bpb_mean']:.4f}")
            if self.metrics.get("bpb_std") is not None:
                lines.append(f"  bpb_std:    {self.metrics['bpb_std']:.4f}")

        # Combo-spezifisch
        if self.metrics.get("features_combined"):
            lines.append("")
            lines.append("KOMBINIERTE FEATURES")
            for feature in self.metrics["features_combined"]:
                lines.append(f"  - {feature}")

        # Submission-Status
        lines.append("")
        lines.append("SUBMISSION-STATUS")
        lines.append(f"  Ready: {'✅ YES' if self.submission_ready else '❌ NO'}")
        for failure in self.submission_failures:
            lines.append(f"    - {failure}")

        # Synergie-Status
        if self.has_synergy:
            lines.append("")
            lines.append("COMBO-SYNERGIE: ✅ Positiv")
        elif self.synergy_reason:
            lines.append("")
            lines.append(f"COMBO-SYNERGIE: ❌ {self.synergy_reason}")

        # Kill-Status
        if self.should_kill:
            lines.append("")
            lines.append(f"KILL-STATUS: 🔴 {self.kill_reason}")

        # Empfehlung
        if self.recommendation:
            lines.append("")
            lines.append(f"EMPFEHLUNG: {self.recommendation}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


class Phase3Evaluator:
    """Evaluator für Phase 3 Runs."""

    def __init__(self, run_id: str, run_type: RunType | None = None):
        self.run_id = run_id
        self.run_type = run_type or self._infer_run_type(run_id)
        self.criteria = Phase3SuccessCriteria(run_type=self.run_type)

    def _infer_run_type(self, run_id: str) -> RunType:
        """Infer RunType aus run_id."""
        if "best_combo_quantized" in run_id:
            return RunType.BEST_COMBO_QUANTIZED
        elif "best_combo" in run_id or "combo_a" in run_id:
            return RunType.BEST_COMBO_A
        elif "control" in run_id and ("seed" in run_id or "_s" in run_id):
            return RunType.MULTI_SEED_CONTROL
        elif "quant" in run_id and ("seed" in run_id or "_s" in run_id):
            return RunType.MULTI_SEED_QUANT_COMBO
        elif "combo" in run_id and ("seed" in run_id or "_s" in run_id):
            return RunType.MULTI_SEED_COMBO
        return RunType.BEST_COMBO_A

    def evaluate(
        self,
        metrics: dict[str, Any],
        parent_metrics: dict[str, Any] | None = None,
        seed_metrics: list[dict[str, Any]] | None = None,
    ) -> Phase3Report:
        """Evaluiere einen Phase 3 Run.

        Args:
            metrics: Run-Metriken
            parent_metrics: Optional Parent-Run-Metriken für Synergie-Check
            seed_metrics: Optional Metriken für mehrere Seeds

        Returns:
            Phase3Report
        """
        # Erweiterte Metriken
        extended_metrics = metrics.copy()
        extended_metrics["run_type"] = self.run_type.value

        # Quant-Gap berechnen
        if metrics.get("quantized_val_bpb") and metrics.get("val_bpb"):
            extended_metrics["quant_gap"] = (
                metrics["quantized_val_bpb"] - metrics["val_bpb"]
            )

        # Seed-Statistiken
        if seed_metrics and len(seed_metrics) >= 2:
            bpb_values = [m.get("val_bpb") for m in seed_metrics if m.get("val_bpb")]
            if bpb_values:
                extended_metrics["num_seeds"] = len(bpb_values)
                extended_metrics["bpb_mean"] = sum(bpb_values) / len(bpb_values)
                extended_metrics["bpb_min"] = min(bpb_values)
                extended_metrics["bpb_max"] = max(bpb_values)
                variance = sum(
                    (v - extended_metrics["bpb_mean"]) ** 2 for v in bpb_values
                ) / len(bpb_values)
                extended_metrics["bpb_std"] = variance ** 0.5

        # Submission-Kriterien prüfen
        submission_ready, submission_failures = self.criteria.check_submission_criteria(
            extended_metrics, seed_metrics
        )

        # Combo-Synergie prüfen
        has_synergy, synergy_reason = self.criteria.check_combo_synergy(
            extended_metrics, parent_metrics
        )

        # Kill-Kriterien prüfen
        should_kill, kill_reason = self._check_kill_criteria(
            extended_metrics, submission_ready
        )

        # Status bestimmen
        if should_kill:
            status = "killed"
        elif submission_ready and has_synergy:
            status = "success"
        elif submission_failures:
            status = "failed"
        else:
            status = "warning"

        # Empfehlung generieren
        recommendation = self._generate_recommendation(
            status, submission_ready, has_synergy, self.run_type
        )

        # Nächste Schritte
        next_steps = self._generate_next_steps(status, submission_ready, self.run_type)

        report = Phase3Report(
            generated_at=datetime.now().isoformat(),
            run_id=self.run_id,
            run_type=self.run_type.value,
            status=status,
            metrics=extended_metrics,
            submission_ready=submission_ready,
            submission_failures=submission_failures,
            has_synergy=has_synergy,
            synergy_reason=synergy_reason,
            should_kill=should_kill,
            kill_reason=kill_reason,
            recommendation=recommendation,
            next_steps=next_steps,
        )

        return report

    def _check_kill_criteria(
        self,
        metrics: dict[str, Any],
        submission_ready: bool,
    ) -> tuple[bool, str | None]:
        """Prüfe Kill-Kriterien für Phase 3."""
        # Artifact > 16MB (hard disqualification)
        if metrics.get("artifact_bytes", 0) > 16_000_000:
            return True, "artifact_bytes > 16MB (disqualifiziert)"

        # Quant-Gap zu groß
        if metrics.get("quant_gap", 0) > 0.08:
            return True, f"Quant-Gap {metrics['quant_gap']:.4f} > 0.08"

        # BPB zu schlecht
        if metrics.get("val_bpb", 0) > 1.60:
            return True, f"val_bpb={metrics['val_bpb']:.4f} > 1.60 (zu schlecht)"

        return False, None

    def _generate_recommendation(
        self,
        status: str,
        submission_ready: bool,
        has_synergy: bool,
        run_type: RunType,
    ) -> str:
        """Generiere Empfehlung."""
        if status == "killed":
            return "Run gestoppt - Kriterien nicht erfüllt"

        if submission_ready and has_synergy:
            if run_type == RunType.BEST_COMBO_QUANTIZED:
                return "Bereit für Submission - alle Kriterien erfüllt"
            return "Bereit für Multi-Seed Validierung (H100)"

        if not submission_ready:
            return "Nicht bereit für Submission - Kriterien überprüfen"

        if not has_synergy:
            return "Keine Combo-Synergie - Einzel-Features bevorzugen"

        return "Weitere Evaluation empfohlen"

    def _generate_next_steps(
        self,
        status: str,
        submission_ready: bool,
        run_type: RunType,
    ) -> list[str]:
        """Generiere nächste Schritte."""
        steps = []

        if submission_ready and status != "killed":
            if run_type == RunType.BEST_COMBO_QUANTIZED:
                steps = [
                    "Submission Bundle erstellen",
                    "Finale Validierung auf H100",
                    "Multi-Seed-Tests (3 Seeds)",
                ]
            elif run_type == RunType.BEST_COMBO_A:
                steps = [
                    "Quantisierte Version erstellen (run017)",
                    "Multi-Seed-Tests vorbereiten",
                ]

        if not submission_ready:
            steps = [
                "Kriterien-Analyse durchführen",
                "Feature-Kombination anpassen",
                "Alternative Kombinationen testen",
            ]

        if status == "killed":
            steps = [
                "Zurück zu Phase 2 Einzel-Features",
                "Alternative Feature-Kombinationen evaluieren",
            ]

        return steps


def create_phase3_evaluator(run_id: str) -> Phase3Evaluator:
    """Convenience function to create a Phase3Evaluator."""
    return Phase3Evaluator(run_id)
