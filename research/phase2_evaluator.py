"""Phase 2 Erfolgskriterien und Metriken.

Dieses Modul implementiert die Erfolgskriterien und Kill-Regeln
für Phase 2 Runs laut roadmap_runs.md.

Phase 2 Fokus: Feature-Gates und Research
- run003_xsa: Cross-Sequence Attention
- run004_leakyrelu: LeakyReLU² Aktivierung
- run005a/b_quant: Mixed-Precision Quantisierung
- run006_film: FiLM Feature
- run007_ttt: TTT Feature (Late-Stage)
- run008a_star_relu: Star-ReLU Aktivierung
- run008b_gated_mlp: Gated MLP
- run009_gqa: GQA Attention
- run010_recurrence: Recurrent Blocks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RunType(Enum):
    """Typen von Phase 2 Runs."""
    XSA = "xsa"
    LEAKY_RELU = "leaky_relu"
    QUANT_MLP5_ATTN6 = "quant_mlp5_attn6"
    QUANT_ATTN5_MLP6 = "quant_attn5_mlp6"
    FILM = "film"
    TTT = "ttt"
    STAR_RELU = "star_relu"
    GATED_MLP = "gated_mlp"
    GQA = "gqa"
    RECURRENCE = "recurrence"


@dataclass
class Phase2SuccessCriteria:
    """Erfolgskriterien für Phase 2 Runs."""

    # Challenge (H100, submission) Kriterien
    challenge_artifact_bytes_max: int = 16_000_000  # Globales Limit
    challenge_ms_per_step_max: int = 100  # ms auf H100

    # Quant-Gap Thresholds
    max_quant_gap: float = 0.08  # Maximum acceptable BPB degradation

    # Run-spezifische Kriterien
    run_type: RunType = RunType.XSA

    def get_bpb_improvement_threshold(self) -> float:
        """Get minimale BPB-Verbesserung für Run-Typ."""
        thresholds = {
            RunType.XSA: 0.05,
            RunType.LEAKY_RELU: 0.0,  # Within ±0.02 OK
            RunType.QUANT_MLP5_ATTN6: 0.0,  # Within +0.03 OK
            RunType.QUANT_ATTN5_MLP6: 0.0,  # Within +0.03 OK
            RunType.FILM: 0.04,
            RunType.TTT: 0.06,
            RunType.STAR_RELU: 0.0,  # Within ±0.02 OK
            RunType.GATED_MLP: 0.04,
            RunType.GQA: 0.0,  # Within ±0.02 OK
            RunType.RECURRENCE: 0.05,
        }
        return thresholds.get(self.run_type, 0.05)

    def get_ms_increase_threshold(self) -> float:
        """Get maximale ms-Erhöhung für Run-Typ."""
        thresholds = {
            RunType.XSA: 20.0,  # %
            RunType.LEAKY_RELU: -10.0,  # Should be faster
            RunType.QUANT_MLP5_ATTN6: 0.0,  # No increase expected
            RunType.QUANT_ATTN5_MLP6: 0.0,
            RunType.FILM: 15.0,
            RunType.TTT: 30.0,  # Inference overhead
            RunType.STAR_RELU: -8.0,  # Should be faster
            RunType.GATED_MLP: 20.0,
            RunType.GQA: -15.0,  # Should be faster
            RunType.RECURRENCE: 40.0,
        }
        return thresholds.get(self.run_type, 20.0)

    def get_artifact_increase_max(self) -> int:
        """Get maximale Artifact-Erhöhung in bytes."""
        thresholds = {
            RunType.XSA: 1_000_000,
            RunType.LEAKY_RELU: 0,
            RunType.QUANT_MLP5_ATTN6: -3_000_000,  # Should be smaller
            RunType.QUANT_ATTN5_MLP6: -3_000_000,
            RunType.FILM: 1_000_000,
            RunType.TTT: 0,
            RunType.STAR_RELU: 0,
            RunType.GATED_MLP: 3_000_000,
            RunType.GQA: 0,
            RunType.RECURRENCE: -2_000_000,  # Should be smaller (parameter sharing)
        }
        return thresholds.get(self.run_type, 1_000_000)

    def check_challenge_criteria(
        self,
        metrics: dict[str, Any],
        parent_metrics: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        """Prüfe Challenge-Erfolgskriterien.

        Args:
            metrics: Dictionary mit Run-Metriken
            parent_metrics: Optional Parent-Run-Metriken für Vergleiche

        Returns:
            Tuple of (success, list of failed criteria)
        """
        failed = []

        # Artifact-Größe (globales Limit)
        artifact_bytes = metrics.get("artifact_bytes", 0)
        if artifact_bytes > self.challenge_artifact_bytes_max:
            failed.append(
                f"artifact_bytes={artifact_bytes:,} > {self.challenge_artifact_bytes_max:,} "
                "(Challenge-Limit)"
            )

        # Step-Zeit (globales Limit)
        ms_per_step = metrics.get("ms_per_step")
        if ms_per_step is not None and ms_per_step > self.challenge_ms_per_step_max:
            failed.append(
                f"ms_per_step={ms_per_step:.2f}ms > {self.challenge_ms_per_step_max}ms"
            )

        # BPB-Verbesserung vs Parent
        if parent_metrics:
            bpb_improvement = self.get_bpb_improvement_threshold()
            if parent_metrics.get("val_bpb") and metrics.get("val_bpb"):
                delta_bpb = metrics["val_bpb"] - parent_metrics["val_bpb"]
                if bpb_improvement > 0:
                    # Positive improvement required
                    if delta_bpb > -bpb_improvement:
                        failed.append(
                            f"BPB improvement {delta_bpb:.4f} < required {bpb_improvement}"
                        )
                else:
                    # Within tolerance
                    tolerance = abs(bpb_improvement) if bpb_improvement != 0 else 0.02
                    if delta_bpb > tolerance:
                        failed.append(
                            f"BPB regression {delta_bpb:.4f} > tolerance {tolerance}"
                        )

        # Quant-Gap für Quant-Runs
        if self.run_type in [RunType.QUANT_MLP5_ATTN6, RunType.QUANT_ATTN5_MLP6]:
            quant_bpb = metrics.get("quantized_val_bpb")
            val_bpb = metrics.get("val_bpb")
            if quant_bpb is not None and val_bpb is not None:
                gap = quant_bpb - val_bpb
                if gap > self.max_quant_gap:
                    failed.append(
                        f"Quant-Gap {gap:.4f} > {self.max_quant_gap} (zu aggressiv)"
                    )

        return len(failed) == 0, failed

    def check_local_criteria(
        self,
        metrics: dict[str, Any],
        run_type_specific: bool = True,
    ) -> tuple[bool, list[str]]:
        """Prüfe lokale Erfolgskriterien.

        Args:
            metrics: Dictionary mit Run-Metriken
            run_type_specific: Ob run-spezifische Kriterien prüfen

        Returns:
            Tuple of (success, list of failed criteria)
        """
        failed = []

        # VRAM-Limit
        peak_vram_mb = metrics.get("peak_vram_mb", 0)
        if peak_vram_mb > 7500:
            failed.append(f"peak_vram_mb={peak_vram_mb:.0f} > 7500 MB")

        # OOM-Fehler
        oom_count = metrics.get("oom_count", 0)
        if oom_count > 0:
            failed.append(f"oom_count={oom_count} > 0")

        # Run-spezifische Kriterien
        if run_type_specific:
            ms_threshold = self.get_ms_increase_threshold()
            if ms_threshold < 0:
                # Should be faster
                if metrics.get("relative_delta_vs_parent_ms") is not None:
                    delta_ms_pct = metrics["relative_delta_vs_parent_ms"]
                    if delta_ms_pct > ms_threshold:
                        failed.append(
                            f"ms/step nicht reduziert genug: {delta_ms_pct:.1f}% > {ms_threshold}%"
                        )

        return len(failed) == 0, failed

    def check_kill_criteria(
        self,
        metrics: dict[str, Any],
        parent_metrics: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        """Prüfe Kill-Kriterien für Phase 2.

        Kill-Kriterien laut Roadmap:
        - Artifact > 16.000.000 bytes → Disqualifiziert
        - ms/step > 100ms (H100) → Zu langsam
        - Quant-Gap > 0.08 → Zu aggressiv
        - Feature-spezifische Kill-Kriterien

        Returns:
            Tuple of (should_kill, reason)
        """
        # Artifact > 16MB
        artifact_bytes = metrics.get("artifact_bytes", 0)
        if artifact_bytes > 16_000_000:
            return True, f"artifact_bytes={artifact_bytes:,} > 16MB (disqualifiziert)"

        # ms/step > 100ms
        ms_per_step = metrics.get("ms_per_step")
        if ms_per_step is not None and ms_per_step > 100:
            return True, f"ms_per_step={ms_per_step:.2f}ms > 100ms (zu langsam)"

        # Quant-Gap > 0.08
        if self.run_type in [RunType.QUANT_MLP5_ATTN6, RunType.QUANT_ATTN5_MLP6]:
            quant_bpb = metrics.get("quantized_val_bpb")
            val_bpb = metrics.get("val_bpb")
            if quant_bpb is not None and val_bpb is not None:
                gap = quant_bpb - val_bpb
                if gap > 0.08:
                    return True, f"Quant-Gap {gap:.4f} > 0.08 (zu aggressiv)"

        # Feature-spezifische Kill-Kriterien
        if parent_metrics:
            ms_increase_threshold = self.get_ms_increase_threshold()
            if ms_increase_threshold > 0:
                # Check if ms increased too much
                if metrics.get("delta_ms") is not None:
                    if metrics["delta_ms"] > ms_increase_threshold:
                        return True, (
                            f"Step-Zeit erhöht um {metrics['delta_ms']:.2f}ms "
                            f"(Threshold: {ms_increase_threshold}ms)"
                        )

            # BPB regression
            if metrics.get("delta_bpb") is not None:
                max_regression = 0.1  # Generic threshold
                if metrics["delta_bpb"] > max_regression:
                    return True, (
                        f"BPB Regression: delta_bpb={metrics['delta_bpb']:.4f} "
                        f">(Threshold: {max_regression})"
                    )

        return False, None


@dataclass
class Phase2Metrics:
    """Metriken für Phase 2 Runs."""

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

    # Relative Metriken
    delta_bpb_vs_parent: float | None = None
    delta_ms_vs_parent: float | None = None
    relative_delta_vs_parent_bpb: float | None = None
    relative_delta_vs_parent_ms: float | None = None

    # Feature-spezifisch
    kv_cache_reduction: float | None = None  # Für GQA
    recurrence_overhead: float | None = None  # Für Recurrence
    film_parameter_pct: float | None = None  # Für FiLM

    # Metadata
    run_id: str = ""
    run_type: RunType = RunType.XSA
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
            "delta_bpb_vs_parent": self.delta_bpb_vs_parent,
            "delta_ms_vs_parent": self.delta_ms_vs_parent,
            "relative_delta_vs_parent_bpb": self.relative_delta_vs_parent_bpb,
            "relative_delta_vs_parent_ms": self.relative_delta_vs_parent_ms,
            "kv_cache_reduction": self.kv_cache_reduction,
            "recurrence_overhead": self.recurrence_overhead,
            "film_parameter_pct": self.film_parameter_pct,
            "run_id": self.run_id,
            "run_type": self.run_type.value,
            "is_local_proxy": self.is_local_proxy,
            "is_smoke_test": self.is_smoke_test,
        }

    def compute_derived_metrics(self, parent_metrics: dict[str, Any] | None = None) -> None:
        """Berechne abgeleitete Metriken."""
        if parent_metrics:
            # Relative Deltas
            if parent_metrics.get("val_bpb") and self.val_bpb:
                self.delta_bpb_vs_parent = self.val_bpb - parent_metrics["val_bpb"]
                if parent_metrics["val_bpb"] > 0:
                    self.relative_delta_vs_parent_bpb = (
                        self.delta_bpb_vs_parent / parent_metrics["val_bpb"] * 100
                    )

            if parent_metrics.get("ms_per_step") and self.ms_per_step:
                self.delta_ms_vs_parent = self.ms_per_step - parent_metrics["ms_per_step"]
                if parent_metrics["ms_per_step"] > 0:
                    self.relative_delta_vs_parent_ms = (
                        self.delta_ms_vs_parent / parent_metrics["ms_per_step"] * 100
                    )


@dataclass
class Phase2Report:
    """Bericht für Phase 2 Runs."""

    generated_at: str = ""
    run_id: str = ""
    run_type: str = ""
    status: str = ""  # success, warning, failed, killed

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

    # Gate-Status für Phase 3
    gate_status: str = "pending"  # pass, watch, fail

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
            "gate_status": self.gate_status,
        }

    def print_summary(self) -> str:
        """Drucke menschlich-lesbare Zusammenfassung."""
        status_icon = {
            "success": "✅",
            "warning": "⚠️",
            "failed": "❌",
            "killed": "🔴",
        }.get(self.status, "❓")

        gate_icon = {
            "pass": "✅",
            "watch": "⚠️",
            "fail": "❌",
            "pending": "⏳",
        }.get(self.gate_status, "❓")

        lines = [
            "=" * 70,
            f"PHASE 2 REPORT: {self.run_id}",
            f"Status: {status_icon} {self.status.upper()}",
            f"Gate-Status: {gate_icon} {self.gate_status.upper()}",
            f"Generated: {self.generated_at or datetime.now().isoformat()}",
            "=" * 70,
            "",
            "METRIKEN",
        ]

        # Core-Metriken
        if self.metrics.get("val_bpb") is not None:
            lines.append(f"  val_bpb:          {self.metrics['val_bpb']:.4f}")
        if self.metrics.get("ms_per_step") is not None:
            lines.append(f"  ms_per_step:      {self.metrics['ms_per_step']:.2f} ms")
        if self.metrics.get("steps_completed") is not None:
            lines.append(f"  steps_completed:  {self.metrics['steps_completed']}")
        if self.metrics.get("artifact_bytes") is not None:
            lines.append(f"  artifact_size:    {self.metrics['artifact_bytes'] / 1_000_000:.2f} MB")

        # Relative Metriken
        if self.metrics.get("delta_bpb_vs_parent") is not None:
            lines.append(f"  Δ BPB:            {self.metrics['delta_bpb_vs_parent']:+.4f}")
        if self.metrics.get("delta_ms_vs_parent") is not None:
            lines.append(f"  Δ ms/step:        {self.metrics['delta_ms_vs_parent']:+.2f} ms")

        # Lokal-spezifisch
        if self.is_local_proxy:
            lines.append("")
            lines.append("LOKALE METRIKEN")
            if self.metrics.get("peak_vram_mb") is not None:
                lines.append(f"  peak_vram_mb:     {self.metrics['peak_vram_mb']:.0f} MB")
            if self.metrics.get("tokens_per_sec") is not None:
                lines.append(f"  tokens_per_sec:   {self.metrics['tokens_per_sec']:.1f}")
            if self.metrics.get("oom_count") is not None:
                lines.append(f"  oom_count:        {self.metrics['oom_count']}")

        # Erfolgskriterien
        lines.append("")
        lines.append("ERFOLGSKRITERIEN")
        lines.append(f"  Challenge: {'✅ PASS' if self.challenge_success else '❌ FAIL'}")
        for failure in self.challenge_failures:
            lines.append(f"    - {failure}")

        lines.append(f"  Lokal:     {'✅ PASS' if self.local_success else '❌ FAIL'}")
        for failure in self.local_failures:
            lines.append(f"    - {failure}")

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


class Phase2Evaluator:
    """Evaluator für Phase 2 Runs."""

    def __init__(self, run_id: str, run_type: RunType | None = None):
        self.run_id = run_id
        self.run_type = run_type or self._infer_run_type(run_id)
        self.criteria = Phase2SuccessCriteria(run_type=self.run_type)

    def _infer_run_type(self, run_id: str) -> RunType:
        """Infer RunType aus run_id."""
        if "xsa" in run_id:
            return RunType.XSA
        elif "leakyrelu" in run_id:
            return RunType.LEAKY_RELU
        elif "quant_mlp5_attn6" in run_id:
            return RunType.QUANT_MLP5_ATTN6
        elif "quant_attn5_mlp6" in run_id:
            return RunType.QUANT_ATTN5_MLP6
        elif "film" in run_id:
            return RunType.FILM
        elif "ttt" in run_id:
            return RunType.TTT
        elif "star_relu" in run_id:
            return RunType.STAR_RELU
        elif "gated_mlp" in run_id:
            return RunType.GATED_MLP
        elif "gqa" in run_id:
            return RunType.GQA
        elif "recurrence" in run_id:
            return RunType.RECURRENCE
        return RunType.XSA

    def evaluate(
        self,
        metrics: dict[str, Any],
        parent_metrics: dict[str, Any] | None = None,
    ) -> Phase2Report:
        """Evaluiere einen Phase 2 Run.

        Args:
            metrics: Run-Metriken
            parent_metrics: Optional Parent-Run-Metriken für Vergleiche

        Returns:
            Phase2Report
        """
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
        challenge_success, challenge_failures = self.criteria.check_challenge_criteria(
            extended_metrics, parent_metrics
        )
        local_success, local_failures = self.criteria.check_local_criteria(extended_metrics)
        should_kill, kill_reason = self.criteria.check_kill_criteria(
            extended_metrics, parent_metrics
        )

        # Status bestimmen
        if should_kill:
            status = "killed"
        elif challenge_success and local_success:
            status = "success"
        elif challenge_failures or local_failures:
            status = "failed"
        else:
            status = "warning"

        # Gate-Status für Phase 3 bestimmen
        gate_status = self._determine_gate_status(status, challenge_success, metrics)

        # Empfehlung generieren
        recommendation = self._generate_recommendation(
            status, gate_status, self.run_type, kill_reason
        )

        # Nächste Schritte
        next_steps = self._generate_next_steps(status, gate_status, self.run_type)

        report = Phase2Report(
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
            gate_status=gate_status,
        )

        return report

    def _determine_gate_status(
        self,
        status: str,
        challenge_success: bool,
        metrics: dict[str, Any],
    ) -> str:
        """Bestimme Gate-Status für Phase 3.

        - PASS: Stabil positiv in ≥2 Runs, klare Verbesserung
        - WATCH: Gemischt oder knapp positiv
        - FAIL: Negativ oder keine klare Verbesserung
        """
        if status == "killed":
            return "fail"

        if status == "success" and challenge_success:
            return "pass"

        if status == "warning":
            return "watch"

        if status == "failed":
            return "fail"

        return "pending"

    def _generate_recommendation(
        self,
        status: str,
        gate_status: str,
        run_type: RunType,
        kill_reason: str | None,
    ) -> str:
        """Generiere Empfehlung basierend auf Ergebnissen."""
        if kill_reason:
            return f"Run stoppen: {kill_reason}"

        if gate_status == "pass":
            return f"Feature ({run_type.value}) für Phase 3 Kombinationen zugelassen"

        if gate_status == "watch":
            return (
                f"Feature ({run_type.value}) benötigt weitere Validierung - "
                "nicht für Phase 3 Kombinationen"
            )

        if gate_status == "fail":
            return f"Feature ({run_type.value}) nicht für Phase 3 empfohlen"

        if status == "success":
            return "Run erfolgreich - Feature zeigt positives Signal"

        if status == "failed":
            return "Run gescheitert - Konfiguration oder Feature überprüfen"

        return "Keine Empfehlung verfügbar"

    def _generate_next_steps(
        self,
        status: str,
        gate_status: str,
        run_type: RunType,
    ) -> list[str]:
        """Generiere nächste Schritte basierend auf Status."""
        steps = []

        if gate_status == "pass":
            steps = [
                f"Feature {run_type.value} für Phase 3 Kombinationen markiert",
                "In Kombination mit anderen PASS-Features testen",
            ]

        elif gate_status == "watch":
            steps = [
                f"Feature {run_type.value} einzeln weiter testen",
                "Nicht mit anderen WATCH-Features kombinieren",
            ]

        elif gate_status == "fail":
            steps = [
                f"Feature {run_type.value} nicht für Phase 3",
                "Alternative Features evaluieren",
            ]

        if status == "failed":
            steps = [
                "Fehlerursache untersuchen",
                "Lokale Proxy-Konfiguration anpassen",
                "Smoke-Test zuerst ausführen",
            ]

        return steps


def create_phase2_evaluator(run_id: str) -> Phase2Evaluator:
    """Convenience function to create a Phase2Evaluator."""
    return Phase2Evaluator(run_id)
