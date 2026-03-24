"""Ablation Reporter with kill rules.

This module provides automated ablation analysis and kill rules
for systematically pruning underperforming configurations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from core.registry import RunRegistry, RunEntry


class KillReason(Enum):
    """Reasons for killing a run."""

    ARTIFACT_TOO_LARGE = "artifact_too_large"
    SLOW_WITHOUT_GAIN = "slow_without_gain"
    QUANT_GAP_TOO_LARGE = "quant_gap_too_large"
    VOLATILE_ACROSS_SEEDS = "volatile_across_seeds"
    DEBUGGING_IMPOSSIBLE = "debugging_impossible"
    BPB_REGRESSION = "bpb_regression"
    CUSTOM = "custom"


@dataclass
class KillRule:
    """Definition of a kill rule."""

    name: str
    reason: KillReason
    condition: Callable[[dict[str, Any]], tuple[bool, str]]
    enabled: bool = True
    priority: int = 1  # Lower = higher priority

    def check(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        """Check if rule is triggered."""
        if not self.enabled:
            return False, ""
        return self.condition(metrics)


@dataclass
class AblationReport:
    """Report on ablation study results."""

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_runs: int = 0
    active_runs: int = 0
    killed_runs: int = 0
    failed_runs: int = 0

    # Best results
    best_bpb: dict[str, Any] | None = None
    best_bpb_per_mb: dict[str, Any] | None = None
    best_bpb_per_ms: dict[str, Any] | None = None

    # Kill statistics
    kill_reasons: dict[str, int] = field(default_factory=dict)

    # Recommendations
    finalists: list[dict[str, Any]] = field(default_factory=list)
    to_rerun: list[dict[str, Any]] = field(default_factory=list)
    to_kill: list[dict[str, Any]] = field(default_factory=list)

    # Lineage summary
    lineages: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at,
            "total_runs": self.total_runs,
            "active_runs": self.active_runs,
            "killed_runs": self.killed_runs,
            "failed_runs": self.failed_runs,
            "best_bpb": self.best_bpb,
            "best_bpb_per_mb": self.best_bpb_per_mb,
            "best_bpb_per_ms": self.best_bpb_per_ms,
            "kill_reasons": self.kill_reasons,
            "finalists": self.finalists,
            "to_rerun": self.to_rerun,
            "to_kill": self.to_kill,
            "lineages": self.lineages,
        }

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> str:
        """Print human-readable summary."""
        lines = [
            "=" * 70,
            "ABLATION REPORT",
            f"Generated: {self.generated_at}",
            "=" * 70,
            "",
            "RUN STATISTICS",
            f"  Total runs:     {self.total_runs}",
            f"  Active:         {self.active_runs}",
            f"  Killed:         {self.killed_runs}",
            f"  Failed:         {self.failed_runs}",
            "",
        ]

        if self.best_bpb:
            lines.append("BEST RESULTS")
            lines.append(f"  Best BPB:       {self.best_bpb.get('run_id')} "
                        f"({self.best_bpb.get('val_bpb', 'N/A'):.4f})")
            if self.best_bpb_per_mb:
                lines.append(f"  Best BPB/MB:    {self.best_bpb_per_mb.get('run_id')}")
            if self.best_bpb_per_ms:
                lines.append(f"  Best BPB/ms:    {self.best_bpb_per_ms.get('run_id')}")
            lines.append("")

        if self.kill_reasons:
            lines.append("KILL REASONS")
            for reason, count in self.kill_reasons.items():
                lines.append(f"  {reason}: {count}")
            lines.append("")

        if self.finalists:
            lines.append("FINALISTS (Top 5)")
            for i, run in enumerate(self.finalists[:5], 1):
                lines.append(f"  {i}. {run.get('run_id')}: "
                           f"BPB={run.get('val_bpb', 'N/A'):.4f}, "
                           f"Size={run.get('artifact_bytes', 0) / 1e6:.2f}MB")
            lines.append("")

        if self.to_kill:
            lines.append("RECOMMENDED TO KILL")
            for run in self.to_kill[:5]:
                lines.append(f"  - {run.get('run_id')}: {run.get('kill_reason', 'N/A')}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


class AblationReporter:
    """Reporter for ablation studies.

    Implements kill rules from the blueprint:
    1. Artifact > 16,000,000 bytes
    2. ms/step deutlich schlechter ohne klaren BPB-Gewinn
    3. Quant-Gap untragbar
    4. Feature volatil über Seeds
    5. Kombi macht Debugging unmöglich
    """

    # Constants
    MAX_ARTIFACT_BYTES = 16_000_000
    MS_THRESHOLD_INCREASE = 2.0  # ms
    BPB_MIN_GAIN = 0.05  # Minimum BPB improvement to justify slowdown
    MAX_QUANT_GAP = 0.1  # Maximum acceptable BPB degradation from quantization
    MIN_SEED_VOLATILITY = 0.02  # Minimum BPB std dev to consider volatile

    def __init__(self, registry: RunRegistry | None = None):
        self.registry = registry or RunRegistry()
        self.rules = self._create_default_rules()

    def _create_default_rules(self) -> list[KillRule]:
        """Create default kill rules."""

        def check_quant_gap(m: dict[str, Any]) -> tuple[bool, str]:
            """Check quantization gap rule."""
            quant_bpb = m.get("quantized_val_bpb")
            val_bpb = m.get("val_bpb")
            if quant_bpb is not None and val_bpb is not None:
                gap = quant_bpb - val_bpb
                if gap > self.MAX_QUANT_GAP:
                    return True, f"Quantization gap {gap:.4f} exceeds threshold {self.MAX_QUANT_GAP}"
            return False, ""

        def check_slow_without_gain(m: dict[str, Any]) -> tuple[bool, str]:
            """Check slow without BPB gain rule."""
            delta_ms = m.get("delta_ms")
            delta_bpb = m.get("delta_bpb")
            if delta_ms is not None and delta_bpb is not None:
                if delta_ms > self.MS_THRESHOLD_INCREASE and delta_bpb > -self.BPB_MIN_GAIN:
                    return True, (
                        f"Step time increased by {delta_ms:.2f}ms "
                        f"without sufficient BPB gain (delta_bpb={delta_bpb:.4f})"
                    )
            return False, ""

        def check_bpb_regression(m: dict[str, Any]) -> tuple[bool, str]:
            """Check BPB regression rule."""
            delta_bpb = m.get("delta_bpb")
            if delta_bpb is not None and delta_bpb > 0.1:
                return True, f"BPB regression: delta_bpb={delta_bpb:.4f} (worse than parent)"
            return False, ""

        rules = [
            # Kill 1: Artifact > 16,000,000 bytes
            KillRule(
                name="artifact_size_limit",
                reason=KillReason.ARTIFACT_TOO_LARGE,
                condition=lambda m: (
                    m.get("artifact_bytes", 0) > self.MAX_ARTIFACT_BYTES,
                    f"Artifact size {m.get('artifact_bytes', 0):,} bytes exceeds "
                    f"{self.MAX_ARTIFACT_BYTES:,} byte limit",
                ),
                priority=1,
            ),

            # Kill 2: Slow without BPB gain
            KillRule(
                name="slow_without_gain",
                reason=KillReason.SLOW_WITHOUT_GAIN,
                condition=check_slow_without_gain,
                priority=2,
            ),

            # Kill 3: Quant gap too large
            KillRule(
                name="quant_gap",
                reason=KillReason.QUANT_GAP_TOO_LARGE,
                condition=check_quant_gap,
                priority=3,
            ),

            # Kill 4: BPB regression (worse than parent)
            KillRule(
                name="bpb_regression",
                reason=KillReason.BPB_REGRESSION,
                condition=check_bpb_regression,
                priority=4,
            ),
        ]

        return rules

    def add_rule(self, rule: KillRule) -> None:
        """Add a custom kill rule."""
        self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a kill rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(i)
                return True
        return False

    def evaluate_run(
        self,
        run_id: str,
        metrics: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        """Evaluate a run against all kill rules.

        Returns:
            Tuple of (should_kill, reason)
        """
        if metrics is None:
            entry = self.registry.get(run_id)
            if entry is None:
                return False, None
            metrics = entry.to_dict()

        for rule in sorted(self.rules, key=lambda r: r.priority):
            triggered, reason = rule.check(metrics)
            if triggered:
                return True, reason

        return False, None

    def evaluate_all_runs(self) -> dict[str, tuple[bool, str | None]]:
        """Evaluate all runs in registry.

        Returns dict of run_id -> (should_kill, reason).
        """
        results = {}
        for entry in self.registry.list_runs():
            should_kill, reason = self.evaluate_run(entry.run_id)
            results[entry.run_id] = (should_kill, reason)
        return results

    def apply_kills(self) -> list[str]:
        """Apply kill rules to all runs and update registry.

        Returns list of killed run IDs.
        """
        killed = []
        evaluations = self.evaluate_all_runs()

        for run_id, (should_kill, reason) in evaluations.items():
            if should_kill:
                self.registry.kill_run(run_id, reason or "Unknown reason")
                killed.append(run_id)

        return killed

    def generate_report(self) -> AblationReport:
        """Generate comprehensive ablation report."""
        report = AblationReport()

        # Get all runs
        all_runs = self.registry.list_runs()
        report.total_runs = len(all_runs)

        # Categorize runs
        active = []
        killed = []
        failed = []

        for entry in all_runs:
            if entry.status == "completed":
                active.append(entry)
            elif entry.status == "killed":
                killed.append(entry)
            elif entry.status == "failed":
                failed.append(entry)

        report.active_runs = len(active)
        report.killed_runs = len(killed)
        report.failed_runs = len(failed)

        # Count kill reasons
        kill_reasons: dict[str, int] = {}
        for entry in killed:
            reason = entry.notes or "unknown"
            # Extract kill reason type
            for kr in KillReason:
                if kr.value in reason.lower():
                    reason = kr.value
                    break
            kill_reasons[reason] = kill_reasons.get(reason, 0) + 1
        report.kill_reasons = kill_reasons

        # Find best runs
        if active:
            # Best BPB
            by_bpb = sorted(
                [e for e in active if e.val_bpb is not None],
                key=lambda e: e.val_bpb or float("inf"),
            )
            if by_bpb:
                report.best_bpb = by_bpb[0].to_dict()

            # Best BPB per MB
            by_efficiency = sorted(
                [e for e in active if e.val_bpb is not None and e.artifact_bytes > 0],
                key=lambda e: (e.val_bpb or float("inf")) / (e.artifact_bytes / 1e6),
            )
            if by_efficiency:
                report.best_bpb_per_mb = by_efficiency[0].to_dict()

            # Best BPB per ms
            by_speed = sorted(
                [e for e in active if e.val_bpb is not None and e.ms_per_step],
                key=lambda e: (e.val_bpb or float("inf")) / (e.ms_per_step or float("inf")),
            )
            if by_speed:
                report.best_bpb_per_ms = by_speed[0].to_dict()

            # Finalists (completed, within limits, sorted by BPB)
            finalists = [
                e.to_dict() for e in by_bpb
                if e.artifact_bytes <= self.MAX_ARTIFACT_BYTES
            ]
            report.finalists = finalists[:10]

        # Runs to kill
        evaluations = self.evaluate_all_runs()
        to_kill = []
        for run_id, (should_kill, reason) in evaluations.items():
            if should_kill:
                entry = self.registry.get(run_id)
                if entry:
                    to_kill.append({
                        "run_id": run_id,
                        "kill_reason": reason,
                    })
        report.to_kill = to_kill

        # Lineage summary
        lineages: dict[str, list[str]] = {}
        for entry in all_runs:
            if entry.parent_run_id:
                if entry.parent_run_id not in lineages:
                    lineages[entry.parent_run_id] = []
                lineages[entry.parent_run_id].append(entry.run_id)
        report.lineages = lineages

        # Recommend re-runs for promising runs without multiple seeds
        report.to_rerun = self._find_runs_to_rerun(active)

        return report

    def _find_runs_to_rerun(
        self,
        active: list[RunEntry],
        min_seeds: int = 3,
    ) -> list[dict[str, Any]]:
        """Find runs that should be re-run with multiple seeds.

        Criteria:
        - Good BPB (top 20%)
        - Less than min_seeds runs with same config hash
        """
        if not active:
            return []

        # Group by config hash
        by_config: dict[str, list[RunEntry]] = {}
        for entry in active:
            config_hash = entry.config_hash
            if config_hash not in by_config:
                by_config[config_hash] = []
            by_config[config_hash].append(entry)

        # Find top 20% by BPB
        by_bpb = sorted(
            [e for e in active if e.val_bpb is not None],
            key=lambda e: e.val_bpb or float("inf"),
        )
        top_20_pct = max(1, len(by_bpb) // 5)
        top_configs = {e.config_hash for e in by_bpb[:top_20_pct]}

        # Find configs that need more seeds
        to_rerun = []
        for config_hash in top_configs:
            entries = by_config.get(config_hash, [])
            if len(entries) < min_seeds:
                # Get best run from this config
                best = min(entries, key=lambda e: e.val_bpb or float("inf"))
                to_rerun.append({
                    "run_id": best.run_id,
                    "config_hash": config_hash,
                    "current_seeds": len(entries),
                    "val_bpb": best.val_bpb,
                    "recommendation": f"Re-run with {min_seeds - len(entries)} more seeds",
                })

        return to_rerun

    def print_kill_report(self, evaluations: dict[str, tuple[bool, str | None]]) -> str:
        """Print kill report."""
        lines = ["=" * 60, "KILL RULE EVALUATION", "=" * 60, ""]

        to_kill = [(rid, reason) for rid, (kill, reason) in evaluations.items() if kill]
        safe = [rid for rid, (kill, _) in evaluations.items() if not kill]

        lines.append(f"Runs to kill: {len(to_kill)}")
        lines.append(f"Safe runs: {len(safe)}")
        lines.append("")

        if to_kill:
            lines.append("RUNS TO KILL:")
            for run_id, reason in to_kill:
                lines.append(f"  - {run_id}")
                lines.append(f"    Reason: {reason}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def create_ablation_reporter(registry: RunRegistry | None = None) -> AblationReporter:
    """Create an ablation reporter."""
    return AblationReporter(registry)
