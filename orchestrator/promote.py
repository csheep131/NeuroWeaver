"""Promotion System for advancing runs through stages.

This module implements the stage-based promotion system:
- Stage 1: 1 Seed Screening
- Stage 2: Top-N advance
- Stage 3: 3 Seeds
- Stage 4: Final Packing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.registry import RunRegistry, RunEntry


class Stage(Enum):
    """Competition stages."""

    SCREENING = "screening"  # Stage 1: 1 seed
    FOCUS = "focus"  # Stage 2: Top-N
    FINAL = "final"  # Stage 3: 3 seeds
    SUBMISSION = "submission"  # Stage 4: Final packing


@dataclass
class StageConfig:
    """Configuration for stage progression."""

    # Stage 1 → Stage 2
    screening_top_n: int = 10  # Advance top N from screening
    screening_min_runs: int = 5  # Minimum runs needed for screening

    # Stage 2 → Stage 3
    focus_top_n: int = 5  # Advance top N from focus
    focus_min_improvement: float = 0.0  # Minimum BPB improvement

    # Stage 3 → Stage 4
    final_top_n: int = 3  # Advance top N from final
    final_min_seeds: int = 3  # Minimum seeds for final

    # General
    max_artifact_bytes: int = 16_000_000
    bpb_threshold: float | None = None  # Maximum BPB to advance


@dataclass
class PromotionResult:
    """Result of a promotion decision."""

    run_id: str
    from_stage: Stage
    to_stage: Stage | None  # None = not promoted
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def was_promoted(self) -> bool:
        """Check if run was promoted."""
        return self.to_stage is not None and self.to_stage != self.from_stage


class PromotionSystem:
    """System for managing run promotions through stages."""

    def __init__(
        self,
        registry: RunRegistry | None = None,
        config: StageConfig | None = None,
    ):
        self.registry = registry or RunRegistry()
        self.config = config or StageConfig()
        self.results: list[PromotionResult] = []
        self._runs_cache: list[RunEntry] | None = None
        self._stage_cache: dict[str, Stage] = {}
        self._runs_by_stage_cache: dict[Stage, list[RunEntry]] = {}

    def _refresh_cache(self) -> None:
        """Refresh cached runs and stage mappings."""
        self._runs_cache = None
        self._stage_cache.clear()
        self._runs_by_stage_cache.clear()

    def _get_all_runs(self) -> list[RunEntry]:
        """Get all completed runs with caching."""
        if self._runs_cache is None:
            self._runs_cache = self.registry.list_runs(status="completed")
        return self._runs_cache

    def get_stage(self, run_id: str) -> Stage | None:
        """Get current stage of a run."""
        if run_id in self._stage_cache:
            return self._stage_cache[run_id]

        entry = self.registry.get(run_id)
        if not entry:
            return None

        # Determine stage from run metadata or tags
        if "stage_final" in entry.tags:
            stage = Stage.FINAL
        elif "stage_focus" in entry.tags:
            stage = Stage.FOCUS
        elif "stage_submission" in entry.tags:
            stage = Stage.SUBMISSION
        else:
            stage = Stage.SCREENING

        self._stage_cache[run_id] = stage
        return stage

    def evaluate_screening(self) -> list[PromotionResult]:
        """Evaluate Stage 1 (Screening) runs for promotion.

        Returns list of PromotionResult for all screening runs.
        """
        results = []
        screening_runs = self._get_runs_by_stage(Stage.SCREENING)

        if len(screening_runs) < self.config.screening_min_runs:
            # Not enough runs for meaningful screening
            for run in screening_runs:
                results.append(PromotionResult(
                    run_id=run.run_id,
                    from_stage=Stage.SCREENING,
                    to_stage=None,
                    reason=f"Need at least {self.config.screening_min_runs} runs for screening",
                ))
            return results

        # Sort by BPB (lower is better)
        valid_runs = [r for r in screening_runs if r.val_bpb is not None]
        valid_runs.sort(key=lambda r: r.val_bpb or float("inf"))

        # Filter by artifact size
        valid_runs = [
            r for r in valid_runs
            if r.artifact_bytes <= self.config.max_artifact_bytes
        ]

        # Apply BPB threshold if set
        if self.config.bpb_threshold:
            valid_runs = [
                r for r in valid_runs
                if (r.val_bpb or float("inf")) <= self.config.bpb_threshold
            ]

        # Select top N
        top_n = valid_runs[: self.config.screening_top_n]

        for run in screening_runs:
            if run in top_n:
                results.append(PromotionResult(
                    run_id=run.run_id,
                    from_stage=Stage.SCREENING,
                    to_stage=Stage.FOCUS,
                    reason=f"Top {self.config.screening_top_n} by BPB",
                    metrics={"val_bpb": run.val_bpb, "rank": valid_runs.index(run) + 1},
                ))
            else:
                results.append(PromotionResult(
                    run_id=run.run_id,
                    from_stage=Stage.SCREENING,
                    to_stage=None,
                    reason="Not in top N",
                    metrics={"val_bpb": run.val_bpb},
                ))

        self.results.extend(results)
        return results

    def evaluate_focus(self) -> list[PromotionResult]:
        """Evaluate Stage 2 (Focus) runs for promotion.

        Returns list of PromotionResult for all focus runs.
        """
        results = []
        focus_runs = self._get_runs_by_stage(Stage.FOCUS)

        # Sort by BPB
        valid_runs = [r for r in focus_runs if r.val_bpb is not None]
        valid_runs.sort(key=lambda r: r.val_bpb or float("inf"))

        # Filter by artifact size
        valid_runs = [
            r for r in valid_runs
            if r.artifact_bytes <= self.config.max_artifact_bytes
        ]

        # Check minimum improvement over baseline
        if len(valid_runs) >= 2:
            baseline_bpb = valid_runs[-1].val_bpb  # Worst is baseline
            valid_runs = [
                r for r in valid_runs
                if (baseline_bpb - (r.val_bpb or 0)) >= self.config.focus_min_improvement
            ]

        # Select top N
        top_n = valid_runs[: self.config.focus_top_n]

        for run in focus_runs:
            if run in top_n:
                results.append(PromotionResult(
                    run_id=run.run_id,
                    from_stage=Stage.FOCUS,
                    to_stage=Stage.FINAL,
                    reason=f"Top {self.config.focus_top_n} from focus",
                    metrics={"val_bpb": run.val_bpb, "rank": valid_runs.index(run) + 1},
                ))
            else:
                results.append(PromotionResult(
                    run_id=run.run_id,
                    from_stage=Stage.FOCUS,
                    to_stage=None,
                    reason="Not in top N or insufficient improvement",
                    metrics={"val_bpb": run.val_bpb},
                ))

        self.results.extend(results)
        return results

    def evaluate_final(self) -> list[PromotionResult]:
        """Evaluate Stage 3 (Final) runs for promotion to submission.

        Returns list of PromotionResult for all final runs.
        """
        results = []
        final_runs = self._get_runs_by_stage(Stage.FINAL)

        # Group by config hash to check seed count
        config_groups: dict[str, list[RunEntry]] = {}
        for run in final_runs:
            if run.config_hash not in config_groups:
                config_groups[run.config_hash] = []
            config_groups[run.config_hash].append(run)

        # Filter configs with enough seeds
        valid_configs: list[RunEntry] = []
        for config_hash, runs in config_groups.items():
            if len(runs) >= self.config.final_min_seeds:
                # Use best run from this config
                best = min(runs, key=lambda r: r.val_bpb or float("inf"))
                valid_configs.append(best)

        # Sort by BPB
        valid_configs.sort(key=lambda r: r.val_bpb or float("inf"))

        # Select top N
        top_n = valid_configs[: self.config.final_top_n]

        for run in final_runs:
            if run in top_n:
                results.append(PromotionResult(
                    run_id=run.run_id,
                    from_stage=Stage.FINAL,
                    to_stage=Stage.SUBMISSION,
                    reason=f"Top {self.config.final_top_n} from final",
                    metrics={"val_bpb": run.val_bpb},
                ))
            else:
                results.append(PromotionResult(
                    run_id=run.run_id,
                    from_stage=Stage.FINAL,
                    to_stage=None,
                    reason="Not in top N or insufficient seeds",
                    metrics={"val_bpb": run.val_bpb},
                ))

        self.results.extend(results)
        return results

    def promote(self, run_id: str, to_stage: Stage) -> bool:
        """Manually promote a run to a stage."""
        entry = self.registry.get(run_id)
        if not entry:
            return False

        # Update tags
        tag_map = {
            Stage.SCREENING: "stage_screening",
            Stage.FOCUS: "stage_focus",
            Stage.FINAL: "stage_final",
            Stage.SUBMISSION: "stage_submission",
        }

        # Remove old stage tags
        for tag in tag_map.values():
            if tag in entry.tags:
                entry.tags.remove(tag)

        # Add new stage tag
        entry.tags.append(tag_map[to_stage])

        self.registry.update(run_id, tags=entry.tags)
        return True

    def get_promotion_report(self) -> dict[str, Any]:
        """Get comprehensive promotion report."""
        return {
            "screening": {
                "count": len(self._get_runs_by_stage(Stage.SCREENING)),
                "promoted": len([r for r in self.results if r.from_stage == Stage.SCREENING and r.was_promoted()]),
            },
            "focus": {
                "count": len(self._get_runs_by_stage(Stage.FOCUS)),
                "promoted": len([r for r in self.results if r.from_stage == Stage.FOCUS and r.was_promoted()]),
            },
            "final": {
                "count": len(self._get_runs_by_stage(Stage.FINAL)),
                "promoted": len([r for r in self.results if r.from_stage == Stage.FINAL and r.was_promoted()]),
            },
            "submission": {
                "count": len(self._get_runs_by_stage(Stage.SUBMISSION)),
            },
            "all_results": [
                {
                    "run_id": r.run_id,
                    "from": r.from_stage.value,
                    "to": r.to_stage.value if r.to_stage else None,
                    "reason": r.reason,
                }
                for r in self.results
            ],
        }

    def _get_runs_by_stage(self, stage: Stage) -> list[RunEntry]:
        """Get all runs at a specific stage with caching."""
        # Check cache first
        if stage in self._runs_by_stage_cache:
            return self._runs_by_stage_cache[stage]
        
        tag_map = {
            Stage.SCREENING: "stage_screening",
            Stage.FOCUS: "stage_focus",
            Stage.FINAL: "stage_final",
            Stage.SUBMISSION: "stage_submission",
        }

        all_runs = self._get_all_runs()

        if stage == Stage.SCREENING:
            # Screening runs have no stage tag or explicit screening tag
            runs = [
                r for r in all_runs
                if tag_map[Stage.SCREENING] in r.tags
                or not any(tag_map[s] in r.tags for s in Stage if s != Stage.SCREENING)
            ]
        else:
            runs = [r for r in all_runs if tag_map[stage] in r.tags]
        
        # Cache the result
        self._runs_by_stage_cache[stage] = runs
        return runs

    def print_summary(self) -> str:
        """Print promotion summary."""
        report = self.get_promotion_report()

        lines = [
            "=" * 60,
            "PROMOTION SYSTEM STATUS",
            "=" * 60,
            "",
            f"Stage 1 (Screening): {report['screening']['count']} runs, "
            f"{report['screening']['promoted']} promoted",
            f"Stage 2 (Focus):     {report['focus']['count']} runs, "
            f"{report['focus']['promoted']} promoted",
            f"Stage 3 (Final):     {report['final']['count']} runs, "
            f"{report['final']['promoted']} promoted",
            f"Stage 4 (Submission): {report['submission']['count']} runs",
            "",
        ]

        # Show recent promotions
        recent = [r for r in self.results if r.was_promoted()]
        if recent:
            lines.append("RECENT PROMOTIONS:")
            for r in recent[-10:]:
                lines.append(f"  {r.run_id}: {r.from_stage.value} → {r.to_stage.value}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def create_promotion_system(
    registry: RunRegistry | None = None,
    **kwargs: Any,
) -> PromotionSystem:
    """Create a promotion system."""
    config = StageConfig(**kwargs) if kwargs else None
    return PromotionSystem(registry, config)
