"""Run comparison and reporting utilities."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunComparison:
    """Comparison result between runs."""

    runs: list[dict[str, Any]] = field(default_factory=list)
    best_bpb: dict[str, Any] | None = None
    best_bpb_per_mb: dict[str, Any] | None = None
    best_bpb_per_ms: dict[str, Any] | None = None
    finalists: list[dict[str, Any]] = field(default_factory=list)
    killed: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "runs": self.runs,
            "best_bpb": self.best_bpb,
            "best_bpb_per_mb": self.best_bpb_per_mb,
            "best_bpb_per_ms": self.best_bpb_per_ms,
            "finalists": self.finalists,
            "killed": self.killed,
        }

    def save(self, path: str | Path) -> None:
        """Save comparison to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class RunComparator:
    """Comparator for run results."""

    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)

    def load_run_metrics(self, run_id: str) -> dict[str, Any] | None:
        """Load metrics for a single run."""
        metrics_path = self.results_dir / run_id / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                return json.load(f)

        # Try registry
        registry_path = self.results_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)
                if run_id in registry:
                    return registry[run_id]

        return None

    def compare_runs(self, run_ids: list[str] | None = None) -> RunComparison:
        """Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare. If None, load all runs from registry.

        Returns:
            RunComparison with analysis
        """
        # Load all runs
        if run_ids is None:
            run_ids = self._get_all_run_ids()

        runs_data = []
        for run_id in run_ids:
            metrics = self.load_run_metrics(run_id)
            if metrics:
                metrics["run_id"] = run_id
                runs_data.append(metrics)

        comparison = RunComparison(runs=runs_data)

        # Find best by different criteria
        if runs_data:
            comparison.best_bpb = self._find_best(runs_data, "val_bpb", lower_is_better=True)
            comparison.best_bpb_per_mb = self._find_best_bpb_per_mb(runs_data)
            comparison.best_bpb_per_ms = self._find_best_bpb_per_ms(runs_data)

            # Categorize runs
            comparison.finalists = self._find_finalists(runs_data)
            comparison.killed = self._find_killed(runs_data)

        return comparison

    def _find_best(
        self,
        runs: list[dict[str, Any]],
        metric: str,
        lower_is_better: bool = True,
    ) -> dict[str, Any] | None:
        """Find the best run by a specific metric."""
        valid_runs = [r for r in runs if metric in r and r[metric] is not None]
        if not valid_runs:
            return None

        if lower_is_better:
            return min(valid_runs, key=lambda r: r[metric])
        else:
            return max(valid_runs, key=lambda r: r[metric])

    def _find_best_bpb_per_mb(self, runs: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Find the best run by BPB per MB."""
        valid_runs = []
        for run in runs:
            if "val_bpb" in run and run["val_bpb"] is not None:
                artifact_mb = run.get("artifact_bytes", 0) / (1024 * 1024)
                if artifact_mb > 0:
                    run_copy = run.copy()
                    run_copy["bpb_per_mb"] = run_copy["val_bpb"] / artifact_mb
                    valid_runs.append(run_copy)

        if not valid_runs:
            return None

        return min(valid_runs, key=lambda r: r["bpb_per_mb"])

    def _find_best_bpb_per_ms(self, runs: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Find the best run by BPB per ms/step."""
        valid_runs = []
        for run in runs:
            if "val_bpb" in run and run["val_bpb"] is not None:
                ms_per_step = run.get("ms_per_step")
                if ms_per_step and ms_per_step > 0:
                    run_copy = run.copy()
                    run_copy["bpb_per_ms"] = run_copy["val_bpb"] / run_copy["ms_per_step"]
                    valid_runs.append(run_copy)

        if not valid_runs:
            return None

        return min(valid_runs, key=lambda r: r["bpb_per_ms"])

    def _find_finalists(
        self,
        runs: list[dict[str, Any]],
        max_artifact_bytes: int = 16_000_000,
    ) -> list[dict[str, Any]]:
        """Find runs that qualify as finalists."""
        finalists = []
        for run in runs:
            # Check artifact size
            if run.get("artifact_bytes", 0) > max_artifact_bytes:
                continue

            # Check status
            if run.get("status") != "completed":
                continue

            # Must have valid BPB
            if "val_bpb" not in run or run["val_bpb"] is None:
                continue

            finalists.append(run)

        # Sort by BPB (lower is better)
        return sorted(finalists, key=lambda r: r.get("val_bpb", float("inf")))

    def _find_killed(self, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find runs that were killed."""
        return [r for r in runs if r.get("status") in ("killed", "failed")]

    def _get_all_run_ids(self) -> list[str]:
        """Get all run IDs from registry."""
        registry_path = self.results_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)
                return list(registry.keys())
        return []

    def generate_leaderboard(self, top_k: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Generate a leaderboard of runs.

        Args:
            top_k: Number of top runs to include in each category

        Returns:
            Dictionary with leaderboards for different categories
        """
        comparison = self.compare_runs()

        # Sort finalists by different criteria
        finalists = comparison.finalists

        leaderboard = {
            "top_by_bpb": sorted(
                finalists,
                key=lambda r: r.get("val_bpb", float("inf")),
            )[:top_k],
            "top_by_bpb_per_mb": sorted(
                finalists,
                key=lambda r: r.get("val_bpb", float("inf")) / max(r.get("artifact_bytes", 1), 1),
            )[:top_k],
            "top_by_speed": sorted(
                finalists,
                key=lambda r: r.get("ms_per_step", float("inf")),
            )[:top_k],
        }

        return leaderboard

    def print_summary(self, comparison: RunComparison) -> str:
        """Print a human-readable summary."""
        lines = ["=" * 60, "RUN COMPARISON SUMMARY", "=" * 60, ""]

        lines.append(f"Total runs analyzed: {len(comparison.runs)}")
        lines.append(f"Finalists: {len(comparison.finalists)}")
        lines.append(f"Killed/Failed: {len(comparison.killed)}")
        lines.append("")

        if comparison.best_bpb:
            lines.append(f"Best BPB: {comparison.best_bpb.get('run_id')} "
                        f"({comparison.best_bpb.get('val_bpb', 'N/A')})")

        if comparison.best_bpb_per_mb:
            lines.append(f"Best BPB/MB: {comparison.best_bpb_per_mb.get('run_id')}")

        if comparison.best_bpb_per_ms:
            lines.append(f"Best BPB/ms: {comparison.best_bpb_per_ms.get('run_id')}")

        lines.append("")
        lines.append("Top 5 Finalists by BPB:")
        for i, run in enumerate(comparison.finalists[:5], 1):
            lines.append(f"  {i}. {run.get('run_id')}: BPB={run.get('val_bpb', 'N/A')}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
