"""Run registry for tracking experiments."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunEntry:
    """Entry for a single run in the registry."""

    run_id: str
    config_hash: str
    git_commit: str | None = None
    parent_run_id: str | None = None
    status: str = "pending"  # pending, running, completed, failed, killed
    start_time: str | None = None
    end_time: str | None = None
    seed: int = 42

    # Metrics
    val_bpb: float | None = None
    ms_per_step: float | None = None
    steps_completed: int = 0
    artifact_bytes: int = 0
    quantized_val_bpb: float | None = None

    # Comparison
    delta_bpb: float | None = None
    delta_ms: float | None = None

    # Metadata
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "parent_run_id": self.parent_run_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "seed": self.seed,
            "val_bpb": self.val_bpb,
            "ms_per_step": self.ms_per_step,
            "steps_completed": self.steps_completed,
            "artifact_bytes": self.artifact_bytes,
            "quantized_val_bpb": self.quantized_val_bpb,
            "delta_bpb": self.delta_bpb,
            "delta_ms": self.delta_ms,
            "notes": self.notes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunEntry":
        """Create RunEntry from dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            config_hash=data.get("config_hash", ""),
            git_commit=data.get("git_commit"),
            parent_run_id=data.get("parent_run_id"),
            status=data.get("status", "pending"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            seed=data.get("seed", 42),
            val_bpb=data.get("val_bpb"),
            ms_per_step=data.get("ms_per_step"),
            steps_completed=data.get("steps_completed", 0),
            artifact_bytes=data.get("artifact_bytes", 0),
            quantized_val_bpb=data.get("quantized_val_bpb"),
            delta_bpb=data.get("delta_bpb"),
            delta_ms=data.get("delta_ms"),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
        )


class RunRegistry:
    """Registry for tracking all runs."""

    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)
        self.registry_path = self.results_dir / "registry.json"
        self.entries: dict[str, RunEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                for run_id, entry_data in data.items():
                    self.entries[run_id] = RunEntry.from_dict(entry_data)

    def _save(self) -> None:
        """Save registry to disk."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        data = {run_id: entry.to_dict() for run_id, entry in self.entries.items()}
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(self, run_id: str, config_hash: str, parent_run_id: str | None = None, seed: int = 42) -> RunEntry:
        """Register a new run."""
        entry = RunEntry(
            run_id=run_id,
            config_hash=config_hash,
            parent_run_id=parent_run_id,
            seed=seed,
            status="pending",
        )
        self.entries[run_id] = entry
        self._save()
        return entry

    def get(self, run_id: str) -> RunEntry | None:
        """Get a run entry by ID."""
        return self.entries.get(run_id)

    def update(self, run_id: str, **kwargs: Any) -> RunEntry | None:
        """Update a run entry."""
        entry = self.entries.get(run_id)
        if entry is None:
            return None

        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        self._save()
        return entry

    def start_run(self, run_id: str, git_commit: str | None = None) -> RunEntry | None:
        """Mark a run as started."""
        return self.update(
            run_id,
            status="running",
            start_time=datetime.now().isoformat(),
            git_commit=git_commit,
        )

    def complete_run(self, run_id: str, metrics: dict[str, Any]) -> RunEntry | None:
        """Mark a run as completed with metrics."""
        entry = self.get(run_id)
        if entry is None:
            return None

        # Compute deltas if parent exists
        delta_bpb = None
        delta_ms = None
        if entry.parent_run_id:
            parent = self.get(entry.parent_run_id)
            if parent:
                if parent.val_bpb is not None and metrics.get("val_bpb") is not None:
                    delta_bpb = metrics["val_bpb"] - parent.val_bpb
                if parent.ms_per_step is not None and metrics.get("ms_per_step") is not None:
                    delta_ms = metrics["ms_per_step"] - parent.ms_per_step

        return self.update(
            run_id,
            status="completed",
            end_time=datetime.now().isoformat(),
            val_bpb=metrics.get("val_bpb"),
            ms_per_step=metrics.get("ms_per_step"),
            steps_completed=metrics.get("steps_completed", 0),
            artifact_bytes=metrics.get("artifact_bytes", 0),
            quantized_val_bpb=metrics.get("quantized_val_bpb"),
            delta_bpb=delta_bpb,
            delta_ms=delta_ms,
        )

    def fail_run(self, run_id: str, notes: str = "") -> RunEntry | None:
        """Mark a run as failed."""
        return self.update(
            run_id,
            status="failed",
            end_time=datetime.now().isoformat(),
            notes=notes,
        )

    def kill_run(self, run_id: str, reason: str = "") -> RunEntry | None:
        """Mark a run as killed (e.g., due to kill rules)."""
        return self.update(
            run_id,
            status="killed",
            end_time=datetime.now().isoformat(),
            notes=reason,
        )

    def list_runs(self, status: str | None = None) -> list[RunEntry]:
        """List all runs, optionally filtered by status."""
        runs = list(self.entries.values())
        if status:
            runs = [r for r in runs if r.status == status]
        return sorted(runs, key=lambda r: r.run_id)

    def get_lineage(self, run_id: str) -> list[RunEntry]:
        """Get the lineage of a run (all ancestors)."""
        lineage = []
        current = self.get(run_id)
        while current and current.parent_run_id:
            parent = self.get(current.parent_run_id)
            if parent:
                lineage.append(parent)
                current = parent
            else:
                break
        return list(reversed(lineage))

    def get_children(self, parent_run_id: str) -> list[RunEntry]:
        """Get all runs that have this run as parent."""
        return [e for e in self.entries.values() if e.parent_run_id == parent_run_id]

    def get_lineage_tree(self, run_id: str) -> dict[str, Any]:
        """Get the full lineage tree of a run.

        Returns a nested dictionary representing the tree.
        """
        entry = self.get(run_id)
        if entry is None:
            return {}

        # Find root ancestor
        root = entry
        while root.parent_run_id:
            parent = self.get(root.parent_run_id)
            if parent:
                root = parent
            else:
                break

        # Build tree from root
        def build_tree(e: RunEntry) -> dict[str, Any]:
            children = self.get_children(e.run_id)
            return {
                "run_id": e.run_id,
                "status": e.status,
                "val_bpb": e.val_bpb,
                "config_hash": e.config_hash,
                "children": [build_tree(c) for c in sorted(children, key=lambda x: x.run_id)],
            }

        return build_tree(root)

    def get_all_lineages(self) -> dict[str, list[str]]:
        """Get all lineages as a dictionary.

        Returns dict mapping parent_run_id -> list of child run_ids.
        """
        lineages: dict[str, list[str]] = {}
        for entry in self.entries.values():
            if entry.parent_run_id:
                if entry.parent_run_id not in lineages:
                    lineages[entry.parent_run_id] = []
                lineages[entry.parent_run_id].append(entry.run_id)
        return lineages

    def get_run_family(self, run_id: str) -> list[RunEntry]:
        """Get all runs in the same family (same root ancestor)."""
        entry = self.get(run_id)
        if entry is None:
            return []

        # Find root
        root = entry
        while root.parent_run_id:
            parent = self.get(root.parent_run_id)
            if parent:
                root = parent
            else:
                break

        # BFS to find all descendants
        family = []
        queue = [root]
        while queue:
            current = queue.pop(0)
            family.append(current)
            queue.extend(self.get_children(current.run_id))

        return family

    def get_config_family(self, config_hash: str) -> list[RunEntry]:
        """Get all runs with the same config hash (different seeds)."""
        return [e for e in self.entries.values() if e.config_hash == config_hash]

    def get_seed_statistics(self, config_hash: str) -> dict[str, Any]:
        """Get statistics across seeds for a config.

        Returns mean, std, min, max for BPB and other metrics.
        """
        entries = self.get_config_family(config_hash)
        if not entries:
            return {}

        # Collect metrics
        bpb_values = [e.val_bpb for e in entries if e.val_bpb is not None]
        ms_values = [e.ms_per_step for e in entries if e.ms_per_step is not None]

        def compute_stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {}
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0
            return {
                "mean": mean,
                "std": variance ** 0.5,
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

        return {
            "config_hash": config_hash,
            "num_seeds": len(entries),
            "seeds": [e.seed for e in entries],
            "bpb": compute_stats(bpb_values),
            "ms_per_step": compute_stats(ms_values),
        }

    def find_volatile_configs(
        self,
        min_seeds: int = 2,
        min_std: float = 0.02,
    ) -> list[dict[str, Any]]:
        """Find configs with high volatility across seeds.

        Args:
            min_seeds: Minimum number of seeds to consider
            min_std: Minimum BPB std dev to consider volatile

        Returns list of volatile configs with statistics.
        """
        # Group by config hash
        config_hashes = set(e.config_hash for e in self.entries.values())
        volatile = []

        for config_hash in config_hashes:
            stats = self.get_seed_statistics(config_hash)
            if stats.get("num_seeds", 0) >= min_seeds:
                bpb_std = stats.get("bpb", {}).get("std", 0)
                if bpb_std >= min_std:
                    volatile.append({
                        "config_hash": config_hash,
                        "num_seeds": stats["num_seeds"],
                        "bpb_std": bpb_std,
                        "bpb_mean": stats.get("bpb", {}).get("mean"),
                        "bpb_range": [
                            stats.get("bpb", {}).get("min"),
                            stats.get("bpb", {}).get("max"),
                        ],
                    })

        return sorted(volatile, key=lambda x: x["bpb_std"], reverse=True)
