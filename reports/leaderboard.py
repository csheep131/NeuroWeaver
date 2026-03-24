"""Leaderboard generation for runs."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class LeaderboardEntry:
    """Entry in the leaderboard."""

    run_id: str
    rank: int
    val_bpb: float
    artifact_bytes: int
    ms_per_step: float | None = None
    config_hash: str = ""
    parent_run_id: str | None = None
    seed: int = 42
    completed_at: str = ""

    @property
    def artifact_mb(self) -> float:
        """Get artifact size in MB."""
        return self.artifact_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "run_id": self.run_id,
            "val_bpb": self.val_bpb,
            "artifact_bytes": self.artifact_bytes,
            "artifact_mb": self.artifact_bytes / (1024 * 1024),
            "ms_per_step": self.ms_per_step,
            "config_hash": self.config_hash,
            "parent_run_id": self.parent_run_id,
            "seed": self.seed,
            "completed_at": self.completed_at,
        }


@dataclass
class Leaderboard:
    """Leaderboard of runs."""

    category: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    entries: list[LeaderboardEntry] = field(default_factory=list)
    total_runs: int = 0
    finalists: int = 0
    killed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "generated_at": self.generated_at,
            "total_runs": self.total_runs,
            "finalists": self.finalists,
            "killed": self.killed,
            "entries": [e.to_dict() for e in self.entries],
        }

    def save(self, path: str | Path) -> None:
        """Save leaderboard to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_table(self) -> str:
        """Print leaderboard as a formatted table."""
        lines = [
            "=" * 80,
            f"LEADERBOARD: {self.category}",
            f"Generated: {self.generated_at}",
            "=" * 80,
            "",
        ]

        if not self.entries:
            lines.append("No entries")
            return "\n".join(lines)

        # Header
        header = f"{'Rank':<5} {'Run ID':<20} {'BPB':<10} {'Size (MB)':<12} {'ms/step':<10} {'Parent':<20}"
        lines.append(header)
        lines.append("-" * 80)

        # Entries
        for entry in self.entries:
            parent = entry.parent_run_id or "-"
            ms_step = f"{entry.ms_per_step:.2f}" if entry.ms_per_step else "N/A"
            row = (
                f"{entry.rank:<5} "
                f"{entry.run_id:<20} "
                f"{entry.val_bpb:<10.4f} "
                f"{entry.artifact_mb:<12.2f} "
                f"{ms_step:<10} "
                f"{parent:<20}"
            )
            lines.append(row)

        lines.append("")
        lines.append(f"Total: {self.total_runs} runs | "
                    f"Finalists: {self.finalists} | "
                    f"Killed: {self.killed}")
        lines.append("=" * 80)

        return "\n".join(lines)


class LeaderboardGenerator:
    """Generator for leaderboards."""

    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)

    def load_registry(self) -> dict[str, Any]:
        """Load the run registry."""
        registry_path = self.results_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                return json.load(f)
        return {}

    def generate_by_bpb(self, top_k: int = 20) -> Leaderboard:
        """Generate leaderboard sorted by BPB."""
        registry = self.load_registry()

        # Filter completed runs with valid BPB
        valid_runs = [
            data for data in registry.values()
            if data.get("status") == "completed"
            and data.get("val_bpb") is not None
            and data.get("artifact_bytes", 0) <= 16_000_000
        ]

        # Sort by BPB (lower is better)
        valid_runs.sort(key=lambda r: r.get("val_bpb", float("inf")))

        # Create entries
        entries = [
            LeaderboardEntry(
                rank=i + 1,
                run_id=data.get("run_id", ""),
                val_bpb=data.get("val_bpb", 0),
                artifact_bytes=data.get("artifact_bytes", 0),
                ms_per_step=data.get("ms_per_step"),
                config_hash=data.get("config_hash", ""),
                parent_run_id=data.get("parent_run_id"),
                seed=data.get("seed", 42),
                completed_at=data.get("end_time", ""),
            )
            for i, data in enumerate(valid_runs[:top_k])
        ]

        # Count statistics
        total = len(registry)
        finalists = len([d for d in registry.values() if d.get("status") == "completed"])
        killed = len([d for d in registry.values() if d.get("status") in ("killed", "failed")])

        return Leaderboard(
            category="Best BPB",
            total_runs=total,
            finalists=finalists,
            killed=killed,
            entries=entries,
        )

    def generate_by_efficiency(self, top_k: int = 20) -> Leaderboard:
        """Generate leaderboard sorted by BPB efficiency (BPB per MB)."""
        registry = self.load_registry()

        # Filter and compute efficiency
        valid_runs = []
        for data in registry.values():
            if (data.get("status") == "completed"
                and data.get("val_bpb") is not None
                and data.get("artifact_bytes", 0) > 0):

                efficiency = data["val_bpb"] / (data.get("artifact_bytes", 1) / (1024 * 1024))
                data_with_eff = data.copy()
                data_with_eff["efficiency"] = efficiency
                valid_runs.append(data_with_eff)

        # Sort by efficiency (lower is better)
        valid_runs.sort(key=lambda r: r.get("efficiency", float("inf")))

        # Create entries
        entries = [
            LeaderboardEntry(
                rank=i + 1,
                run_id=data.get("run_id", ""),
                val_bpb=data.get("val_bpb", 0),
                artifact_bytes=data.get("artifact_bytes", 0),
                ms_per_step=data.get("ms_per_step"),
                config_hash=data.get("config_hash", ""),
                parent_run_id=data.get("parent_run_id"),
                seed=data.get("seed", 42),
                completed_at=data.get("end_time", ""),
            )
            for i, data in enumerate(valid_runs[:top_k])
        ]

        total = len(registry)
        finalists = len([d for d in registry.values() if d.get("status") == "completed"])
        killed = len([d for d in registry.values() if d.get("status") in ("killed", "failed")])

        return Leaderboard(
            category="Best BPB/MB (Efficiency)",
            total_runs=total,
            finalists=finalists,
            killed=killed,
            entries=entries,
        )

    def generate_by_speed(self, top_k: int = 20) -> Leaderboard:
        """Generate leaderboard sorted by speed (ms/step)."""
        registry = self.load_registry()

        # Filter runs with timing data
        valid_runs = [
            data for data in registry.values()
            if data.get("status") == "completed"
            and data.get("ms_per_step") is not None
            and data.get("val_bpb") is not None
        ]

        # Sort by speed (lower is better)
        valid_runs.sort(key=lambda r: r.get("ms_per_step", float("inf")))

        # Create entries
        entries = [
            LeaderboardEntry(
                rank=i + 1,
                run_id=data.get("run_id", ""),
                val_bpb=data.get("val_bpb", 0),
                artifact_bytes=data.get("artifact_bytes", 0),
                ms_per_step=data.get("ms_per_step"),
                config_hash=data.get("config_hash", ""),
                parent_run_id=data.get("parent_run_id"),
                seed=data.get("seed", 42),
                completed_at=data.get("end_time", ""),
            )
            for i, data in enumerate(valid_runs[:top_k])
        ]

        total = len(registry)
        finalists = len([d for d in registry.values() if d.get("status") == "completed"])
        killed = len([d for d in registry.values() if d.get("status") in ("killed", "failed")])

        return Leaderboard(
            category="Fastest (ms/step)",
            total_runs=total,
            finalists=finalists,
            killed=killed,
            entries=entries,
        )

    def generate_all(self, output_dir: str | Path | None = None) -> dict[str, Leaderboard]:
        """Generate all leaderboards and save them."""
        output_dir = output_dir or self.results_dir
        output_path = Path(output_dir)

        leaderboards = {
            "bpb": self.generate_by_bpb(),
            "efficiency": self.generate_by_efficiency(),
            "speed": self.generate_by_speed(),
        }

        for name, leaderboard in leaderboards.items():
            leaderboard.save(output_path / f"leaderboard_{name}.json")

        return leaderboards
