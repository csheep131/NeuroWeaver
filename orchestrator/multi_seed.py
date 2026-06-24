"""Multi-Seed Orchestrierung für reproduzierbare Ergebnisse.

This module handles running experiments with multiple seeds
to ensure statistical significance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from core.config import Config, load_config
from core.registry import RunRegistry, RunEntry
from core.seed import set_seed


@dataclass
class MultiSeedConfig:
"""Configuration for multi-seed runs."""

base_config_path: str
num_seeds: int = 3
seeds: list[int] = field(default_factory=list)
seed_range: tuple[int, int] = (0, 1000)

def __post_init__(self):
if not self.seeds and self.num_seeds:
# Generate random seeds
random.seed(42) # Deterministic seed generation
self.seeds = [
random.randint(*self.seed_range) for _ in range(self.num_seeds)
]


@dataclass
class SeedRun:
"""A single seed run."""

run_id: str
seed: int
config: Config
status: str = "pending"
metrics: dict[str, Any] = field(default_factory=dict)


class MultiSeedOrchestrator:
"""Orchestrator for multi-seed experiments.

Manages running the same configuration with different seeds
and aggregating results.
"""

def __init__(
self,
config: MultiSeedConfig,
registry: RunRegistry | None = None,
):
self.config = config
self.registry = registry or RunRegistry()
self.runs: list[SeedRun] = []

def prepare_runs(self, run_id_prefix: str = "") -> list[SeedRun]:
"""Prepare all seed runs.

Args:
run_id_prefix: Prefix for run IDs

Returns:
List of prepared SeedRun objects
"""
base_config = load_config(self.config.base_config_path)
runs = []

for i, seed in enumerate(self.config.seeds):
# Generate run ID
if run_id_prefix:
run_id = f"{run_id_prefix}_seed{i+1:03d}"
else:
run_id = f"{base_config.run_id or 'run'}_seed{i+1:03d}"

# Create config for this seed
config_dict = base_config.to_dict()
config_dict["run_id"] = run_id
config_dict["seed"] = seed

from core.config import _parse_config

seed_config = _parse_config(config_dict)

run = SeedRun(
run_id=run_id,
seed=seed,
config=seed_config,
)
runs.append(run)

# Register in registry
self.registry.register(
run_id=run_id,
config_hash=seed_config.config_hash,
parent_run_id=base_config.parent_run_id,
seed=seed,
)

self.runs = runs
return runs

def get_run_configs(self) -> list[dict[str, Any]]:
"""Get run configurations for execution.

Returns list of config dicts ready for saving.
"""
if not self.runs:
self.prepare_runs()

return [run.config.to_dict() for run in self.runs]

def update_run_status(
self,
run_id: str,
status: str,
metrics: dict[str, Any] | None = None,
) -> None:
"""Update status of a seed run."""
for run in self.runs:
if run.run_id == run_id:
run.status = status
if metrics:
run.metrics = metrics
break

def get_aggregated_results(self) -> dict[str, Any]:
"""Get aggregated results across all seeds.

Returns statistics like mean, std, min, max for each metric.
"""
# Collect metrics from registry
entries: list[RunEntry] = []
for run in self.runs:
entry = self.registry.get(run.run_id)
if entry and entry.status == "completed":
entries.append(entry)

if not entries:
return {"error": "No completed runs found"}

# Aggregate metrics
result = {
"num_seeds": len(entries),
"seeds": [e.seed for e in entries],
"completed": len(entries),
"pending": len(self.runs) - len(entries),
}

# BPB statistics
bpb_values = [e.val_bpb for e in entries if e.val_bpb is not None]
if bpb_values:
result["bpb"] = {
"mean": sum(bpb_values) / len(bpb_values),
"std": self._compute_std(bpb_values),
"min": min(bpb_values),
"max": max(bpb_values),
"values": bpb_values,
}

# Timing statistics
ms_values = [e.ms_per_step for e in entries if e.ms_per_step is not None]
if ms_values:
result["ms_per_step"] = {
"mean": sum(ms_values) / len(ms_values),
"std": self._compute_std(ms_values),
"min": min(ms_values),
"max": max(ms_values),
}

# Artifact size
sizes = [e.artifact_bytes for e in entries]
result["artifact_bytes"] = {
"mean": sum(sizes) / len(sizes) if sizes else 0,
"max": max(sizes) if sizes else 0,
}

# Steps
steps = [e.steps_completed for e in entries]
result["steps_completed"] = {
"total": sum(steps),
"mean": sum(steps) / len(steps) if steps else 0,
}

return result

def _compute_std(self, values: list[float]) -> float:
"""Compute standard deviation."""
if len(values) < 2:
return 0.0
mean = sum(values) / len(values)
variance = sum((v - mean) ** 2 for v in values) / len(values)
return variance ** 0.5

def print_summary(self) -> str:
"""Print summary of multi-seed runs."""
lines = [
"=" * 60,
"MULTI-SEED EXPERIMENT",
f"Base Config: {self.config.base_config_path}",
f"Number of Seeds: {len(self.config.seeds)}",
f"Seeds: {self.config.seeds}",
"=" * 60,
"",
"Run Status:",
]

for run in self.runs:
status_icon = "" if run.status == "completed" else ""
lines.append(f" [{status_icon}] {run.run_id} (seed={run.seed})")

# Add aggregated results if available
results = self.get_aggregated_results()
if "bpb" in results:
lines.append("")
lines.append("Aggregated Results:")
lines.append(f" BPB: {results['bpb']['mean']:.4f} ± {results['bpb']['std']:.4f}")
lines.append(f" Range: [{results['bpb']['min']:.4f}, {results['bpb']['max']:.4f}]")

if "ms_per_step" in results:
lines.append(f" ms/step: {results['ms_per_step']['mean']:.2f} ± {results['ms_per_step']['std']:.2f}")

lines.append("")
lines.append("=" * 60)

return "\n".join(lines)

def check_significance(self, min_improvement: float = 0.05) -> dict[str, Any]:
"""Check if results are statistically significant.

Args:
min_improvement: Minimum BPB improvement to consider significant

Returns:
Dictionary with significance analysis
"""
results = self.get_aggregated_results()

if "bpb" not in results:
return {"significant": False, "reason": "No BPB data available"}

bpb = results["bpb"]
std = bpb["std"]
mean = bpb["mean"]

# Coefficient of variation
cv = std / mean if mean > 0 else float("inf")

# Check if std is small relative to improvement threshold
is_significant = std < min_improvement / 2

return {
"significant": is_significant,
"coefficient_of_variation": cv,
"std": std,
"mean": mean,
"min_improvement_threshold": min_improvement,
"recommendation": (
"Results are consistent across seeds" if is_significant
else "High variance - consider more seeds"
),
}


def create_multi_seed_orchestrator(
base_config_path: str,
num_seeds: int = 3,
seeds: list[int] | None = None,
registry: RunRegistry | None = None,
) -> MultiSeedOrchestrator:
"""Convenience function to create orchestrator."""
config = MultiSeedConfig(
base_config_path=base_config_path,
num_seeds=num_seeds,
seeds=seeds or [],
)
return MultiSeedOrchestrator(config, registry)
