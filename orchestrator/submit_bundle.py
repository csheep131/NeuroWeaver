"""Submission Bundle Creator für Phase 3.

Dieses Modul erstellt Submission-Bundles für die H100 Challenge.
Ein Bundle enthält:
- Model weights (quantisiert)
- Config-Dateien
- Trainings-Logs (optional)
- Metriken-Zusammenfassung
- Seed-Statistiken
- Lineage-Übersicht
- README

Challenge-Anforderungen:
- artifact_bytes < 16 MB
- val_bpb < 1.50
- Stabil mit 3 Seeds (σ < 0.03 BPB)
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.registry import RunRegistry, RunEntry


@dataclass
class SubmissionBundle:
"""Container für Submission-Bundle-Daten."""

bundle_id: str
run_ids: list[str]
created_at: str = field(default_factory=lambda: datetime.now().isoformat())

# Metriken
val_bpb: float | None = None
quantized_val_bpb: float | None = None
artifact_bytes: int = 0
ms_per_step: float | None = None

# Seed-Statistiken
num_seeds: int = 1
bpb_mean: float | None = None
bpb_std: float | None = None
bpb_min: float | None = None
bpb_max: float | None = None

# Lineage
lineage: dict[str, Any] = field(default_factory=dict)

# Files
config_files: list[str] = field(default_factory=list)
weight_files: list[str] = field(default_factory=list)
log_files: list[str] = field(default_factory=list)

# Metadata
notes: str = ""
tags: list[str] = field(default_factory=list)

def to_dict(self) -> dict[str, Any]:
"""Convert to dictionary."""
return {
"bundle_id": self.bundle_id,
"run_ids": self.run_ids,
"created_at": self.created_at,
"metrics": {
"val_bpb": self.val_bpb,
"quantized_val_bpb": self.quantized_val_bpb,
"artifact_bytes": self.artifact_bytes,
"ms_per_step": self.ms_per_step,
},
"seed_statistics": {
"num_seeds": self.num_seeds,
"bpb_mean": self.bpb_mean,
"bpb_std": self.bpb_std,
"bpb_min": self.bpb_min,
"bpb_max": self.bpb_max,
},
"lineage": self.lineage,
"files": {
"config_files": self.config_files,
"weight_files": self.weight_files,
"log_files": self.log_files,
},
"metadata": {
"notes": self.notes,
"tags": self.tags,
},
}

def check_submission_criteria(self) -> tuple[bool, list[str]]:
"""Check if bundle meets submission criteria.

Returns:
Tuple of (meets_criteria, list of failures)
"""
failures = []

# Artifact size limit
if self.artifact_bytes > 16_000_000:
failures.append(
f"artifact_bytes={self.artifact_bytes:,} > 16MB (Challenge-Limit)"
)

# BPB threshold
if self.quantized_val_bpb is not None and self.quantized_val_bpb > 1.50:
failures.append(
f"quantized_val_bpb={self.quantized_val_bpb:.4f} > 1.50"
)
elif self.val_bpb is not None and self.val_bpb > 1.50:
failures.append(f"val_bpb={self.val_bpb:.4f} > 1.50")

# Seed stability (if multiple seeds)
if self.num_seeds >= 3 and self.bpb_std is not None:
if self.bpb_std > 0.03:
failures.append(
f"bpb_std={self.bpb_std:.4f} > 0.03 (Seed-Stabilität)"
)

return len(failures) == 0, failures


@dataclass
class SubmissionBuilder:
"""Builder für Submission-Bundles."""

registry: RunRegistry
output_dir: str | Path = "submissions"
_run_cache: dict[str, RunEntry | None] = field(default_factory=dict, init=False)

def _get_run_entry(self, run_id: str) -> RunEntry | None:
"""Get run entry with caching."""
if run_id not in self._run_cache:
self._run_cache[run_id] = self.registry.get(run_id)
return self._run_cache[run_id]

def create_bundle(
self,
bundle_id: str,
run_ids: list[str],
include_configs: bool = True,
include_logs: bool = False,
include_weights: bool = True,
) -> SubmissionBundle:
"""Create a submission bundle.

Args:
bundle_id: ID for the bundle
run_ids: List of run IDs to include
include_configs: Whether to include config files
include_logs: Whether to include training logs
include_weights: Whether to include model weights

Returns:
Created SubmissionBundle
"""
# Pre-cache all run entries
for run_id in run_ids:
self._get_run_entry(run_id)

bundle = SubmissionBundle(
bundle_id=bundle_id,
run_ids=run_ids,
)

# Collect metrics from runs
self._collect_metrics(bundle)

# Compute seed statistics
self._compute_seed_statistics(bundle)

# Build lineage
self._build_lineage(bundle)

# Collect files
if include_configs:
self._collect_configs(bundle)
if include_logs:
self._collect_logs(bundle)
if include_weights:
self._collect_weights(bundle)

return bundle

def _collect_metrics(self, bundle: SubmissionBundle) -> None:
"""Collect metrics from bundle runs."""
val_bpbs = []
quantized_bpbs = []
ms_steps = []
total_artifact = 0

for run_id in bundle.run_ids:
entry = self._get_run_entry(run_id)
if not entry:
continue

if entry.val_bpb is not None:
val_bpbs.append(entry.val_bpb)
if entry.quantized_val_bpb is not None:
quantized_bpbs.append(entry.quantized_val_bpb)
if entry.ms_per_step is not None:
ms_steps.append(entry.ms_per_step)
total_artifact += entry.artifact_bytes

if val_bpbs:
bundle.val_bpb = min(val_bpbs) # Best BPB
if quantized_bpbs:
bundle.quantized_val_bpb = min(quantized_bpbs)
if ms_steps:
bundle.ms_per_step = sum(ms_steps) / len(ms_steps) # Average
bundle.artifact_bytes = total_artifact

def _compute_seed_statistics(self, bundle: SubmissionBundle) -> None:
"""Compute statistics across seeds."""
bundle.num_seeds = len(bundle.run_ids)

if bundle.num_seeds == 1:
return

# Get BPB values
bpb_values = []
for run_id in bundle.run_ids:
entry = self._get_run_entry(run_id)
if entry and entry.val_bpb is not None:
bpb_values.append(entry.val_bpb)

if len(bpb_values) >= 2:
bundle.bpb_mean = sum(bpb_values) / len(bpb_values)
bundle.bpb_min = min(bpb_values)
bundle.bpb_max = max(bpb_values)

# Compute std dev
variance = sum(
(v - bundle.bpb_mean) ** 2 for v in bpb_values
) / len(bpb_values)
bundle.bpb_std = variance ** 0.5

def _build_lineage(self, bundle: SubmissionBundle) -> None:
"""Build lineage tree for bundle."""
lineage = {}
for run_id in bundle.run_ids:
entry = self.registry.get(run_id)
if entry:
lineage[run_id] = {
"parent": entry.parent_run_id,
"config_hash": entry.config_hash,
"status": entry.status,
}
bundle.lineage = lineage

def _collect_configs(self, bundle: SubmissionBundle) -> None:
"""Collect config files from runs."""
configs_dir = Path("configs/runs")
for run_id in bundle.run_ids:
config_path = configs_dir / f"{run_id}.yaml"
if config_path.exists():
bundle.config_files.append(str(config_path))

def _collect_logs(self, bundle: SubmissionBundle) -> None:
"""Collect log files from runs."""
results_dir = Path("results")
for run_id in bundle.run_ids:
log_path = results_dir / f"{run_id}" / "train.log"
if log_path.exists():
bundle.log_files.append(str(log_path))

def _collect_weights(self, bundle: SubmissionBundle) -> None:
"""Collect model weight files."""
results_dir = Path("results")
for run_id in bundle.run_ids:
# Check for weight files
for pattern in ["*.pt", "*.pth", "*.safetensors"]:
weight_files = list((results_dir / run_id).glob(pattern))
for wf in weight_files:
bundle.weight_files.append(str(wf))

def save_bundle(
self,
bundle: SubmissionBundle,
format: str = "zip",
) -> Path:
"""Save bundle to disk.

Args:
bundle: Bundle to save
format: Output format (zip, json)

Returns:
Path to saved bundle
"""
output_dir = Path(self.output_dir) / bundle.bundle_id
output_dir.mkdir(parents=True, exist_ok=True)

if format == "json":
# Save as JSON
output_path = output_dir / "bundle.json"
with open(output_path, "w") as f:
json.dump(bundle.to_dict(), f, indent=2)
return output_path

elif format == "zip":
# Create zip archive
output_path = output_dir / f"{bundle.bundle_id}.zip"

with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
# Add metadata
metadata_path = output_dir / "bundle.json"
with open(metadata_path, "w") as f:
json.dump(bundle.to_dict(), f, indent=2)
zf.write(metadata_path, "bundle.json")

# Add config files
for config_path in bundle.config_files:
if Path(config_path).exists():
zf.write(
config_path,
f"configs/{Path(config_path).name}",
)

# Add log files
for log_path in bundle.log_files:
if Path(log_path).exists():
zf.write(
log_path,
f"logs/{Path(log_path).name}",
)

# Add weight files
for weight_path in bundle.weight_files:
if Path(weight_path).exists():
zf.write(
weight_path,
f"weights/{Path(weight_path).name}",
)

return output_path

else:
raise ValueError(f"Unknown format: {format}")

def create_readme(self, bundle: SubmissionBundle) -> str:
"""Create README for bundle."""
lines = [
f"# Submission Bundle: {bundle.bundle_id}",
"",
f"Created: {bundle.created_at}",
"",
"## Runs Included",
"",
]

for run_id in bundle.run_ids:
entry = self.registry.get(run_id)
if entry:
lines.append(f"- {run_id}")
lines.append(f" - Status: {entry.status}")
if entry.val_bpb:
lines.append(f" - val_bpb: {entry.val_bpb:.4f}")
if entry.parent_run_id:
lines.append(f" - Parent: {entry.parent_run_id}")
lines.append("")

lines.extend([
"## Metrics Summary",
"",
f"- val_bpb: {bundle.val_bpb:.4f}" if bundle.val_bpb else "- val_bpb: N/A",
f"- quantized_val_bpb: {bundle.quantized_val_bpb:.4f}" if bundle.quantized_val_bpb else "- quantized_val_bpb: N/A",
f"- artifact_bytes: {bundle.artifact_bytes:,}",
f"- ms_per_step: {bundle.ms_per_step:.2f}" if bundle.ms_per_step else "- ms_per_step: N/A",
"",
])

if bundle.num_seeds > 1:
lines.extend([
"## Seed Statistics",
"",
f"- num_seeds: {bundle.num_seeds}",
f"- bpb_mean: {bundle.bpb_mean:.4f}" if bundle.bpb_mean else "- bpb_mean: N/A",
f"- bpb_std: {bundle.bpb_std:.4f}" if bundle.bpb_std else "- bpb_std: N/A",
f"- bpb_min: {bundle.bpb_min:.4f}" if bundle.bpb_min else "- bpb_min: N/A",
f"- bpb_max: {bundle.bpb_max:.4f}" if bundle.bpb_max else "- bpb_max: N/A",
"",
])

lines.extend([
"## Submission Criteria",
"",
])

meets_criteria, failures = bundle.check_submission_criteria()
if meets_criteria:
lines.append(" Meets all submission criteria")
else:
lines.append(" Does not meet submission criteria:")
for failure in failures:
lines.append(f" - {failure}")

lines.append("")

return "\n".join(lines)


def create_submission_builder(
registry: RunRegistry | None = None,
output_dir: str = "submissions",
) -> SubmissionBuilder:
"""Create a submission builder."""
return SubmissionBuilder(
registry=registry or RunRegistry(),
output_dir=output_dir,
)


def create_submission_bundle(
bundle_id: str,
run_ids: list[str],
output_dir: str = "submissions",
include_configs: bool = True,
include_logs: bool = False,
include_weights: bool = True,
) -> tuple[SubmissionBundle, Path]:
"""Convenience function to create and save a submission bundle.

Args:
bundle_id: ID for the bundle
run_ids: List of run IDs to include
output_dir: Directory to save bundle
include_configs: Whether to include config files
include_logs: Whether to include training logs
include_weights: Whether to include model weights

Returns:
Tuple of (bundle, output_path)
"""
builder = create_submission_builder(output_dir=output_dir)
bundle = builder.create_bundle(
bundle_id=bundle_id,
run_ids=run_ids,
include_configs=include_configs,
include_logs=include_logs,
include_weights=include_weights,
)

# Save bundle
output_path = builder.save_bundle(bundle)

# Create README
readme_path = output_path.parent / "README.md"
with open(readme_path, "w") as f:
f.write(builder.create_readme(bundle))

return bundle, output_path
