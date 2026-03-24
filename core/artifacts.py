"""Artifact tracking and reporting."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArtifactReport:
    """Report on model artifacts."""

    run_id: str
    total_bytes: int = 0
    component_sizes: dict[str, int] = None
    quantized: bool = False
    quantized_bytes: int = 0
    compression_ratio: float = 1.0

    def __post_init__(self):
        if self.component_sizes is None:
            self.component_sizes = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "total_bytes": self.total_bytes,
            "component_sizes": self.component_sizes,
            "quantized": self.quantized,
            "quantized_bytes": self.quantized_bytes,
            "compression_ratio": self.compression_ratio,
        }

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_artifact_size(directory: str | Path) -> int:
    """Compute total size of all files in a directory."""
    path = Path(directory)
    if not path.exists():
        return 0

    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def compute_file_size(filepath: str | Path) -> int:
    """Compute size of a single file."""
    path = Path(filepath)
    if path.exists() and path.is_file():
        return path.stat().st_size
    return 0


class ArtifactReporter:
    """Reporter for model artifacts."""

    def __init__(self, run_id: str, results_dir: str | Path = "results"):
        self.run_id = run_id
        self.results_dir = Path(results_dir)
        self.artifacts_dir = self.results_dir / run_id / "artifacts"

    def create_report(
        self,
        model_path: str | Path | None = None,
        quantized_path: str | Path | None = None,
    ) -> ArtifactReport:
        """Create an artifact report."""
        report = ArtifactReport(run_id=self.run_id)

        # Compute model size
        if model_path:
            path = Path(model_path)
            if path.is_file():
                report.total_bytes = compute_file_size(path)
                report.component_sizes["model"] = report.total_bytes
            elif path.is_dir():
                report.total_bytes = compute_artifact_size(path)
                report.component_sizes["model_dir"] = report.total_bytes

        # Compute quantized size
        if quantized_path:
            path = Path(quantized_path)
            if path.exists():
                report.quantized = True
                report.quantized_bytes = (
                    compute_file_size(path)
                    if path.is_file()
                    else compute_artifact_size(path)
                )
                report.component_sizes["quantized"] = report.quantized_bytes

                if report.total_bytes > 0:
                    report.compression_ratio = (
                        report.quantized_bytes / report.total_bytes
                    )

        return report

    def check_limit(self, max_bytes: int = 16_000_000) -> tuple[bool, str]:
        """Check if artifacts are within the size limit.

        Returns:
            Tuple of (within_limit, message)
        """
        total_size = compute_artifact_size(self.artifacts_dir)

        if total_size > max_bytes:
            return False, f"Artifact size {total_size:,} bytes exceeds limit of {max_bytes:,} bytes"
        return True, f"Artifact size {total_size:,} bytes within limit"

    def save_report(
        self,
        model_path: str | Path | None = None,
        quantized_path: str | Path | None = None,
    ) -> ArtifactReport:
        """Create and save an artifact report."""
        report = self.create_report(model_path, quantized_path)
        report_path = self.results_dir / self.run_id / "artifact_report.json"
        report.save(report_path)
        return report
