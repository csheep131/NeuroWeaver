"""Lokale 8GB-Proxy-Run Konfiguration und Smoke-Test-Utilities.

Dieses Modul bietet Utilities für lokale Entwicklung mit begrenztem VRAM (8GB):
- Proxy-Run-Konfiguration (kurze, valide Tests)
- Smoke-Tests (Startet Modell, kein OOM, Schritt läuft)
- Metriken für lokale Entwicklung (peak_vram_mb, tokens_per_sec, etc.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import Config


@dataclass
class LocalProxyConfig:
    """Konfiguration für lokale 8GB-Proxy-Runs."""

    enabled: bool = False
    seq_len: int = 256  # Reduzierte Sequenzlänge für 8GB VRAM
    microbatch: int = 1  # Minimiert VRAM-Spitzen
    grad_accumulation: int = 8  # Kompensiert kleine Batches
    eval_size: int = 100  # Sehr kleine Evaluierung
    steps: int = 50  # Kurze Runs für Validierung
    smoke_test: bool = False  # Nur Smoke-Test (1-2 Steps)

    @classmethod
    def from_config(cls, config: Config) -> "LocalProxyConfig":
        """Create from main Config object."""
        local_cfg = config.to_dict().get("local_proxy", {})
        return cls(
            enabled=local_cfg.get("enabled", False),
            seq_len=local_cfg.get("seq_len", 256),
            microbatch=local_cfg.get("microbatch", 1),
            grad_accumulation=local_cfg.get("grad_accumulation", 8),
            eval_size=local_cfg.get("eval_size", 100),
            steps=local_cfg.get("steps", 50),
            smoke_test=local_cfg.get("smoke_test", False),
        )

    def apply_to_training(self, training_cfg: dict[str, Any]) -> dict[str, Any]:
        """Apply proxy settings to training config."""
        if not self.enabled:
            return training_cfg

        result = training_cfg.copy()

        # Für Smoke-Test: Nur 1-2 Steps
        if self.smoke_test:
            result["num_steps"] = 2
        else:
            # Für Proxy-Run: Wenige Steps für Validierung
            result["num_steps"] = self.steps

        # Batch-Größe anpassen
        result["batch_size"] = self.microbatch
        result["grad_accumulation"] = self.grad_accumulation

        # Eval-Größe anpassen
        if "eval" in result:
            result["eval"]["batch_size"] = self.eval_size
        else:
            result["eval"] = {"batch_size": self.eval_size}

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "seq_len": self.seq_len,
            "microbatch": self.microbatch,
            "grad_accumulation": self.grad_accumulation,
            "eval_size": self.eval_size,
            "steps": self.steps,
            "smoke_test": self.smoke_test,
        }


@dataclass
class LocalMetrics:
    """Lokale Metriken für 8GB-Proxy-Runs."""

    # VRAM-Metriken
    peak_vram_mb: float = 0.0
    avg_vram_mb: float = 0.0

    # Durchsatz-Metriken
    tokens_per_sec: float = 0.0
    ms_per_step: float = 0.0

    # Stabilitäts-Metriken
    oom_count: int = 0
    steps_completed: int = 0
    compile_time_s: float = 0.0

    # Relative Metriken
    relative_delta_vs_parent_bpb: float | None = None
    relative_delta_vs_parent_ms: float | None = None

    # Metadata
    is_smoke_test: bool = False
    is_proxy: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "peak_vram_mb": self.peak_vram_mb,
            "avg_vram_mb": self.avg_vram_mb,
            "tokens_per_sec": self.tokens_per_sec,
            "ms_per_step": self.ms_per_step,
            "oom_count": self.oom_count,
            "steps_completed": self.steps_completed,
            "compile_time_s": self.compile_time_s,
            "relative_delta_vs_parent_bpb": self.relative_delta_vs_parent_bpb,
            "relative_delta_vs_parent_ms": self.relative_delta_vs_parent_ms,
            "is_smoke_test": self.is_smoke_test,
            "is_proxy": self.is_proxy,
        }

    def check_success_criteria(self) -> tuple[bool, list[str]]:
        """Prüfe Erfolgskriterien für lokalen Run.

        Returns:
            Tuple of (success, list of failed criteria)
        """
        failed = []

        # VRAM-Limit
        if self.peak_vram_mb > 7500:
            failed.append(f"peak_vram_mb={self.peak_vram_mb:.0f} > 7500 MB")

        # OOM-Fehler
        if self.oom_count > 0:
            failed.append(f"oom_count={self.oom_count} > 0")

        # Smoke-Test: Mindestens 1 Step
        if self.is_smoke_test and self.steps_completed < 1:
            failed.append(f"steps_completed={self.steps_completed} < 1 (smoke test)")

        # Proxy-Test: Mindestens 10 Steps
        if self.is_proxy and not self.is_smoke_test and self.steps_completed < 10:
            failed.append(f"steps_completed={self.steps_completed} < 10 (proxy)")

        return len(failed) == 0, failed


@dataclass
class SmokeTestResult:
    """Ergebnis eines Smoke-Tests."""

    run_id: str
    success: bool
    started: bool  # Modell konnte initialisiert werden
    first_step_completed: bool
    metrics_written: bool
    peak_vram_mb: float = 0.0
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "success": self.success,
            "started": self.started,
            "first_step_completed": self.first_step_completed,
            "metrics_written": self.metrics_written,
            "peak_vram_mb": self.peak_vram_mb,
            "error_message": self.error_message,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "✅ PASS" if self.success else "❌ FAIL"
        lines = [
            f"Smoke Test: {self.run_id} - {status}",
            f"  Started: {self.started}",
            f"  First Step: {self.first_step_completed}",
            f"  Metrics Written: {self.metrics_written}",
            f"  Peak VRAM: {self.peak_vram_mb:.0f} MB",
        ]
        if self.error_message:
            lines.append(f"  Error: {self.error_message}")
        return "\n".join(lines)


class LocalProxyRunner:
    """Runner für lokale 8GB-Proxy-Runs."""

    def __init__(self, config: Config):
        self.config = config
        self.proxy_config = LocalProxyConfig.from_config(config)
        self.metrics = LocalMetrics()

    def prepare_for_local_run(self) -> dict[str, Any]:
        """Bereite Training-Konfiguration für lokalen Run vor."""
        if not self.proxy_config.enabled:
            return self.config.to_dict().get("training", {})

        training_cfg = self.config.to_dict().get("training", {})
        return self.proxy_config.apply_to_training(training_cfg)

    def record_vram(self, vram_mb: float) -> None:
        """Record VRAM usage."""
        if vram_mb > self.metrics.peak_vram_mb:
            self.metrics.peak_vram_mb = vram_mb

        # Update average (simple moving average)
        if self.metrics.steps_completed > 0:
            n = self.metrics.steps_completed
            self.metrics.avg_vram_mb = (self.metrics.avg_vram_mb * n + vram_mb) / (n + 1)
        else:
            self.metrics.avg_vram_mb = vram_mb

    def record_step(self, tokens_processed: int, step_time_ms: float) -> None:
        """Record a completed training step."""
        self.metrics.steps_completed += 1
        self.metrics.ms_per_step = step_time_ms

        if step_time_ms > 0:
            self.metrics.tokens_per_sec = tokens_processed / (step_time_ms / 1000.0)

    def record_oom(self) -> None:
        """Record an OOM event."""
        self.metrics.oom_count += 1

    def set_compile_time(self, compile_time_s: float) -> None:
        """Set compile time."""
        self.metrics.compile_time_s = compile_time_s

    def compute_relative_deltas(self, parent_metrics: dict[str, Any]) -> None:
        """Compute relative deltas vs parent run."""
        if parent_metrics.get("val_bpb") is not None and self.metrics.relative_delta_vs_parent_bpb is None:
            # Wird später gesetzt wenn val_bpb bekannt
            pass

        if parent_metrics.get("ms_per_step") is not None:
            parent_ms = parent_metrics["ms_per_step"]
            if parent_ms > 0:
                self.metrics.relative_delta_vs_parent_ms = (
                    (self.metrics.ms_per_step - parent_ms) / parent_ms * 100
                )

    def check_success(self) -> tuple[bool, list[str]]:
        """Check success criteria for local run."""
        return self.metrics.check_success_criteria()

    def get_metrics(self) -> LocalMetrics:
        """Get collected metrics."""
        return self.metrics


def create_smoke_test_config(run_config_path: str) -> Config:
    """Create a smoke test configuration from a run config.

    Args:
        run_config_path: Path to the run configuration file

    Returns:
        Config object with smoke test settings
    """
    from core.config import load_config

    config = load_config(run_config_path)

    # Override with smoke test settings
    raw = config.to_dict()
    raw["local_proxy"] = {
        "enabled": True,
        "seq_len": 256,
        "microbatch": 1,
        "grad_accumulation": 1,
        "eval_size": 10,
        "steps": 2,
        "smoke_test": True,
    }

    # Override training steps
    if "training" in raw:
        raw["training"]["num_steps"] = 2
        raw["training"]["batch_size"] = 1

    return Config(
        run_id=raw.get("run_id", "") + "_smoke",
        parent_run_id=raw.get("parent_run_id"),
        seed=raw.get("seed", 42),
        model=raw.get("model", {}),
        tokenizer=raw.get("tokenizer", {}),
        training=raw.get("training", {}),
        eval=raw.get("eval", {}),
        quant=raw.get("quant", {}),
        features=raw.get("features", {}),
        _raw=raw,
    )


def check_local_prerequisites() -> dict[str, Any]:
    """Check local prerequisites for running.

    Returns:
        Dictionary with check results
    """
    result = {
        "cuda_available": False,
        "vram_total_mb": 0,
        "vram_available_mb": 0,
        "can_run": False,
        "warnings": [],
    }

    try:
        import torch

        if torch.cuda.is_available():
            result["cuda_available"] = True
            result["vram_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            result["vram_available_mb"] = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated(0)
            ) / 1024 / 1024

            if result["vram_total_mb"] < 7000:
                result["warnings"].append(
                    f"VRAM ({result['vram_total_mb']:.0f} MB) is below recommended 8GB"
                )

            result["can_run"] = True
        else:
            result["warnings"].append("CUDA not available - will run on CPU (slow)")
            result["can_run"] = True  # Can still run on CPU

    except ImportError:
        result["warnings"].append("PyTorch not installed - will use stub implementations")
        result["can_run"] = True  # Can use stubs

    return result
