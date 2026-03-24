"""Logging utilities for runs."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logging(run_id: str, results_dir: str | Path = "results") -> "RunLogger":
    """Set up logging for a run."""
    return RunLogger(run_id, results_dir)


class RunLogger:
    """Logger for a single run."""

    def __init__(self, run_id: str, results_dir: str | Path = "results"):
        self.run_id = run_id
        self.results_dir = Path(results_dir)
        self.run_dir = self.results_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Set up Python logger
        self.logger = logging.getLogger(f"run.{run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            f"%(asctime)s [%(levelname)s] {run_id}: %(message)s"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler for JSONL log
        self.log_path = self.run_dir / "train_log.jsonl"
        
        # Buffering for log writes
        self._log_buffer: list[str] = []
        self._buffer_size = 100  # Flush after 100 entries
        
        # Metrics buffer
        self.metrics_buffer: list[dict[str, Any]] = []

    def _write_log_entry(self, entry: dict[str, Any]) -> None:
        """Write a log entry to the JSONL file with buffering."""
        self._log_buffer.append(json.dumps(entry))
        if len(self._log_buffer) >= self._buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Flush buffered log entries to disk."""
        if not self._log_buffer:
            return
        with open(self.log_path, "a") as f:
            for entry in self._log_buffer:
                f.write(entry + "\n")
        self._log_buffer.clear()

    def log_step(self, step: int, metrics: dict[str, Any], phase: str = "train") -> None:
        """Log metrics for a training step."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "step": step,
            **metrics,
        }
        self._write_log_entry(entry)
        self.metrics_buffer.append(entry)

    def log_eval(self, step: int, metrics: dict[str, Any]) -> None:
        """Log evaluation metrics."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": "eval",
            "step": step,
            **metrics,
        }
        self._write_log_entry(entry)

    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
        self._write_log_entry({
            "timestamp": datetime.now().isoformat(),
            "level": "info",
            "message": message,
        })

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
        self._write_log_entry({
            "timestamp": datetime.now().isoformat(),
            "level": "warning",
            "message": message,
        })

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
        self._write_log_entry({
            "timestamp": datetime.now().isoformat(),
            "level": "error",
            "message": message,
        })

    def save_metrics(self, filename: str = "metrics.json") -> None:
        """Save aggregated metrics to JSON file."""
        metrics_path = self.run_dir / filename

        # Aggregate metrics
        aggregated: dict[str, Any] = {
            "run_id": self.run_id,
            "total_steps": len([m for m in self.metrics_buffer if m.get("phase") == "train"]),
            "eval_count": len([m for m in self.metrics_buffer if m.get("phase") == "eval"]),
        }

        # Compute averages for numeric metrics
        numeric_metrics: dict[str, list[float]] = {}
        for entry in self.metrics_buffer:
            for key, value in entry.items():
                if key not in ("timestamp", "phase", "step", "level", "message", "run_id"):
                    if isinstance(value, (int, float)):
                        if key not in numeric_metrics:
                            numeric_metrics[key] = []
                        numeric_metrics[key].append(float(value))

        for key, values in numeric_metrics.items():
            aggregated[f"{key}_mean"] = sum(values) / len(values)
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)
            aggregated[f"{key}_last"] = values[-1] if values else None

        with open(metrics_path, "w") as f:
            json.dump(aggregated, f, indent=2)

        self.log_info(f"Saved metrics to {metrics_path}")

    def save_final_report(self, final_metrics: dict[str, Any]) -> None:
        """Save final metrics report."""
        self._flush_buffer()  # Ensure all logs are written before final report
        report_path = self.run_dir / "eval.json"
        with open(report_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
        self.log_info(f"Saved final report to {report_path}")
    
    def flush(self) -> None:
        """Force flush any buffered log entries."""
        self._flush_buffer()
    
    def close(self) -> None:
        """Close the logger and flush all buffers."""
        self.flush()
