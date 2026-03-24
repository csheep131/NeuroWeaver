"""Benchmark utilities for timing and performance."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    total_time_ms: float
    iterations: int
    time_per_iteration_ms: float
    throughput_per_second: float | None = None
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_time_ms": self.total_time_ms,
            "iterations": self.iterations,
            "time_per_iteration_ms": self.time_per_iteration_ms,
            "throughput_per_second": self.throughput_per_second,
            "extra_metrics": self.extra_metrics,
        }


class Benchmark:
    """Benchmark utility for measuring performance."""

    def __init__(self, name: str = "benchmark"):
        self.name = name
        self.results: list[BenchmarkResult] = []

    @contextmanager
    def measure(self, name: str | None = None):
        """Context manager for measuring time.

        Usage:
            with benchmark.measure("forward_pass"):
                result = model.forward(inputs)
        """
        benchmark_name = name or self.name
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            self.results.append(BenchmarkResult(
                name=benchmark_name,
                total_time_ms=elapsed_ms,
                iterations=1,
                time_per_iteration_ms=elapsed_ms,
            ))

    def time_function(self, func: callable, *args: Any, iterations: int = 10, **kwargs: Any) -> BenchmarkResult:
        """Time a function over multiple iterations.

        Args:
            func: Function to time
            *args: Arguments to pass to the function
            iterations: Number of iterations
            **kwargs: Keyword arguments to pass to the function

        Returns:
            BenchmarkResult with timing statistics
        """
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        total_time = sum(times)
        avg_time = total_time / iterations

        benchmark_result = BenchmarkResult(
            name=self.name,
            total_time_ms=total_time,
            iterations=iterations,
            time_per_iteration_ms=avg_time,
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated statistics across all benchmarks."""
        if not self.results:
            return {"total_benchmarks": 0}

        total_time = sum(r.total_time_ms for r in self.results)
        avg_time = sum(r.time_per_iteration_ms for r in self.results) / len(self.results)

        return {
            "total_benchmarks": len(self.results),
            "total_time_ms": total_time,
            "avg_time_per_iteration_ms": avg_time,
            "by_name": self._group_by_name(),
        }

    def _group_by_name(self) -> dict[str, dict[str, float]]:
        """Group results by benchmark name."""
        grouped: dict[str, list[float]] = {}
        for result in self.results:
            if result.name not in grouped:
                grouped[result.name] = []
            grouped[result.name].append(result.time_per_iteration_ms)

        return {
            name: {
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "count": len(times),
            }
            for name, times in grouped.items()
        }

    def reset(self) -> None:
        """Reset all results."""
        self.results.clear()


def benchmark_step(
    step_fn: callable,
    *args: Any,
    warmup: int = 5,
    iterations: int = 20,
    **kwargs: Any,
) -> float:
    """Benchmark a single training step.

    Args:
        step_fn: Function that performs one training step
        *args: Arguments to pass to step_fn
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        **kwargs: Keyword arguments to pass to step_fn

    Returns:
        Average milliseconds per step
    """
    # Warmup
    for _ in range(warmup):
        step_fn(*args, **kwargs)

    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        step_fn(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)
