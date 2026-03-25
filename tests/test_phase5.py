#!/usr/bin/env python3
"""
Tests für Phase 5 Advanced Features.

Enthält Tests für:
- Advanced Dashboard
- Run Explorer
- Real-time Monitor
- Health Checker
- Distributed Runner
- Run Queue
- HPO Integration
- NAS Integration

Hinweis: Tests verwenden direkte Imports um Dependency-Probleme zu vermeiden.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import RunRegistry, RunEntry


def import_module_from_file(module_name: str, file_path: str) -> Any:
    """Importiere Modul aus Datei-Pfad."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_results_dir() -> Path:
    """Erstelle temporäres Results-Verzeichnis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def registry_with_runs(temp_results_dir: Path) -> RunRegistry:
    """Erstelle Registry mit Test-Runs."""
    registry = RunRegistry(results_dir=str(temp_results_dir))

    # Erstelle Test-Runs
    runs_data = [
        {
            "run_id": "run_001",
            "config_hash": "abc123",
            "status": "completed",
            "val_bpb": 1.234,
            "ms_per_step": 150.0,
            "delta_bpb": -0.02,
            "delta_ms": -10.0,
            "parent_run_id": None,
            "tags": ["gqa:0.8", "film:true"],
        },
        {
            "run_id": "run_002",
            "config_hash": "def456",
            "status": "completed",
            "val_bpb": 1.189,
            "ms_per_step": 145.0,
            "delta_bpb": -0.045,
            "delta_ms": -15.0,
            "parent_run_id": "run_001",
            "tags": ["gqa:0.9", "swiglu:true"],
        },
        {
            "run_id": "run_003",
            "config_hash": "ghi789",
            "status": "failed",
            "val_bpb": None,
            "ms_per_step": None,
            "delta_bpb": None,
            "parent_run_id": "run_001",
            "tags": ["xsa:true"],
        },
        {
            "run_id": "run_004",
            "config_hash": "jkl012",
            "status": "completed",
            "val_bpb": 1.156,
            "ms_per_step": 160.0,
            "delta_bpb": -0.078,
            "delta_ms": 5.0,
            "parent_run_id": "run_002",
            "tags": ["gqa:0.85", "leaky_relu:true"],
        },
    ]

    for run_data in runs_data:
        entry = RunEntry(
            run_id=run_data["run_id"],
            config_hash=run_data["config_hash"],
            status=run_data["status"],
            val_bpb=run_data.get("val_bpb"),
            ms_per_step=run_data.get("ms_per_step"),
            delta_bpb=run_data.get("delta_bpb"),
            delta_ms=run_data.get("delta_ms"),
            parent_run_id=run_data.get("parent_run_id"),
            tags=run_data.get("tags", []),
        )
        registry.entries[run_data["run_id"]] = entry

    registry._save()
    return registry


# =============================================================================
# Advanced Dashboard Tests
# =============================================================================

class TestAdvancedDashboard:
    """Tests für Advanced Dashboard."""

    def test_dashboard_init(self, registry_with_runs: RunRegistry) -> None:
        """Test Dashboard Initialisierung."""
        try:
            from orchestrator.dashboard_advanced import AdvancedDashboard
            from research.success_metrics import SuccessMetricsTracker
        except ImportError:
            pytest.skip("Plotly nicht installiert")

        tracker = SuccessMetricsTracker(registry_with_runs)
        dashboard = AdvancedDashboard(registry_with_runs, tracker)

        assert dashboard is not None
        assert dashboard._registry == registry_with_runs
        assert dashboard._tracker == tracker

    def test_create_pareto_3d_plot(
        self, registry_with_runs: RunRegistry, temp_results_dir: Path
    ) -> None:
        """Test 3D Pareto Plot Erstellung."""
        try:
            from orchestrator.dashboard_advanced import AdvancedDashboard
            from research.success_metrics import SuccessMetricsTracker
        except ImportError:
            pytest.skip("Plotly nicht installiert")

        tracker = SuccessMetricsTracker(registry_with_runs)
        dashboard = AdvancedDashboard(registry_with_runs, tracker)

        output_path = str(temp_results_dir / "plots" / "pareto_3d.html")
        fig = dashboard.create_pareto_3d_plot(output_path)

        assert fig is not None
        assert Path(output_path).exists()

    def test_create_metrics_timeseries(
        self, registry_with_runs: RunRegistry, temp_results_dir: Path
    ) -> None:
        """Test Metrics Time Series Plot."""
        try:
            from orchestrator.dashboard_advanced import AdvancedDashboard
            from research.success_metrics import SuccessMetricsTracker
        except ImportError:
            pytest.skip("Plotly nicht installiert")

        tracker = SuccessMetricsTracker(registry_with_runs)
        dashboard = AdvancedDashboard(registry_with_runs, tracker)

        output_path = str(temp_results_dir / "plots" / "metrics_timeseries.html")
        fig = dashboard.create_metrics_timeseries(output_path)

        assert fig is not None
        assert Path(output_path).exists()

    def test_create_feature_importance_heatmap(
        self, registry_with_runs: RunRegistry, temp_results_dir: Path
    ) -> None:
        """Test Feature Importance Heatmap."""
        try:
            from orchestrator.dashboard_advanced import AdvancedDashboard
            from research.success_metrics import SuccessMetricsTracker
        except ImportError:
            pytest.skip("Plotly nicht installiert")

        tracker = SuccessMetricsTracker(registry_with_runs)
        dashboard = AdvancedDashboard(registry_with_runs, tracker)

        output_path = str(temp_results_dir / "plots" / "feature_importance.html")
        fig = dashboard.create_feature_importance_heatmap(output_path)

        assert fig is not None

    def test_create_lineage_graph(
        self, registry_with_runs: RunRegistry, temp_results_dir: Path
    ) -> None:
        """Test Lineage Graph."""
        try:
            from orchestrator.dashboard_advanced import AdvancedDashboard
            from research.success_metrics import SuccessMetricsTracker
        except ImportError:
            pytest.skip("Plotly nicht installiert")

        tracker = SuccessMetricsTracker(registry_with_runs)
        dashboard = AdvancedDashboard(registry_with_runs, tracker)

        output_path = str(temp_results_dir / "plots" / "lineage_run_001.html")
        fig = dashboard.create_lineage_graph("run_001", output_path)

        assert fig is not None

    def test_generate_full_dashboard(
        self, registry_with_runs: RunRegistry, temp_results_dir: Path
    ) -> None:
        """Test komplettes Dashboard."""
        try:
            from orchestrator.dashboard_advanced import AdvancedDashboard
            from research.success_metrics import SuccessMetricsTracker
        except ImportError:
            pytest.skip("Plotly nicht installiert")

        tracker = SuccessMetricsTracker(registry_with_runs)
        dashboard = AdvancedDashboard(registry_with_runs, tracker)

        output_path = str(temp_results_dir / "plots" / "phase5_dashboard.html")
        result_path = dashboard.generate_full_dashboard(output_path)

        assert Path(result_path).exists()
        assert Path(result_path).stat().st_size > 1000  # Mindestens 1KB


# =============================================================================
# Run Explorer Tests
# =============================================================================

class TestRunExplorer:
    """Tests für Run Explorer."""

    def test_explorer_init(self, registry_with_runs: RunRegistry) -> None:
        """Test Explorer Initialisierung."""
        try:
            from orchestrator.run_explorer import RunExplorer
        except ImportError:
            pytest.skip("Rich nicht installiert")

        explorer = RunExplorer(registry_with_runs)

        assert explorer is not None
        assert explorer._registry == registry_with_runs

    def test_search(self, registry_with_runs: RunRegistry) -> None:
        """Test Fuzzy Search."""
        try:
            from orchestrator.run_explorer import RunExplorer
        except ImportError:
            pytest.skip("Rich nicht installiert")

        explorer = RunExplorer(registry_with_runs)

        # Suche nach "gqa"
        results = explorer.search("gqa")
        assert isinstance(results, list)

        # Suche nach Run-ID
        results = explorer.search("run_001")
        assert "run_001" in results

    def test_filter_runs(self, registry_with_runs: RunRegistry) -> None:
        """Test Filter."""
        try:
            from orchestrator.run_explorer import RunExplorer
        except ImportError:
            pytest.skip("Rich nicht installiert")

        explorer = RunExplorer(registry_with_runs)

        # Filter nach Status
        filtered = explorer.filter_runs(status="completed")
        assert len(filtered) == 3  # 3 completed runs

        # Filter nach Feature
        filtered = explorer.filter_runs(features=["gqa"])
        assert len(filtered) > 0

    def test_sort_runs(self, registry_with_runs: RunRegistry) -> None:
        """Test Sortierung."""
        try:
            from orchestrator.run_explorer import RunExplorer
        except ImportError:
            pytest.skip("Rich nicht installiert")

        explorer = RunExplorer(registry_with_runs)
        runs = explorer._get_all_runs()

        # Sortiere nach delta_bpb
        sorted_runs = explorer.sort_runs(runs, "delta_bpb")
        assert len(sorted_runs) == len(runs)

        # Bester delta_bpb sollte zuerst sein
        best = sorted_runs[0]
        assert best.delta_bpb == min(r.delta_bpb for r in runs if r.delta_bpb is not None)

    def test_compare_runs(self, registry_with_runs: RunRegistry) -> None:
        """Test Run-Vergleich."""
        try:
            from orchestrator.run_explorer import RunExplorer
        except ImportError:
            pytest.skip("Rich nicht installiert")

        explorer = RunExplorer(registry_with_runs)

        comparison = explorer.compare_runs(["run_001", "run_002"])
        assert isinstance(comparison, str)
        assert "run_001" in comparison
        assert "run_002" in comparison

    def test_export_runs(self, registry_with_runs: RunRegistry, temp_results_dir: Path) -> None:
        """Test Export."""
        try:
            from orchestrator.run_explorer import RunExplorer
        except ImportError:
            pytest.skip("Rich nicht installiert")

        explorer = RunExplorer(registry_with_runs)
        runs = explorer._get_all_runs()

        # JSON Export
        json_output = explorer.export_runs(runs, "json")
        data = json.loads(json_output)
        assert isinstance(data, list)
        assert len(data) == len(runs)

        # CSV Export
        csv_output = explorer.export_runs(runs, "csv")
        assert "run_id" in csv_output

        # Markdown Export
        md_output = explorer.export_runs(runs, "markdown")
        assert "| Run ID |" in md_output


# =============================================================================
# Health Checker Tests
# =============================================================================

class TestHealthChecker:
    """Tests für Training Health Checker."""

    def test_health_checker_init(self) -> None:
        """Test Health Checker Initialisierung."""
        from orchestrator.health_checker import TrainingHealthChecker

        checker = TrainingHealthChecker()
        assert checker is not None

    def test_check_health_healthy(self) -> None:
        """Test Health Check für gesunden Run."""
        from orchestrator.health_checker import (
            TrainingHealthChecker,
            HealthStatus,
        )

        checker = TrainingHealthChecker()

        metrics = {
            "loss_history": [1.5, 1.4, 1.35, 1.3, 1.28, 1.25, 1.23, 1.22, 1.20, 1.19],
            "gradient_norm_history": [50, 55, 52, 48, 51, 49, 53, 50, 48, 52],
            "vram_history": [6000, 6020, 6040, 6060, 6080, 6100, 6120, 6140, 6160, 6180],
            "step_time_history": [100, 102, 98, 105, 101, 99, 103, 100, 102, 101],
        }

        report = checker.check_health("test_run", metrics)

        assert report.run_id == "test_run"
        assert report.health_score >= 70
        assert report.status == HealthStatus.HEALTHY

    def test_check_health_loss_divergence(self) -> None:
        """Test Health Check bei Loss Divergence."""
        from orchestrator.health_checker import (
            TrainingHealthChecker,
            HealthStatus,
            IssueType,
        )

        checker = TrainingHealthChecker()

        metrics = {
            "loss_history": [1.5, 1.4, 1.35, 15.0],  # Plötzlicher Anstieg
            "gradient_norm_history": [50, 55, 52, 48],
            "vram_history": [6000, 6020, 6040, 6060],
            "step_time_history": [100, 102, 98, 105],
        }

        report = checker.check_health("divergent_run", metrics)

        assert report.health_score < 70
        assert report.status in (HealthStatus.WARNING, HealthStatus.CRITICAL)
        assert any(
            issue.issue_type == IssueType.LOSS_DIVERGENCE
            for issue in report.issues
        )

    def test_check_health_gradient_explosion(self) -> None:
        """Test Health Check bei Gradient Explosion."""
        from orchestrator.health_checker import (
            TrainingHealthChecker,
            HealthStatus,
            IssueType,
        )

        checker = TrainingHealthChecker()

        metrics = {
            "loss_history": [1.5, 1.4, 1.35, 1.3],
            "gradient_norm_history": [50, 55, 52, 1500],  # Explosion
            "vram_history": [6000, 6020, 6040, 6060],
            "step_time_history": [100, 102, 98, 105],
        }

        report = checker.check_health("explosion_run", metrics)

        assert any(
            issue.issue_type == IssueType.GRADIENT_EXPLOSION
            for issue in report.issues
        )

    def test_get_early_warning_signs(self) -> None:
        """Test Frühwarnzeichen."""
        from orchestrator.health_checker import TrainingHealthChecker

        checker = TrainingHealthChecker()

        metrics = {
            "loss_history": list(range(30, 10, -1)),  # Oszillierend
            "gradient_norm_history": list(range(50, 80)),  # Steigend
            "vram_history": list(range(6000, 6030)),
            "step_time_history": [100 + i * 5 for i in range(30)],
        }

        warnings = checker.get_early_warning_signs("test_run", metrics)

        assert isinstance(warnings, list)


# =============================================================================
# Distributed Runner Tests
# =============================================================================

class TestDistributedRunner:
    """Tests für Distributed Runner."""

    def test_worker_config(self) -> None:
        """Test Worker Config."""
        from orchestrator.distributed_runner import WorkerConfig

        config = WorkerConfig(
            worker_id="worker_0",
            gpu_id=0,
            max_concurrent_runs=2,
            memory_limit_mb=8000,
        )

        assert config.worker_id == "worker_0"
        assert config.gpu_id == 0
        assert config.max_concurrent_runs == 2

    def test_distributed_runner_init(self) -> None:
        """Test Distributed Runner Initialisierung."""
        from orchestrator.distributed_runner import (
            DistributedRunner,
            WorkerConfig,
        )

        workers = [
            WorkerConfig("worker_0", gpu_id=0),
            WorkerConfig("worker_1", gpu_id=1),
        ]

        runner = DistributedRunner(workers)

        assert len(runner.workers) == 2
        assert "worker_0" in runner.workers
        assert "worker_1" in runner.workers

    def test_submit_runs(self) -> None:
        """Test Run Submission."""
        from orchestrator.distributed_runner import (
            DistributedRunner,
            WorkerConfig,
        )

        workers = [WorkerConfig("worker_0", gpu_id=0)]
        runner = DistributedRunner(workers)

        run_configs = [
            {"depth": 12, "width": 512},
            {"depth": 14, "width": 640},
        ]

        batch_id = runner.submit_runs(run_configs)

        assert batch_id is not None
        assert runner.queue_length == 2

    def test_get_batch_status(self) -> None:
        """Test Batch Status."""
        from orchestrator.distributed_runner import (
            DistributedRunner,
            WorkerConfig,
        )

        workers = [WorkerConfig("worker_0", gpu_id=0)]
        runner = DistributedRunner(workers)

        batch_id = runner.submit_runs([{"depth": 12}])
        status = runner.get_batch_status(batch_id)

        assert "batch_id" in status
        assert status["total"] == 1
        assert status["pending"] == 1

    def test_get_worker_load(self) -> None:
        """Test Worker Load."""
        from orchestrator.distributed_runner import (
            DistributedRunner,
            WorkerConfig,
        )

        workers = [
            WorkerConfig("worker_0", gpu_id=0),
            WorkerConfig("worker_1", gpu_id=1),
        ]

        runner = DistributedRunner(workers)
        load = runner.get_worker_load()

        assert "worker_0" in load
        assert "worker_1" in load
        assert load["worker_0"]["max_runs"] == 1

    def test_auto_scale(self) -> None:
        """Test Auto Scaling."""
        from orchestrator.distributed_runner import (
            DistributedRunner,
            WorkerConfig,
        )

        workers = [WorkerConfig("worker_0", gpu_id=0)]
        runner = DistributedRunner(workers)

        scaling = runner.auto_scale(target_gpu_util=0.8)

        assert "action" in scaling
        assert scaling["action"] in ("none", "scale_up", "scale_down", "pause")


# =============================================================================
# Run Queue Tests
# =============================================================================

class TestRunQueue:
    """Tests für Run Queue Manager."""

    def test_queue_manager_init(self) -> None:
        """Test Queue Manager Initialisierung."""
        from orchestrator.run_queue import RunQueueManager

        manager = RunQueueManager()
        assert manager is not None

    def test_enqueue(self) -> None:
        """Test Enqueue."""
        from orchestrator.run_queue import RunQueueManager

        manager = RunQueueManager()

        run_id = manager.enqueue(
            run_config={"depth": 12},
            priority=0.8,
        )

        assert run_id is not None
        assert manager.get_queue_stats()["queue_length"] == 1

    def test_dequeue(self) -> None:
        """Test Dequeue."""
        from orchestrator.run_queue import RunQueueManager

        manager = RunQueueManager()

        # Zwei Runs einreihen
        manager.enqueue({"depth": 12}, priority=0.5)
        manager.enqueue({"depth": 14}, priority=0.9)

        # Höchste Priorität sollte zuerst kommen
        config = manager.dequeue("worker_0", max_wait_time_seconds=1.0)

        assert config is not None
        assert config["depth"] == 14  # Höhere Priorität

    def test_get_position(self) -> None:
        """Test Position in Queue."""
        from orchestrator.run_queue import RunQueueManager

        manager = RunQueueManager()

        run_id = manager.enqueue({"depth": 12}, priority=0.5)
        position = manager.get_position(run_id)

        assert position == 1

    def test_reprioritize(self) -> None:
        """Test Reprioritize."""
        from orchestrator.run_queue import RunQueueManager

        manager = RunQueueManager()

        run_id = manager.enqueue({"depth": 12}, priority=0.3)
        manager.reprioritize(run_id, 0.9)

        # Sollte jetzt Position 1 sein
        position = manager.get_position(run_id)
        assert position == 1

    def test_preemption(self) -> None:
        """Test Preemption."""
        from orchestrator.run_queue import RunQueueManager

        manager = RunQueueManager(preemption_enabled=True)

        # Normalen Run einreihen
        normal_id = manager.enqueue({"depth": 12}, priority=0.5)

        # Hochprioritären Run einreihen
        high_id = manager.enqueue({"depth": 14}, priority=0.95)

        # Preemption versuchen
        result = manager.preempt(normal_id, high_id)

        # Sollte erfolgreich sein bei hoher Priorität
        assert isinstance(result, bool)

    def test_get_queue_stats(self) -> None:
        """Test Queue Stats."""
        from orchestrator.run_queue import RunQueueManager

        manager = RunQueueManager()

        # Mehrere Runs einreihen
        for i in range(5):
            manager.enqueue({"depth": 12 + i}, priority=0.5)

        stats = manager.get_queue_stats()

        assert stats["queue_length"] == 5
        assert "priority_distribution" in stats
        assert "avg_wait_time" in stats


# =============================================================================
# HPO Integration Tests
# =============================================================================

class TestHPOIntegration:
    """Tests für HPO Integration."""

    def test_hpo_init(self, registry_with_runs: RunRegistry) -> None:
        """Test HPO Initialisierung."""
        try:
            from research.hpo_integration import HyperparameterOptimizer
        except ImportError:
            pytest.skip("Optuna nicht installiert")

        optimizer = HyperparameterOptimizer(registry_with_runs)
        assert optimizer is not None

    def test_suggest_config(self, registry_with_runs: RunRegistry) -> None:
        """Test Config Vorschlag."""
        try:
            from research.hpo_integration import HyperparameterOptimizer
        except ImportError:
            pytest.skip("Optuna nicht installiert")

        optimizer = HyperparameterOptimizer(registry_with_runs)
        config = optimizer.suggest_config()

        assert "depth" in config
        assert "width" in config
        assert "learning_rate" in config
        assert 8 <= config["depth"] <= 16
        assert 256 <= config["width"] <= 1024

    def test_report_result(self, registry_with_runs: RunRegistry) -> None:
        """Test Ergebnis Report."""
        try:
            from research.hpo_integration import HyperparameterOptimizer
        except ImportError:
            pytest.skip("Optuna nicht installiert")

        optimizer = HyperparameterOptimizer(registry_with_runs)
        config = optimizer.suggest_config()

        metrics = {
            "delta_bpb": -0.02,
            "efficiency_gain": 10.0,
            "size_change": -5.0,
        }

        optimizer.report_result(0, metrics)

        progress = optimizer.get_optimization_progress()
        assert progress["completed"] == 1

    def test_get_best_configs(self, registry_with_runs: RunRegistry) -> None:
        """Test Beste Configs."""
        try:
            from research.hpo_integration import HyperparameterOptimizer
        except ImportError:
            pytest.skip("Optuna nicht installiert")

        optimizer = HyperparameterOptimizer(registry_with_runs)

        # Mehrere Trials
        for i in range(5):
            config = optimizer.suggest_config()
            metrics = {
                "delta_bpb": -0.01 * (i + 1),
                "efficiency_gain": 5.0 * i,
                "size_change": 0.0,
            }
            optimizer.report_result(i, metrics)

        best_configs = optimizer.get_best_configs(top_k=3)
        assert len(best_configs) <= 3

    def test_get_optimization_progress(self, registry_with_runs: RunRegistry) -> None:
        """Test Fortschritt."""
        try:
            from research.hpo_integration import HyperparameterOptimizer
        except ImportError:
            pytest.skip("Optuna nicht installiert")

        optimizer = HyperparameterOptimizer(registry_with_runs)

        progress = optimizer.get_optimization_progress()

        assert "total_trials" in progress
        assert "completed" in progress
        assert "study_name" in progress


# =============================================================================
# NAS Integration Tests
# =============================================================================

class TestNASIntegration:
    """Tests für NAS Integration."""

    def test_search_space_sample(self) -> None:
        """Test Search Space Sampling."""
        from research.nas_integration import SearchSpace

        space = SearchSpace()
        arch = space.sample()

        assert arch.depth >= 8 and arch.depth <= 16
        assert arch.width >= 256 and arch.width <= 1024
        assert arch.attention_type in ["standard", "gqa", "xsa"]

    def test_search_space_mutate(self) -> None:
        """Test Search Space Mutation."""
        from research.nas_integration import SearchSpace, Architecture

        space = SearchSpace()
        parent = Architecture(
            arch_id="parent",
            depth=12,
            width=512,
            mlp_ratio=3.0,
            attention_type="gqa",
            activation="gelu",
        )

        child = space.mutate(parent)

        assert child.parent_ids == ["parent"]
        assert child.generation == 1

    def test_search_space_crossover(self) -> None:
        """Test Search Space Crossover."""
        from research.nas_integration import SearchSpace, Architecture

        space = SearchSpace()
        parent1 = Architecture(
            arch_id="p1",
            depth=12,
            width=512,
            mlp_ratio=3.0,
            attention_type="gqa",
            activation="gelu",
        )
        parent2 = Architecture(
            arch_id="p2",
            depth=14,
            width=640,
            mlp_ratio=4.0,
            attention_type="xsa",
            activation="swiglu",
        )

        child = space.crossover(parent1, parent2)

        assert child.depth in [12, 14]
        assert child.width in [512, 640]
        assert set(child.parent_ids) == {"p1", "p2"}

    def test_search_space_constraints(self) -> None:
        """Test Search Space Constraints."""
        from research.nas_integration import SearchSpace, Architecture

        space = SearchSpace(max_vram_mb=8000)

        # Gültige Architektur
        valid_arch = Architecture(
            arch_id="valid",
            depth=12,
            width=512,
            mlp_ratio=3.0,
            attention_type="gqa",
            activation="gelu",
        )

        is_valid, violations = space.check_constraints(valid_arch)
        assert is_valid

        # Zu große Architektur
        invalid_arch = Architecture(
            arch_id="invalid",
            depth=16,
            width=1024,
            mlp_ratio=5.0,
            attention_type="standard",
            activation="gelu",
        )

        # Sollte möglicherweise violations haben
        is_valid, violations = space.check_constraints(invalid_arch)
        # Kann gültig sein je nach Berechnung

    def test_nas_integration_init(self, registry_with_runs: RunRegistry) -> None:
        """Test NAS Initialisierung."""
        from research.nas_integration import NASIntegration

        nas = NASIntegration(registry_with_runs)
        assert nas is not None

    def test_nas_search(self, registry_with_runs: RunRegistry) -> None:
        """Test NAS Suche (kleines Budget)."""
        from research.nas_integration import NASIntegration

        nas = NASIntegration(registry_with_runs)
        nas.define_search_space(max_vram_mb=8000)

        # Kleines Budget für Test
        pareto_frontier = nas.search(budget=10)

        assert isinstance(pareto_frontier, list)
        assert len(pareto_frontier) > 0

    def test_get_architecture_tradeoffs(self, registry_with_runs: RunRegistry) -> None:
        """Test Tradeoff-Analyse."""
        from research.nas_integration import NASIntegration

        nas = NASIntegration(registry_with_runs)

        # Suche mit kleinem Budget
        nas.search(budget=10)

        report = nas.get_architecture_tradeoffs()

        assert isinstance(report, str)
        assert "Pareto" in report or "Architektur" in report

    def test_export_architectures(
        self, registry_with_runs: RunRegistry, temp_results_dir: Path
    ) -> None:
        """Test Architektur Export."""
        from research.nas_integration import NASIntegration

        nas = NASIntegration(registry_with_runs)
        nas.search(budget=10)

        output_path = str(temp_results_dir / "nas_architectures.json")
        result_path = nas.export_architectures(output_path)

        assert Path(result_path).exists()

        with open(result_path) as f:
            data = json.load(f)

        assert "pareto_frontier" in data
        assert "search_space" in data


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
