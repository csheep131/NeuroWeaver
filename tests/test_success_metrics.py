#!/usr/bin/env python3
"""
Tests für Success Metrics Tracker (Phase 4 Evaluation).

15 Tests für:
- Search Efficiency Berechnung
- Failure Rate Reduction
- Pareto Frontier Expansion
- Human Time Saved
- Confidence Accuracy
- Report-Generierung
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from research.success_metrics import SuccessMetricsTracker, MetricDefinition
from core.registry import RunRegistry, RunEntry


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_results_dir():
    """Temporäres Verzeichnis für Test-Daten."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def registry(temp_results_dir) -> RunRegistry:
    """RunRegistry mit temporärem Verzeichnis."""
    return RunRegistry(results_dir=str(temp_results_dir))


@pytest.fixture
def registry_with_completed_runs(temp_results_dir) -> RunRegistry:
    """Registry mit abgeschlossenen Runs."""
    registry = RunRegistry(results_dir=str(temp_results_dir))

    # Erstelle Parent-Runs
    for i in range(5):
        parent_id = f"parent_{i:03d}"
        registry.register(
            run_id=parent_id,
            config_hash=f"config_{i}",
            seed=42,
        )
        registry.complete_run(parent_id, {
            "val_bpb": 1.50 - i * 0.05,
            "ms_per_step": 100 + i * 5,
            "steps_completed": 1000,
        })

    # Erstelle Child-Runs mit Verbesserungen
    for i in range(5):
        child_id = f"child_{i:03d}"
        registry.register(
            run_id=child_id,
            config_hash=f"config_{i}",
            parent_run_id=f"parent_{i:03d}",
            seed=42,
        )
        registry.complete_run(child_id, {
            "val_bpb": 1.45 - i * 0.05,  # Besser als Parent
            "ms_per_step": 95 + i * 5,
            "steps_completed": 1000,
        })

    return registry


@pytest.fixture
def registry_with_failures(temp_results_dir) -> RunRegistry:
    """Registry mit fehlgeschlagenen Runs."""
    registry = RunRegistry(results_dir=str(temp_results_dir))

    # Erstelle erfolgreiche Runs
    for i in range(30):
        run_id = f"success_{i:03d}"
        registry.register(run_id=run_id, config_hash=f"config_{i % 5}", seed=42)
        registry.complete_run(run_id, {
            "val_bpb": 1.50,
            "ms_per_step": 100,
            "steps_completed": 1000,
        })

    # Erstelle fehlgeschlagene Runs
    for i in range(10):
        run_id = f"fail_{i:03d}"
        registry.register(run_id=run_id, config_hash=f"config_{i % 5}", seed=42)
        registry.fail_run(run_id, notes="OOM" if i % 2 == 0 else "NaN")

    return registry


@pytest.fixture
def tracker(registry) -> SuccessMetricsTracker:
    """SuccessMetricsTracker mit Standard-Baselines."""
    return SuccessMetricsTracker(
        registry=registry,
        baseline_search_efficiency=100.0,
        baseline_failure_rate=0.20,
        baseline_human_time=10.0,
    )


# ============================================================================
# Tests: Initialization
# ============================================================================


class TestInitialization:
    """Tests für Initialisierung."""

    def test_init_default(self, registry):
        """Test Standard-Initialisierung."""
        tracker = SuccessMetricsTracker(registry)

        assert tracker.registry == registry
        assert tracker.baseline_search_efficiency == 100.0
        assert tracker.baseline_failure_rate == 0.20
        assert tracker.baseline_human_time == 10.0

    def test_init_custom_baselines(self, registry):
        """Test Initialisierung mit benutzerdefinierten Baselines."""
        tracker = SuccessMetricsTracker(
            registry=registry,
            baseline_search_efficiency=150.0,
            baseline_failure_rate=0.30,
            baseline_human_time=15.0,
        )

        assert tracker.baseline_search_efficiency == 150.0
        assert tracker.baseline_failure_rate == 0.30
        assert tracker.baseline_human_time == 15.0

    def test_targets_defined(self, registry):
        """Test dass Zielwerte definiert sind."""
        tracker = SuccessMetricsTracker(registry)

        assert "search_efficiency_improvement" in tracker.targets
        assert "failure_rate_reduction" in tracker.targets
        assert "pareto_expansion" in tracker.targets
        assert "human_time_saved" in tracker.targets
        assert "confidence_accuracy" in tracker.targets


# ============================================================================
# Tests: Search Efficiency
# ============================================================================


class TestSearchEfficiency:
    """Tests für Search Efficiency Berechnung."""

    def test_compute_search_efficiency_no_runs(self, registry):
        """Test mit keine Runs."""
        tracker = SuccessMetricsTracker(registry)
        result = tracker.compute_search_efficiency()

        assert result["runs_needed_autonomous"] == 0
        assert result["target_met"] is False
        assert "error" in result

    def test_compute_search_efficiency_with_improvements(self, registry_with_completed_runs):
        """Test mit verbesserten Runs."""
        tracker = SuccessMetricsTracker(registry_with_completed_runs)
        result = tracker.compute_search_efficiency(target_delta_bpb=-0.05)

        assert "runs_needed_autonomous" in result
        assert "runs_needed_manual" in result
        assert "improvement_percent" in result
        assert "target_met" in result

    def test_compute_search_efficiency_target_met(self, temp_results_dir):
        """Test dass Target erreicht werden kann."""
        registry = RunRegistry(results_dir=str(temp_results_dir))

        # Erstelle viele erfolgreiche Runs mit guter Verbesserung
        for i in range(20):
            parent_id = f"parent_{i:03d}"
            registry.register(run_id=parent_id, config_hash="config", seed=42)
            registry.complete_run(parent_id, {"val_bpb": 1.50, "ms_per_step": 100, "steps_completed": 1000})

            child_id = f"child_{i:03d}"
            registry.register(run_id=child_id, config_hash="config", parent_run_id=parent_id, seed=42)
            registry.complete_run(child_id, {
                "val_bpb": 1.40,  # ΔBPB = -0.10
                "ms_per_step": 95,
                "steps_completed": 1000,
            })

        tracker = SuccessMetricsTracker(registry, baseline_search_efficiency=100.0)
        result = tracker.compute_search_efficiency(target_delta_bpb=-0.05)

        # Sollte Target erreichen (deutlich weniger Runs als Baseline)
        assert result["runs_needed_autonomous"] < result["runs_needed_manual"]
        assert result["improvement_percent"] > 0


# ============================================================================
# Tests: Failure Rate Reduction
# ============================================================================


class TestFailureRateReduction:
    """Tests für Failure Rate Reduction."""

    def test_compute_failure_rate_empty(self, registry):
        """Test mit leerer Registry."""
        tracker = SuccessMetricsTracker(registry)
        result = tracker.compute_failure_rate_reduction()

        assert "failure_rate_before" in result
        assert "failure_rate_after" in result
        assert result["failure_rate_before"] == 0.0
        assert result["failure_rate_after"] == 0.0

    def test_compute_failure_rate_with_failures(self, registry_with_failures):
        """Test mit fehlgeschlagenen Runs."""
        tracker = SuccessMetricsTracker(registry_with_failures)
        result = tracker.compute_failure_rate_reduction()

        assert result["failure_rate_before"] >= 0
        assert result["failure_rate_after"] >= 0
        assert "reduction_percent" in result
        assert "target_met" in result

    def test_failure_rate_reduction_calculation(self, temp_results_dir):
        """Test korrekte Berechnung der Reduktion."""
        registry = RunRegistry(results_dir=str(temp_results_dir))

        # Ältere Runs mit hoher Failure-Rate
        for i in range(20):
            run_id = f"old_{i:03d}"
            registry.register(run_id=run_id, config_hash="old", seed=42)
            if i < 8:  # 40% Failure-Rate
                registry.fail_run(run_id, notes="OOM")
            else:
                registry.complete_run(run_id, {"val_bpb": 1.50, "ms_per_step": 100, "steps_completed": 1000})

        # Neuere Runs mit niedriger Failure-Rate
        for i in range(20):
            run_id = f"new_{i:03d}"
            registry.register(run_id=run_id, config_hash="new", seed=42)
            if i < 2:  # 10% Failure-Rate
                registry.fail_run(run_id, notes="OOM")
            else:
                registry.complete_run(run_id, {"val_bpb": 1.45, "ms_per_step": 95, "steps_completed": 1000})

        tracker = SuccessMetricsTracker(registry)
        result = tracker.compute_failure_rate_reduction()

        # Reduktion sollte positiv sein (von 40% auf 10% = 75% Reduktion)
        assert result["reduction_percent"] > 0


# ============================================================================
# Tests: Pareto Frontier Expansion
# ============================================================================


class TestParetoFrontierExpansion:
    """Tests für Pareto Frontier Expansion."""

    def test_compute_pareto_empty(self, registry):
        """Test mit leerer Registry."""
        tracker = SuccessMetricsTracker(registry)
        result = tracker.compute_pareto_frontier_expansion()

        assert result["frontier_volume_now"] == 0
        assert result["frontier_volume_before"] == 0
        assert result["expansion_percent"] == 0.0

    def test_compute_pareto_with_runs(self, registry_with_completed_runs):
        """Test mit Runs."""
        tracker = SuccessMetricsTracker(registry_with_completed_runs)
        result = tracker.compute_pareto_frontier_expansion()

        assert "frontier_volume_now" in result
        assert "frontier_volume_before" in result
        assert "expansion_percent" in result
        assert "pareto_points_now" in result
        assert "pareto_points_before" in result


# ============================================================================
# Tests: Human Time Saved
# ============================================================================


class TestHumanTimeSaved:
    """Tests für Human Time Saved."""

    def test_compute_human_time_empty(self, registry):
        """Test mit leerer Registry."""
        tracker = SuccessMetricsTracker(registry)
        result = tracker.compute_human_time_saved()

        assert "manual_hours_per_week" in result
        assert "autonomous_hours_per_week" in result
        assert "time_saved_percent" in result
        assert result["time_saved_percent"] == 100.0  # Keine Runs = 100% gespart

    def test_compute_human_time_with_runs(self, registry_with_completed_runs):
        """Test mit Runs."""
        tracker = SuccessMetricsTracker(registry_with_completed_runs)
        result = tracker.compute_human_time_saved(weeks=4)

        assert result["manual_hours_per_week"] > 0
        assert result["autonomous_hours_per_week"] > 0
        assert result["autonomous_hours_per_week"] < result["manual_hours_per_week"]
        assert result["time_saved_percent"] > 0

    def test_human_time_calculation(self, temp_results_dir):
        """Test korrekte Berechnung der Zeitersparnis."""
        registry = RunRegistry(results_dir=str(temp_results_dir))

        # Erstelle 100 Runs
        for i in range(100):
            run_id = f"run_{i:03d}"
            registry.register(run_id=run_id, config_hash="config", seed=42)
            registry.complete_run(run_id, {"val_bpb": 1.50, "ms_per_step": 100, "steps_completed": 1000})

        tracker = SuccessMetricsTracker(registry)
        result = tracker.compute_human_time_saved(weeks=4)

        # Manuell: 100 Runs * 5 Min = 500 Min = 8.33 Stunden / 4 Wochen = 2.08 Stunden/Woche
        # Autonom: 100 Runs * 0.5 Min = 50 Min = 0.83 Stunden / 4 Wochen = 0.21 Stunden/Woche
        # Ersparnis: ~90%

        assert result["time_saved_percent"] > 80  # Mindestens 80% Ersparnis


# ============================================================================
# Tests: Confidence Accuracy
# ============================================================================


class TestConfidenceAccuracy:
    """Tests für Confidence Accuracy."""

    def test_compute_confidence_empty(self, registry):
        """Test mit leerer Registry."""
        tracker = SuccessMetricsTracker(registry)
        result = tracker.compute_confidence_accuracy()

        assert result["predicted_confidence_avg"] == 0.0
        assert result["actual_success_rate"] == 0.0
        assert result["target_met"] is False

    def test_compute_confidence_with_runs(self, registry_with_completed_runs):
        """Test mit Runs (ohne Confidence-Tags)."""
        tracker = SuccessMetricsTracker(registry_with_completed_runs)
        result = tracker.compute_confidence_accuracy()

        # Sollte Fallback verwenden
        assert "actual_success_rate" in result
        assert "calibration_error" in result

    def test_confidence_with_tags(self, temp_results_dir):
        """Test mit Confidence-Tags."""
        registry = RunRegistry(results_dir=str(temp_results_dir))

        # Erstelle Runs mit Confidence-Tags
        for i in range(20):
            run_id = f"run_{i:03d}"
            tags = [f"confidence:{0.7 + i * 0.01}"]  # Confidence 0.70-0.89

            registry.register(run_id=run_id, config_hash="config", seed=42)
            registry.entries[run_id].tags = tags

            # 80% erfolgreich
            if i < 16:
                registry.complete_run(run_id, {"val_bpb": 1.45, "ms_per_step": 95, "steps_completed": 1000})
            else:
                registry.fail_run(run_id, notes="OOM")

        tracker = SuccessMetricsTracker(registry)
        result = tracker.compute_confidence_accuracy(min_confidence=0.6)

        assert result["samples"] > 0
        assert "predicted_confidence_avg" in result
        assert "actual_success_rate" in result
        assert "correlation" in result
        assert "accuracy" in result


# ============================================================================
# Tests: Get All Metrics
# ============================================================================


class TestGetAllMetrics:
    """Tests für get_all_metrics."""

    def test_get_all_metrics(self, registry_with_completed_runs):
        """Test dass alle Metriken zurückgegeben werden."""
        tracker = SuccessMetricsTracker(registry_with_completed_runs)
        metrics = tracker.get_all_metrics()

        assert len(metrics) == 5
        assert "search_efficiency" in metrics
        assert "failure_rate_reduction" in metrics
        assert "pareto_expansion" in metrics
        assert "human_time_saved" in metrics
        assert "confidence_accuracy" in metrics

    def test_metric_definition_fields(self, registry_with_completed_runs):
        """Test dass alle MetricDefinition-Felder gefüllt sind."""
        tracker = SuccessMetricsTracker(registry_with_completed_runs)
        metrics = tracker.get_all_metrics()

        for name, metric in metrics.items():
            assert isinstance(metric, MetricDefinition)
            assert metric.name is not None
            assert metric.description is not None
            assert metric.formula is not None
            assert metric.target_value is not None
            assert metric.direction in ("higher_better", "lower_better")
            assert metric.baseline_value is not None
            assert metric.current_value is not None
            assert metric.unit is not None
            assert isinstance(metric.target_met, bool)


# ============================================================================
# Tests: Report Generation
# ============================================================================


class TestReportGeneration:
    """Tests für Report-Generierung."""

    def test_generate_report(self, registry_with_completed_runs):
        """Test Report-Generierung."""
        tracker = SuccessMetricsTracker(registry_with_completed_runs)
        report = tracker.generate_report()

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Phase 4 Success Metrics Report" in report
        assert "Zusammenfassung" in report
        assert "Detaillierte Metriken" in report

    def test_generate_report_contains_metrics(self, registry_with_completed_runs):
        """Test dass Report alle Metriken enthält."""
        tracker = SuccessMetricsTracker(registry_with_completed_runs)
        report = tracker.generate_report()

        assert "Search Efficiency" in report
        assert "Failure Rate Reduction" in report
        assert "Pareto Frontier Expansion" in report
        assert "Human Time Saved" in report
        assert "Confidence Accuracy" in report

    def test_generate_report_contains_summary(self, registry_with_completed_runs):
        """Test dass Report Zusammenfassung enthält."""
        tracker = SuccessMetricsTracker(registry_with_completed_runs)
        report = tracker.generate_report()

        # Zusammenfassung sollte Anzahl erreichter Ziele enthalten
        assert "Ziele erreicht" in report or "erreicht" in report


# ============================================================================
# Tests: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests für Edge Cases."""

    def test_only_failed_runs(self, temp_results_dir):
        """Test mit nur fehlgeschlagenen Runs."""
        registry = RunRegistry(results_dir=str(temp_results_dir))

        for i in range(10):
            run_id = f"fail_{i:03d}"
            registry.register(run_id=run_id, config_hash="config", seed=42)
            registry.fail_run(run_id, notes="OOM")

        tracker = SuccessMetricsTracker(registry)
        metrics = tracker.get_all_metrics()

        # Sollte nicht crashen
        assert len(metrics) == 5

    def test_only_completed_runs_no_parents(self, temp_results_dir):
        """Test mit nur abgeschlossenen Runs ohne Parents."""
        registry = RunRegistry(results_dir=str(temp_results_dir))

        for i in range(10):
            run_id = f"run_{i:03d}"
            registry.register(run_id=run_id, config_hash="config", seed=42)
            registry.complete_run(run_id, {"val_bpb": 1.50, "ms_per_step": 100, "steps_completed": 1000})

        tracker = SuccessMetricsTracker(registry)

        # Search Efficiency sollte mit Fehler/Warning umgehen
        result = tracker.compute_search_efficiency()
        assert "error" in result or result["runs_needed_autonomous"] == 0

    def test_single_run(self, temp_results_dir):
        """Test mit nur einem Run."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        registry.register(run_id="single_run", config_hash="config", seed=42)
        registry.complete_run("single_run", {"val_bpb": 1.50, "ms_per_step": 100, "steps_completed": 1000})

        tracker = SuccessMetricsTracker(registry)
        metrics = tracker.get_all_metrics()

        # Sollte nicht crashen
        assert len(metrics) == 5
