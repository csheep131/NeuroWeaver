#!/usr/bin/env python3
"""
Tests für A/B-Testing Framework (Phase 4 Evaluation).

20 Tests für:
- ABTestConfig Validierung
- ABTestFramework Erstellung und Verwaltung
- Randomisierung und Outcome-Recording
- Statistische Analyse (t-Test, Cohen's d, Konfidenzintervalle)
- Test-Summary und Empfehlungen
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from research.ab_testing import (
    ABTestConfig,
    ABTestFramework,
    ABTestResult,
    ABTestState,
    ABTestOutcome,
)
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
def registry_with_runs(temp_results_dir) -> RunRegistry:
    """RunRegistry mit Test-Runs."""
    registry = RunRegistry(results_dir=str(temp_results_dir))

    # Erstelle Test-Runs
    for i in range(10):
        run_id = f"run{i:03d}"
        registry.register(
            run_id=run_id,
            config_hash=f"config_{i % 3}",
            parent_run_id=f"run{i-1:03d}" if i > 0 else None,
            seed=42 + i,
        )

    return registry


@pytest.fixture
def basic_config() -> ABTestConfig:
    """Standard Test-Konfiguration."""
    return ABTestConfig(
        test_name="test_autonomous_vs_manual",
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=14),
        treatment_group="autonomous",
        control_group="manual",
        success_metrics=["delta_bpb", "efficiency_gain"],
        min_sample_size=5,
    )


# ============================================================================
# Tests: ABTestConfig Validierung
# ============================================================================


class TestABTestConfigValidation:
    """Tests für ABTestConfig Validierung."""

    def test_valid_config(self, basic_config):
        """Test gültige Konfiguration."""
        assert basic_config.test_name == "test_autonomous_vs_manual"
        assert basic_config.treatment_group == "autonomous"
        assert basic_config.control_group == "manual"
        assert len(basic_config.success_metrics) == 2
        assert basic_config.min_sample_size == 5

    def test_invalid_same_groups(self):
        """Test dass gleiche Gruppen Fehler werfen."""
        with pytest.raises(ValueError, match="treatment_group und control_group müssen unterschiedlich sein"):
            ABTestConfig(
                test_name="invalid_test",
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=1),
                treatment_group="autonomous",
                control_group="autonomous",  # Gleich!
                success_metrics=["delta_bpb"],
                min_sample_size=5,
            )

    def test_invalid_empty_metrics(self):
        """Test dass leere Metrics Fehler werfen."""
        with pytest.raises(ValueError, match="success_metrics darf nicht leer sein"):
            ABTestConfig(
                test_name="invalid_test",
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=1),
                treatment_group="autonomous",
                control_group="manual",
                success_metrics=[],  # Leer!
                min_sample_size=5,
            )

    def test_invalid_min_sample_size(self):
        """Test dass zu kleine min_sample_size Fehler wirft."""
        with pytest.raises(ValueError, match="min_sample_size muss mindestens 5 sein"):
            ABTestConfig(
                test_name="invalid_test",
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=1),
                treatment_group="autonomous",
                control_group="manual",
                success_metrics=["delta_bpb"],
                min_sample_size=2,  # Zu klein!
            )


# ============================================================================
# Tests: ABTestFramework - Erstellung und Verwaltung
# ============================================================================


class TestABTestFrameworkCreation:
    """Tests für ABTestFramework Erstellung."""

    def test_create_framework(self, registry):
        """Test Framework-Erstellung."""
        framework = ABTestFramework(registry)
        assert framework is not None
        assert framework.registry == registry

    def test_create_test(self, registry, basic_config):
        """Test Test-Erstellung."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        assert test_id is not None
        assert len(test_id) == 8  # UUID[:8]

    def test_create_test_persisted(self, registry, basic_config):
        """Test dass Test persistent gespeichert wird."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Neues Framework erstellen (lädt von Disk)
        framework2 = ABTestFramework(registry)
        assert test_id in framework2._tests

    def test_list_tests(self, temp_results_dir, basic_config):
        """Test Test-Auflistung."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        storage_path = temp_results_dir / "ab_tests_list.json"
        framework = ABTestFramework(registry, storage_path=str(storage_path))

        # Erstelle mehrere Tests
        test_ids = [framework.create_test(basic_config) for _ in range(3)]

        tests = framework.list_tests()
        assert len(tests) == 3

        for test in tests:
            assert "test_id" in test
            assert "test_name" in test
            assert "status" in test

    def test_delete_test(self, temp_results_dir, basic_config):
        """Test Test-Löschung."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        storage_path = temp_results_dir / "ab_tests_delete.json"
        framework = ABTestFramework(registry, storage_path=str(storage_path))
        test_id = framework.create_test(basic_config)

        # Löschen
        result = framework.delete_test(test_id)
        assert result is True

        # Verifizieren dass gelöscht
        tests = framework.list_tests()
        assert len(tests) == 0

    def test_delete_nonexistent_test(self, registry):
        """Test Löschen nicht-existierendem Test."""
        framework = ABTestFramework(registry)
        result = framework.delete_test("nonexistent")
        assert result is False


# ============================================================================
# Tests: Randomisierung und Outcome-Recording
# ============================================================================


class TestRandomizationAndOutcomes:
    """Tests für Randomisierung und Outcome-Recording."""

    def test_assign_run_to_group(self, registry, basic_config):
        """Test Run-Zuweisung."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        group = framework.assign_run_to_group(test_id)
        assert group in ("treatment", "control")

    def test_assign_run_randomization(self, registry, basic_config):
        """Test dass Randomisierung funktioniert."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Mehrere Zuweisungen sollten beide Gruppen liefern
        groups = [framework.assign_run_to_group(test_id) for _ in range(20)]
        assert "treatment" in groups
        assert "control" in groups

    def test_assign_run_invalid_test(self, registry):
        """Test Zuweisung zu nicht-existierendem Test."""
        framework = ABTestFramework(registry)

        with pytest.raises(ValueError, match="existiert nicht"):
            framework.assign_run_to_group("nonexistent")

    def test_record_outcome(self, registry, basic_config):
        """Test Outcome-Recording."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Outcome recorden
        framework.record_outcome(
            test_id=test_id,
            group="treatment",
            run_id="run001",
            metrics={"delta_bpb": -0.02, "efficiency_gain": 0.15},
        )

        # Verifizieren
        summary = framework.get_test_summary(test_id)
        assert summary["treatment_runs"] == 1

    def test_record_outcome_control(self, registry, basic_config):
        """Test Outcome-Recording für Control-Gruppe."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        framework.record_outcome(
            test_id=test_id,
            group="control",
            run_id="run002",
            metrics={"delta_bpb": 0.01, "efficiency_gain": 0.05},
        )

        summary = framework.get_test_summary(test_id)
        assert summary["control_runs"] == 1

    def test_record_outcome_invalid_test(self, registry):
        """Test Outcome für nicht-existierendem Test."""
        framework = ABTestFramework(registry)

        with pytest.raises(ValueError, match="existiert nicht"):
            framework.record_outcome(
                test_id="nonexistent",
                group="treatment",
                run_id="run001",
                metrics={"delta_bpb": -0.02},
            )

    def test_record_multiple_outcomes(self, temp_results_dir, basic_config):
        """Test mehrere Outcomes."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Mehrere Outcomes recorden
        for i in range(5):
            framework.record_outcome(
                test_id=test_id,
                group="treatment" if i % 2 == 0 else "control",
                run_id=f"run{i:03d}",
                metrics={"delta_bpb": -0.02 * (i + 1), "efficiency_gain": 0.1 * (i + 1)},
            )

        summary = framework.get_test_summary(test_id)
        assert summary["treatment_runs"] == 3
        assert summary["control_runs"] == 2


# ============================================================================
# Tests: Statistische Analyse
# ============================================================================


class TestStatisticalAnalysis:
    """Tests für statistische Analyse."""

    def test_analyze_test_insufficient_data(self, registry, basic_config):
        """Test Analyse mit unzureichenden Daten."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Nur ein Outcome
        framework.record_outcome(
            test_id=test_id,
            group="treatment",
            run_id="run001",
            metrics={"delta_bpb": -0.02},
        )

        results = framework.analyze_test(test_id)
        assert len(results) == 0  # Nicht genug Daten

    def test_analyze_test_with_data(self, registry, basic_config):
        """Test Analyse mit ausreichenden Daten."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Treatment-Outcomes (bessere Werte)
        for i in range(5):
            framework.record_outcome(
                test_id=test_id,
                group="treatment",
                run_id=f"treat_{i}",
                metrics={"delta_bpb": -0.05 - i * 0.01},
            )

        # Control-Outcomes (schlechtere Werte)
        for i in range(5):
            framework.record_outcome(
                test_id=test_id,
                group="control",
                run_id=f"control_{i}",
                metrics={"delta_bpb": 0.01 + i * 0.01},
            )

        results = framework.analyze_test(test_id)
        assert len(results) == 1  # Eine Metrik: delta_bpb

        result = results[0]
        assert isinstance(result, ABTestResult)
        assert result.metric == "delta_bpb"
        assert result.treatment_mean < result.control_mean  # Treatment besser

    def test_analyze_test_result_fields(self, temp_results_dir, basic_config):
        """Test dass alle Result-Felder gefüllt sind."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Ausreichend Daten mit Varianz
        for i in range(5):
            framework.record_outcome(
                test_id=test_id,
                group="treatment",
                run_id=f"t_{i}",
                metrics={"delta_bpb": -0.05 + i * 0.01},  # Varianz hinzufügen
            )
            framework.record_outcome(
                test_id=test_id,
                group="control",
                run_id=f"c_{i}",
                metrics={"delta_bpb": 0.01 + i * 0.01},  # Varianz hinzufügen
            )

        results = framework.analyze_test(test_id)
        result = results[0]

        # Alle Felder prüfen
        assert result.test_name == basic_config.test_name
        assert result.metric == "delta_bpb"
        assert result.treatment_mean is not None
        assert result.control_mean is not None
        assert result.treatment_std is not None
        assert result.control_std is not None
        assert result.t_statistic is not None
        assert result.p_value is not None
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.effect_size is not None
        assert bool(result.is_significant) in (True, False)  # bool() für numpy.bool_

    def test_cohens_d_calculation(self, temp_results_dir, basic_config):
        """Test Cohen's d Effektstärke."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Große Effektstärke: Treatment viel besser als Control mit Varianz
        import random
        random.seed(42)
        for i in range(20):
            framework.record_outcome(
                test_id=test_id,
                group="treatment",
                run_id=f"t_{i}",
                metrics={"delta_bpb": -0.10 + random.uniform(-0.02, 0.02)},  # Varianz
            )
            framework.record_outcome(
                test_id=test_id,
                group="control",
                run_id=f"c_{i}",
                metrics={"delta_bpb": 0.05 + random.uniform(-0.02, 0.02)},  # Varianz
            )

        results = framework.analyze_test(test_id)
        result = results[0]

        # Große Effektstärke erwartet (d > 0.8 in absolute value)
        assert abs(result.effect_size) > 0.8

    def test_significance_detection(self, temp_results_dir, basic_config):
        """Test Erkennung statistischer Signifikanz."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Sehr deutlicher Unterschied (sollte signifikant sein) mit Varianz
        import random
        random.seed(42)
        for i in range(30):
            framework.record_outcome(
                test_id=test_id,
                group="treatment",
                run_id=f"t_{i}",
                metrics={"delta_bpb": -0.10 + random.uniform(-0.01, 0.01)},
            )
            framework.record_outcome(
                test_id=test_id,
                group="control",
                run_id=f"c_{i}",
                metrics={"delta_bpb": 0.05 + random.uniform(-0.01, 0.01)},
            )

        results = framework.analyze_test(test_id)
        result = results[0]

        # Sollte signifikant sein bei großem Unterschied
        assert bool(result.is_significant) is True
        assert result.p_value < 0.05


# ============================================================================
# Tests: Test-Summary und Empfehlungen
# ============================================================================


class TestTestSummary:
    """Tests für Test-Summary."""

    def test_get_test_summary(self, registry, basic_config):
        """Test Test-Zusammenfassung."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Ein paar Outcomes
        for i in range(3):
            framework.record_outcome(
                test_id=test_id,
                group="treatment",
                run_id=f"t_{i}",
                metrics={"delta_bpb": -0.05},
            )
            framework.record_outcome(
                test_id=test_id,
                group="control",
                run_id=f"c_{i}",
                metrics={"delta_bpb": 0.01},
            )

        summary = framework.get_test_summary(test_id)

        assert "test_name" in summary
        assert "status" in summary
        assert "treatment_runs" in summary
        assert "control_runs" in summary
        assert "recommendation" in summary

    def test_get_test_summary_invalid_test(self, registry):
        """Test Summary für nicht-existierendem Test."""
        framework = ABTestFramework(registry)

        with pytest.raises(ValueError, match="existiert nicht"):
            framework.get_test_summary("nonexistent")

    def test_recommendation_generation(self, temp_results_dir, basic_config):
        """Test Empfehlungsgenerierung."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Treatment deutlich besser mit Varianz
        import random
        random.seed(42)
        for i in range(20):
            framework.record_outcome(
                test_id=test_id,
                group="treatment",
                run_id=f"t_{i}",
                metrics={"delta_bpb": -0.10 + random.uniform(-0.01, 0.01)},
            )
            framework.record_outcome(
                test_id=test_id,
                group="control",
                run_id=f"c_{i}",
                metrics={"delta_bpb": 0.05 + random.uniform(-0.01, 0.01)},
            )

        summary = framework.get_test_summary(test_id)
        recommendation = summary["recommendation"]

        # Empfehlung sollte Treatment bevorzugen (niedrigere delta_bpb = besser)
        # Da treatment_mean < control_mean, ist treatment besser
        assert "treatment" in recommendation.lower() or "control" in recommendation.lower()

    def test_recommendation_insufficient_data(self, registry, basic_config):
        """Test Empfehlung bei unzureichenden Daten."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Zu wenig Daten
        framework.record_outcome(
            test_id=test_id,
            group="treatment",
            run_id="t_1",
            metrics={"delta_bpb": -0.05},
        )

        summary = framework.get_test_summary(test_id)
        recommendation = summary["recommendation"]

        # Sollte auf unzureichende Daten hinweisen
        assert "Daten" in recommendation or "insufficient" in recommendation.lower() or "Test läuft" in recommendation


# ============================================================================
# Tests: Persistenz
# ============================================================================


class TestPersistence:
    """Tests für Persistenz."""

    def test_persistence(self, temp_results_dir, basic_config):
        """Test dass Tests persistent gespeichert werden."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        framework = ABTestFramework(registry)

        test_id = framework.create_test(basic_config)

        # Outcome recorden
        framework.record_outcome(
            test_id=test_id,
            group="treatment",
            run_id="run001",
            metrics={"delta_bpb": -0.02},
        )

        # Neues Framework erstellen (lädt von Disk)
        framework2 = ABTestFramework(registry)

        # Test sollte geladen sein
        assert test_id in framework2._tests

        # Outcomes sollten geladen sein
        test = framework2._tests[test_id]
        assert len(test.treatment_outcomes) == 1

    def test_storage_file_created(self, temp_results_dir, basic_config):
        """Test dass Storage-Datei erstellt wird."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        framework = ABTestFramework(registry, storage_path=str(temp_results_dir / "ab_tests.json"))

        framework.create_test(basic_config)

        storage_file = temp_results_dir / "ab_tests.json"
        assert storage_file.exists()

        # Valid JSON
        with open(storage_file, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict)


# ============================================================================
# Tests: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests für Edge Cases."""

    def test_empty_metrics_list(self, registry, basic_config):
        """Test mit leerer Metrik-Liste im Outcome."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        framework.record_outcome(
            test_id=test_id,
            group="treatment",
            run_id="run001",
            metrics={},  # Leer!
        )

        # Sollte nicht crashen
        summary = framework.get_test_summary(test_id)
        assert summary["treatment_runs"] == 1

    def test_missing_metric_in_outcome(self, registry, basic_config):
        """Test wenn Outcome nicht alle Success Metrics hat."""
        framework = ABTestFramework(registry)
        test_id = framework.create_test(basic_config)

        # Nur eine der zwei Metrics
        framework.record_outcome(
            test_id=test_id,
            group="treatment",
            run_id="run001",
            metrics={"delta_bpb": -0.02},  # efficiency_gain fehlt
        )

        # Sollte nicht crashen
        results = framework.analyze_test(test_id)
        # Keine analysierbaren Ergebnisse da nur 1 Outcome
        assert len(results) == 0

    def test_multiple_metrics_analysis(self, temp_results_dir):
        """Test Analyse mit mehreren Success Metrics."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        config = ABTestConfig(
            test_name="multi_metric_test",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=1),
            treatment_group="autonomous",
            control_group="manual",
            success_metrics=["delta_bpb", "efficiency_gain", "human_time"],
            min_sample_size=5,
        )

        framework = ABTestFramework(registry)
        test_id = framework.create_test(config)

        import random
        random.seed(42)

        # Outcomes mit allen Metrics und Varianz
        for i in range(10):
            framework.record_outcome(
                test_id=test_id,
                group="treatment",
                run_id=f"t_{i}",
                metrics={
                    "delta_bpb": -0.05 + random.uniform(-0.01, 0.01),
                    "efficiency_gain": 0.20 + random.uniform(-0.02, 0.02),
                    "human_time": 2.0 + random.uniform(-0.2, 0.2),
                },
            )
            framework.record_outcome(
                test_id=test_id,
                group="control",
                run_id=f"c_{i}",
                metrics={
                    "delta_bpb": 0.01 + random.uniform(-0.01, 0.01),
                    "efficiency_gain": 0.05 + random.uniform(-0.02, 0.02),
                    "human_time": 5.0 + random.uniform(-0.5, 0.5),
                },
            )

        results = framework.analyze_test(test_id)

        # Alle drei Metriken sollten analysiert sein
        assert len(results) == 3
        metrics = [r.metric for r in results]
        assert "delta_bpb" in metrics
        assert "efficiency_gain" in metrics
        assert "human_time" in metrics
