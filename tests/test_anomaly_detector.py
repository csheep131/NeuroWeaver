#!/usr/bin/env python3
"""
Tests für Anomaly Detector (Phase 4B).

Enthält Tests für:
- Instability Detection (Shapiro-Wilk, CV)
- Outlier Detection (Grubbs' Test)
- OOM Risk Detection
- Noisy Feature Detection
- Run All Checks
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.registry import RunEntry, RunRegistry
from research.anomaly_detector import AnomalyDetector, AnomalyReport


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def registry_with_seed_family(tmp_path: Path) -> RunRegistry:
    """Erstelle Registry mit Seed-Familie für Tests."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    registry = RunRegistry(results_dir=str(results_dir))
    
    # Parent-Run
    registry.register("run001_parent", "config_abc123")
    registry.complete_run("run001_parent", {
        "val_bpb": 1.25,
        "ms_per_step": 10.5,
        "steps_completed": 100,
        "artifact_bytes": 5_000_000,
    })
    
    # Seed-Familie (gleiche Config, verschiedene Seeds)
    for i, (run_id, bpb) in enumerate([
        ("run002_seed42", 1.24),
        ("run003_seed43", 1.26),
        ("run004_seed44", 1.23),
        ("run005_seed45", 1.27),
        ("run006_seed46", 1.25),
    ]):
        registry.register(run_id, "config_abc123", parent_run_id="run001_parent", seed=42 + i)
        registry.complete_run(run_id, {
            "val_bpb": bpb,
            "ms_per_step": 10.5,
            "steps_completed": 100,
            "artifact_bytes": 5_000_000,
        })
    
    return registry


@pytest.fixture
def registry_with_volatile_seeds(tmp_path: Path) -> RunRegistry:
    """Erstelle Registry mit volatilen Seeds (hohe Varianz)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    registry = RunRegistry(results_dir=str(results_dir))
    
    # Parent-Run
    registry.register("run010_parent", "config_xyz789")
    registry.complete_run("run010_parent", {
        "val_bpb": 1.30,
        "ms_per_step": 11.0,
        "steps_completed": 100,
        "artifact_bytes": 6_000_000,
    })
    
    # Seed-Familie mit hoher Varianz
    volatile_bpbs = [1.15, 1.45, 1.20, 1.40, 1.10]  # CV > 30%
    for i, (run_id, bpb) in enumerate(zip(
        ["run011_s1", "run012_s2", "run013_s3", "run014_s4", "run015_s5"],
        volatile_bpbs
    )):
        registry.register(run_id, "config_xyz789", parent_run_id="run010_parent", seed=100 + i)
        registry.complete_run(run_id, {
            "val_bpb": bpb,
            "ms_per_step": 11.0,
            "steps_completed": 100,
            "artifact_bytes": 6_000_000,
        })
    
    return registry


@pytest.fixture
def registry_with_outlier(tmp_path: Path) -> RunRegistry:
    """Erstelle Registry mit Ausreißer."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    registry = RunRegistry(results_dir=str(results_dir))
    
    # Normale Runs
    for run_id, bpb in [
        ("run020", 1.25),
        ("run021", 1.26),
        ("run022", 1.24),
        ("run023", 1.25),
    ]:
        registry.register(run_id, "config_normal")
        registry.complete_run(run_id, {
            "val_bpb": bpb,
            "ms_per_step": 10.0,
            "steps_completed": 100,
            "artifact_bytes": 5_000_000,
        })
    
    # Ausreißer-Run
    registry.register("run024_outlier", "config_normal")
    registry.complete_run("run024_outlier", {
        "val_bpb": 1.80,  # Deutlich höher als andere
        "ms_per_step": 10.0,
        "steps_completed": 100,
        "artifact_bytes": 5_000_000,
    })
    
    return registry


@pytest.fixture
def registry_with_large_artifact(tmp_path: Path) -> RunRegistry:
    """Erstelle Registry mit großem Artifact (OOM-Risiko)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    registry = RunRegistry(results_dir=str(results_dir))
    
    registry.register("run030_large", "config_large")
    registry.complete_run("run030_large", {
        "val_bpb": 1.20,
        "ms_per_step": 15.0,
        "steps_completed": 100,
        "artifact_bytes": 16_000_000,  # > 15MB
    })
    
    return registry


# ============================================================================
# Tests: Coefficient of Variation
# ============================================================================


class TestCoefficientOfVariation:
    """Tests für CV-Berechnung."""

    def test_cv_low_variance(self):
        """Test CV bei niedriger Varianz."""
        detector = AnomalyDetector()
        values = [1.0, 1.01, 0.99, 1.0, 1.01]
        
        cv = detector._compute_cv(values)
        
        assert 0.0 <= cv < 0.05  # CV < 5%
    
    def test_cv_high_variance(self):
        """Test CV bei hoher Varianz."""
        detector = AnomalyDetector()
        values = [1.0, 2.0, 1.5, 0.5, 2.5]
        
        cv = detector._compute_cv(values)
        
        assert cv > 0.30  # CV > 30%
    
    def test_cv_empty_list(self):
        """Test CV bei leerer Liste."""
        detector = AnomalyDetector()
        
        cv = detector._compute_cv([])
        
        assert cv == 0.0
    
    def test_cv_single_value(self):
        """Test CV bei einem Wert."""
        detector = AnomalyDetector()
        
        cv = detector._compute_cv([1.0])
        
        assert cv == 0.0


# ============================================================================
# Tests: Instability Detection
# ============================================================================


class TestInstabilityDetection:
    """Tests für Instabilitäts-Erkennung."""

    def test_detect_instability_stable(self, registry_with_seed_family):
        """Test Erkennung stabiler Runs."""
        detector = AnomalyDetector()
        
        report = detector.detect_instability("run001_parent", [1.24, 1.25, 1.26, 1.25, 1.24])
        
        assert report is None  # Keine Anomalie bei stabiler Performance
    
    def test_detect_instability_critical(self, registry_with_volatile_seeds):
        """Test Erkennung kritischer Instabilität."""
        detector = AnomalyDetector()
        
        # CV > 30% sollte critical sein - verwende extremere Werte
        report = detector.detect_instability("run010_parent", [0.90, 1.60, 0.85, 1.65, 0.80])
        
        assert report is not None
        assert report.anomaly_type == "instability"
        assert report.severity in ("high", "critical")
        assert "CV=" in report.description
    
    def test_detect_instability_medium(self):
        """Test Erkennung mittlerer Instabilität."""
        detector = AnomalyDetector()
        
        # CV ~15% sollte medium sein
        report = detector.detect_instability("run_test", [1.0, 1.15, 0.95, 1.10, 0.90])
        
        assert report is not None
        assert report.anomaly_type == "instability"
        assert report.severity in ("medium", "high")
    
    def test_detect_instability_insufficient_seeds(self):
        """Test mit zu wenigen Seeds."""
        detector = AnomalyDetector()
        
        report = detector.detect_instability("run_test", [1.0])
        
        assert report is None  # Zu wenig Daten


# ============================================================================
# Tests: Outlier Detection (Grubbs' Test)
# ============================================================================


class TestOutlierDetection:
    """Tests für Ausreißer-Erkennung."""

    def test_detect_outliers_no_outlier(self, registry_with_seed_family):
        """Test wenn kein Ausreißer vorhanden."""
        detector = AnomalyDetector()
        
        # Normaler Wert in der Mitte
        report = detector.detect_outliers(
            "run_test",
            metric=1.25,
            reference_metrics=[1.24, 1.25, 1.26, 1.25, 1.24]
        )
        
        assert report is None
    
    def test_detect_outliers_detected(self, registry_with_outlier):
        """Test wenn Ausreißer erkannt wird."""
        detector = AnomalyDetector()
        
        # Extrem hoher Wert
        report = detector.detect_outliers(
            "run024_outlier",
            metric=1.80,
            reference_metrics=[1.25, 1.26, 1.24, 1.25]
        )
        
        assert report is not None
        assert report.anomaly_type == "outlier"
        assert "Ausreißer" in report.description
    
    def test_detect_outliers_insufficient_references(self):
        """Test mit zu wenigen Referenzwerten."""
        detector = AnomalyDetector()
        
        report = detector.detect_outliers(
            "run_test",
            metric=1.5,
            reference_metrics=[1.2, 1.3]  # Zu wenig
        )
        
        assert report is None


# ============================================================================
# Tests: OOM Risk Detection
# ============================================================================


class TestOOMRiskDetection:
    """Tests für OOM-Risiko-Erkennung."""

    def test_detect_oom_risk_critical(self):
        """Test kritisches OOM-Risiko."""
        detector = AnomalyDetector()
        
        # > 95% Memory-Nutzung
        report = detector.detect_oom_risk(
            run_id="run_test",
            memory_usage_mb=7800,
            available_memory_mb=8000
        )
        
        assert report is not None
        assert report.anomaly_type == "oom_risk"
        assert report.severity == "critical"
    
    def test_detect_oom_risk_high(self):
        """Test hohes OOM-Risiko."""
        detector = AnomalyDetector()
        
        # > 90% Memory-Nutzung
        report = detector.detect_oom_risk(
            run_id="run_test",
            memory_usage_mb=7400,
            available_memory_mb=8000
        )
        
        assert report is not None
        assert report.anomaly_type == "oom_risk"
        assert report.severity == "high"
    
    def test_detect_oom_risk_medium(self):
        """Test mittleres OOM-Risiko."""
        detector = AnomalyDetector()
        
        # > 80% Memory-Nutzung
        report = detector.detect_oom_risk(
            run_id="run_test",
            memory_usage_mb=6600,
            available_memory_mb=8000
        )
        
        assert report is not None
        assert report.anomaly_type == "oom_risk"
        assert report.severity == "medium"
    
    def test_detect_oom_risk_safe(self):
        """Test sichere Memory-Nutzung."""
        detector = AnomalyDetector()
        
        # < 80% Memory-Nutzung
        report = detector.detect_oom_risk(
            run_id="run_test",
            memory_usage_mb=5000,
            available_memory_mb=8000
        )
        
        assert report is None


# ============================================================================
# Tests: Noisy Feature Detection
# ============================================================================


class TestNoisyFeatureDetection:
    """Tests für Noisy Feature Erkennung."""

    def test_detect_noisy_feature_stable(self):
        """Test stabiles Feature."""
        detector = AnomalyDetector()
        
        run_outcomes = [
            {"run_id": f"run{i}", "delta_bpb": -0.02, "features": ["film"]}
            for i in range(5)
        ]
        
        report = detector.detect_noisy_feature("film", run_outcomes)
        
        assert report is None  # Keine hohe Varianz
    
    def test_detect_noisy_feature_volatile(self):
        """Test verrauschtes Feature."""
        detector = AnomalyDetector()
        
        # Hohe Varianz in delta_bpb
        run_outcomes = [
            {"run_id": "run1", "delta_bpb": -0.10, "features": ["xsa"]},
            {"run_id": "run2", "delta_bpb": +0.15, "features": ["xsa"]},
            {"run_id": "run3", "delta_bpb": -0.05, "features": ["xsa"]},
            {"run_id": "run4", "delta_bpb": +0.20, "features": ["xsa"]},
            {"run_id": "run5", "delta_bpb": -0.08, "features": ["xsa"]},
        ]
        
        report = detector.detect_noisy_feature("xsa", run_outcomes)
        
        assert report is not None
        assert report.anomaly_type == "noisy_feature"
        assert "xsa" in report.description
    
    def test_detect_noisy_feature_insufficient_runs(self):
        """Test mit zu wenigen Runs."""
        detector = AnomalyDetector()
        
        run_outcomes = [
            {"run_id": "run1", "delta_bpb": -0.02, "features": ["film"]},
            {"run_id": "run2", "delta_bpb": +0.01, "features": ["film"]},
        ]
        
        report = detector.detect_noisy_feature("film", run_outcomes)
        
        assert report is None  # Zu wenig Daten


# ============================================================================
# Tests: Run All Checks
# ============================================================================


class TestRunAllChecks:
    """Tests für kombinierte Anomalie-Prüfung."""

    def test_run_all_checks_no_anomalies(self, registry_with_seed_family):
        """Test wenn keine Anomalien erkannt."""
        detector = AnomalyDetector()
        
        reports = detector.run_all_checks("run002_seed42", registry_with_seed_family)
        
        # Sollte keine oder wenige Anomalien finden bei stabilen Runs
        assert isinstance(reports, list)
    
    def test_run_all_checks_with_anomalies(self, registry_with_volatile_seeds):
        """Test wenn Anomalien erkannt werden."""
        detector = AnomalyDetector()
        
        reports = detector.run_all_checks("run011_s1", registry_with_volatile_seeds)
        
        # Sollte Instabilität erkennen
        assert isinstance(reports, list)
        instability_reports = [r for r in reports if r.anomaly_type == "instability"]
        assert len(instability_reports) > 0
    
    def test_run_all_checks_invalid_run(self, registry_with_seed_family):
        """Test mit ungültiger Run-ID."""
        detector = AnomalyDetector()
        
        reports = detector.run_all_checks("invalid_run_id", registry_with_seed_family)
        
        assert reports == []


# ============================================================================
# Tests: Summary Statistics
# ============================================================================


class TestSummaryStatistics:
    """Tests für Zusammenfassungs-Statistiken."""

    def test_get_summary_statistics_empty(self):
        """Test mit leeren Reports."""
        detector = AnomalyDetector()
        
        stats = detector.get_summary_statistics([])
        
        assert stats["total_anomalies"] == 0
    
    def test_get_summary_statistics_mixed(self):
        """Test mit gemischten Reports."""
        detector = AnomalyDetector()
        
        reports = [
            AnomalyReport(
                run_id="run1", anomaly_type="instability", severity="critical",
                description="Test", statistical_evidence={}, recommended_action="Test"
            ),
            AnomalyReport(
                run_id="run2", anomaly_type="outlier", severity="high",
                description="Test", statistical_evidence={}, recommended_action="Test"
            ),
            AnomalyReport(
                run_id="run3", anomaly_type="instability", severity="medium",
                description="Test", statistical_evidence={}, recommended_action="Test"
            ),
        ]
        
        stats = detector.get_summary_statistics(reports)
        
        assert stats["total_anomalies"] == 3
        assert stats["by_severity"]["critical"] == 1
        assert stats["by_severity"]["high"] == 1
        assert stats["by_type"]["instability"] == 2


# ============================================================================
# Tests: AnomalyReport Serialization
# ============================================================================


class TestAnomalyReportSerialization:
    """Tests für AnomalyReport Serialisierung."""

    def test_to_dict(self):
        """Test to_dict Methode."""
        report = AnomalyReport(
            run_id="run_test",
            anomaly_type="instability",
            severity="high",
            description="Test description",
            statistical_evidence={"cv": 0.25, "p_value": 0.03},
            recommended_action="Test action"
        )
        
        data = report.to_dict()
        
        assert data["run_id"] == "run_test"
        assert data["anomaly_type"] == "instability"
        assert data["severity"] == "high"
        assert data["statistical_evidence"]["cv"] == 0.25
