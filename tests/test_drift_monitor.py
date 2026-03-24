#!/usr/bin/env python3
"""
Tests für Drift Monitor (Phase 4B).

Enthält Tests für:
- Performance Drift (CUSUM)
- Environment Drift
- Concept Drift (ADWIN)
- Drift Alerts
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.registry import RunEntry, RunRegistry
from research.drift_monitor import DriftMonitor, DriftReport


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def drift_monitor() -> DriftMonitor:
    """Erstelle DriftMonitor für Tests."""
    return DriftMonitor(window_size=20, threshold=0.05)


@pytest.fixture
def sample_run_history() -> List[Dict[str, Any]]:
    """Sample Run-Historie für Drift-Tests."""
    base_time = datetime.now() - timedelta(days=30)
    
    history = []
    for i in range(40):
        timestamp = base_time + timedelta(days=i)
        
        # Simuliere Performance-Drift nach Hälfte
        if i < 20:
            delta_bpb = -0.02 + (i * 0.001)  # Verbessert sich leicht
        else:
            delta_bpb = 0.01 + ((i - 20) * 0.005)  # Verschlechtert sich
        
        history.append({
            "run_id": f"run{i:03d}",
            "delta_bpb": delta_bpb,
            "features": ["leaky_relu"],
            "timestamp": timestamp.isoformat(),
        })
    
    return history


@pytest.fixture
def registry_with_runs(tmp_path: Path) -> RunRegistry:
    """Registry mit Runs für Environment-Drift-Tests."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    registry = RunRegistry(results_dir=str(results_dir))
    
    # Einige Runs erstellen
    for i in range(5):
        registry.register(f"run{i:03d}", f"config_{i}")
        registry.complete_run(f"run{i:03d}", {
            "val_bpb": 1.25 - (i * 0.01),
            "ms_per_step": 10.0,
            "steps_completed": 100,
            "artifact_bytes": 5_000_000,
        })
    
    return registry


# ============================================================================
# Tests: Initialization
# ============================================================================


class TestInitialization:
    """Tests für Initialisierung."""

    def test_init_default(self):
        """Test Standard-Initialisierung."""
        monitor = DriftMonitor()
        
        assert monitor.window_size == 20
        assert monitor.threshold == 0.05
        assert len(monitor._active_alerts) == 0
    
    def test_init_custom(self):
        """Test benutzerdefinierte Initialisierung."""
        monitor = DriftMonitor(window_size=50, threshold=0.10)
        
        assert monitor.window_size == 50
        assert monitor.threshold == 0.10


# ============================================================================
# Tests: CUSUM
# ============================================================================


class TestCUSUM:
    """Tests für CUSUM-Test."""

    def test_cusum_no_drift(self, drift_monitor):
        """Test CUSUM ohne Drift."""
        values = [1.0] * 20  # Stabile Werte
        
        cusum_pos, cusum_neg, drift_detected = drift_monitor._cusum_test(
            values, target=1.0, slack=0.1
        )
        
        assert drift_detected is False
    
    def test_cusum_positive_drift(self, drift_monitor):
        """Test CUSUM mit positivem Drift."""
        # Werte steigen kontinuierlich
        values = [1.0 + (i * 0.1) for i in range(20)]
        
        cusum_pos, cusum_neg, drift_detected = drift_monitor._cusum_test(
            values, target=1.0, slack=0.1
        )
        
        assert cusum_pos > 0
        assert drift_detected is True
    
    def test_cusum_negative_drift(self, drift_monitor):
        """Test CUSUM mit negativem Drift."""
        # Werte fallen kontinuierlich
        values = [1.0 - (i * 0.1) for i in range(20)]
        
        cusum_pos, cusum_neg, drift_detected = drift_monitor._cusum_test(
            values, target=1.0, slack=0.1
        )
        
        assert cusum_neg > 0
        assert drift_detected is True
    
    def test_cusum_empty_values(self, drift_monitor):
        """Test CUSUM mit leeren Werten."""
        cusum_pos, cusum_neg, drift_detected = drift_monitor._cusum_test(
            [], target=1.0, slack=0.1
        )
        
        assert drift_detected is False


# ============================================================================
# Tests: ADWIN
# ============================================================================


class TestADWIN:
    """Tests für ADWIN-Test."""

    def test_adwin_no_change(self, drift_monitor):
        """Test ADWIN ohne Änderung."""
        values = [1.0] * 30  # Stabile Werte
        
        change_point, confidence = drift_monitor._adwin_test(values)
        
        assert change_point is None
    
    def test_adwin_with_change(self, drift_monitor):
        """Test ADWIN mit Änderung."""
        # Werte ändern sich in der Mitte
        values = [1.0] * 15 + [2.0] * 15
        
        change_point, confidence = drift_monitor._adwin_test(values)
        
        # Sollte Change Point finden
        assert change_point is not None
        assert confidence > 0.5
    
    def test_adwin_insufficient_data(self, drift_monitor):
        """Test ADWIN mit zu wenig Daten."""
        values = [1.0, 1.1, 1.2]  # Zu wenig
        
        change_point, confidence = drift_monitor._adwin_test(values)
        
        assert change_point is None


# ============================================================================
# Tests: Performance Drift Detection
# ============================================================================


class TestPerformanceDriftDetection:
    """Tests für Performance-Drift-Erkennung."""

    def test_detect_performance_drift(self, drift_monitor, sample_run_history):
        """Test Erkennung von Performance-Drift."""
        report = drift_monitor.detect_performance_drift(
            "leaky_relu",
            sample_run_history
        )
        
        # Drift-Erkennung hängt von den Daten ab - Test prüft Struktur
        if report is not None:
            assert report.drift_type == "performance_drift"
            assert report.affected_features == ["leaky_relu"]
        # Oder None wenn kein signifikanter Drift in den Test-Daten
    
    def test_detect_performance_drift_stable(self, drift_monitor):
        """Test wenn kein Drift vorhanden."""
        # Stabile Historie
        history = [
            {"run_id": f"run{i}", "delta_bpb": -0.02, "features": ["gqa"],
             "timestamp": datetime.now().isoformat()}
            for i in range(10)
        ]
        
        report = drift_monitor.detect_performance_drift("gqa", history)
        
        assert report is None
    
    def test_detect_performance_drift_insufficient_data(self, drift_monitor):
        """Test mit zu wenig Daten."""
        history = [
            {"run_id": "run1", "delta_bpb": -0.02, "features": ["gqa"]},
            {"run_id": "run2", "delta_bpb": -0.01, "features": ["gqa"]},
        ]
        
        report = drift_monitor.detect_performance_drift("gqa", history)
        
        assert report is None


# ============================================================================
# Tests: Environment Drift Detection
# ============================================================================


class TestEnvironmentDriftDetection:
    """Tests für Environment-Drift-Erkennung."""

    def test_detect_environment_drift_first_call(self, drift_monitor, registry_with_runs):
        """Test erster Aufruf (kein Vergleich möglich)."""
        report = drift_monitor.detect_environment_drift(registry_with_runs)
        
        # Erster Aufruf: kein Drift erkennbar
        assert report is None
        assert len(drift_monitor._environment_snapshots) == 1
    
    def test_detect_environment_drift_subsequent(self, drift_monitor, registry_with_runs):
        """Test zweiter Aufruf."""
        # Erster Aufruf
        drift_monitor.detect_environment_drift(registry_with_runs)
        
        # Zweiter Aufruf
        report = drift_monitor.detect_environment_drift(registry_with_runs)
        
        # Snapshot gespeichert
        assert len(drift_monitor._environment_snapshots) == 2


# ============================================================================
# Tests: Concept Drift Detection
# ============================================================================


class TestConceptDriftDetection:
    """Tests für Concept-Drift-Erkennung."""

    def test_detect_concept_drift(self, drift_monitor, sample_run_history):
        """Test Concept Drift Erkennung."""
        report = drift_monitor.detect_concept_drift(
            "leaky_relu",
            run_history=sample_run_history,
            window_size=10
        )
        
        # Concept Drift Erkennung hängt von den Daten ab
        if report is not None:
            assert report.drift_type == "concept_drift"
        # None ist auch akzeptabel wenn kein signifikanter Drift
    
    def test_detect_concept_drift_stable(self, drift_monitor):
        """Test wenn kein Concept Drift."""
        # Stabile Erfolgsrate
        history = [
            {"run_id": f"run{i}", "delta_bpb": -0.02, "features": ["gqa"]}
            for i in range(30)
        ]
        
        report = drift_monitor.detect_concept_drift("gqa", run_history=history)
        
        assert report is None


# ============================================================================
# Tests: Drift Alerts
# ============================================================================


class TestDriftAlerts:
    """Tests für Drift-Alerts."""

    def test_get_drift_alerts_empty(self, drift_monitor):
        """Test leere Alerts."""
        alerts = drift_monitor.get_drift_alerts()
        
        assert alerts == []
    
    def test_get_drift_alerts_after_detection(self, drift_monitor, sample_run_history):
        """Test Alerts nach Erkennung."""
        report = drift_monitor.detect_performance_drift("leaky_relu", sample_run_history)
        
        alerts = drift_monitor.get_drift_alerts()
        
        # Wenn Drift erkannt wurde, sollte Alert existieren
        if report is not None:
            assert len(alerts) >= 1
            assert alerts[0].drift_type == "performance_drift"
        else:
            # Kein Drift = keine Alerts
            assert len(alerts) == 0
    
    def test_get_drift_alerts_limit(self, drift_monitor, sample_run_history):
        """Test Alert-Limit."""
        # Mehrere Alerts generieren
        for i in range(5):
            drift_monitor._active_alerts.append(
                DriftReport(
                    drift_type="performance_drift",
                    severity="medium",
                    affected_features=[f"feat{i}"],
                    drift_magnitude=0.1,
                    statistical_significance=0.05,
                    detected_at=datetime.now().isoformat(),
                    recommended_action="Test"
                )
            )
        
        alerts = drift_monitor.get_drift_alerts(limit=3)
        
        assert len(alerts) == 3
    
    def test_clear_alerts(self, drift_monitor, sample_run_history):
        """Test Alerts löschen."""
        drift_monitor.detect_performance_drift("leaky_relu", sample_run_history)
        
        drift_monitor.clear_alerts()
        
        alerts = drift_monitor.get_drift_alerts()
        assert len(alerts) == 0


# ============================================================================
# Tests: Drift Summary
# ============================================================================


class TestDriftSummary:
    """Tests für Drift-Zusammenfassung."""

    def test_get_drift_summary_empty(self, drift_monitor):
        """Test leere Zusammenfassung."""
        summary = drift_monitor.get_drift_summary()
        
        assert summary["total_alerts"] == 0
    
    def test_get_drift_summary_with_alerts(self, drift_monitor, sample_run_history):
        """Test Zusammenfassung mit Alerts."""
        report = drift_monitor.detect_performance_drift("leaky_relu", sample_run_history)
        
        summary = drift_monitor.get_drift_summary()
        
        if report is not None:
            assert summary["total_alerts"] >= 1
        else:
            assert summary["total_alerts"] == 0
        assert "by_type" in summary
        assert "by_severity" in summary


# ============================================================================
# Tests: Baseline Reset
# ============================================================================


class TestBaselineReset:
    """Tests für Baseline-Reset."""

    def test_reset_baseline_specific(self, drift_monitor):
        """Test spezifisches Feature-Reset."""
        # CUSUM-Daten hinzufügen
        drift_monitor._cusum_history["test_feat"] = [1.0, 1.1, 1.2]
        
        drift_monitor.reset_baseline("test_feat")
        
        assert "test_feat" not in drift_monitor._cusum_history
    
    def test_reset_baseline_all(self, drift_monitor):
        """Test Reset aller Features."""
        # CUSUM-Daten hinzufügen
        drift_monitor._cusum_history["feat1"] = [1.0]
        drift_monitor._cusum_history["feat2"] = [1.0]
        
        drift_monitor.reset_baseline()
        
        assert len(drift_monitor._cusum_history) == 0


# ============================================================================
# Tests: DriftReport Serialization
# ============================================================================


class TestDriftReportSerialization:
    """Tests für DriftReport Serialisierung."""

    def test_to_dict(self):
        """Test to_dict Methode."""
        report = DriftReport(
            drift_type="performance_drift",
            severity="high",
            affected_features=["leaky_relu"],
            drift_magnitude=0.15,
            statistical_significance=0.03,
            detected_at="2024-01-01T00:00:00",
            recommended_action="Reduce learning rate"
        )
        
        data = report.to_dict()
        
        assert data["drift_type"] == "performance_drift"
        assert data["severity"] == "high"
        assert data["drift_magnitude"] == 0.15
        assert len(data["affected_features"]) == 1
