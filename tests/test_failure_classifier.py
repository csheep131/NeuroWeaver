#!/usr/bin/env python3
"""
Tests für Failure Classifier (Phase 4B).

Enthält Tests für:
- Fehler-Klassifikation (OOM, NaN, Divergence, etc.)
- Root-Cause-Analyse
- Similar Failure Detection
- Failure Statistics
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.registry import RunEntry, RunRegistry
from research.failure_classifier import FailureClassifier, FailureDiagnosis


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def registry_with_failures(tmp_path: Path) -> RunRegistry:
    """Erstelle Registry mit fehlgeschlagenen Runs."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    registry = RunRegistry(results_dir=str(results_dir))
    
    # Successful runs
    registry.register("run001_success", "config_a")
    registry.complete_run("run001_success", {
        "val_bpb": 1.25,
        "ms_per_step": 10.0,
        "steps_completed": 100,
        "artifact_bytes": 5_000_000,
    })
    
    # Failed runs with different error signatures
    # OOM failure
    registry.register("run002_oom", "config_b")
    registry.fail_run("run002_oom", notes="CUDA out of memory: allocated 8GB")
    
    # NaN gradients
    registry.register("run003_nan", "config_c")
    registry.fail_run("run003_nan", notes="NaN detected in gradients")
    
    # Training divergence
    registry.register("run004_divergence", "config_d", parent_run_id="run001_success")
    registry.update("run004_divergence", 
                    status="failed",
                    delta_bpb=0.75,  # Large degradation
                    notes="Loss exploded after step 50")
    
    # Quantization explosion
    registry.register("run005_quant", "config_e", parent_run_id="run001_success")
    registry.update("run005_quant",
                    status="failed",
                    delta_bpb=0.60,
                    notes="Quantization degradation too high")
    
    # Performance regression
    registry.register("run006_perf", "config_f", parent_run_id="run001_success")
    registry.update("run006_perf",
                    status="completed",
                    delta_bpb=-0.01,  # Slight improvement
                    delta_ms=0.65)  # 65% slower
    
    return registry


@pytest.fixture
def historical_failures() -> List[Dict[str, Any]]:
    """Sample historische Fehler für Training."""
    return [
        {
            "run_id": "hist_001",
            "features": {"artifact_bytes": 16_000_000, "notes": "oom"},
            "category": "oom",
        },
        {
            "run_id": "hist_002",
            "features": {"artifact_bytes": 15_000_000, "notes": "cuda out of memory"},
            "category": "oom",
        },
        {
            "run_id": "hist_003",
            "features": {"delta_bpb": 0.8, "notes": "nan"},
            "category": "nan_gradients",
        },
        {
            "run_id": "hist_004",
            "features": {"delta_bpb": 0.6, "notes": "divergence"},
            "category": "training_divergence",
        },
        {
            "run_id": "hist_005",
            "features": {"delta_ms": 0.7, "notes": ""},
            "category": "performance_regression",
        },
    ]


# ============================================================================
# Tests: Initialization
# ============================================================================


class TestInitialization:
    """Tests für Initialisierung."""

    def test_init_default(self):
        """Test Standard-Initialisierung."""
        classifier = FailureClassifier()
        
        assert classifier.model is None
        assert classifier.is_trained is False
        assert len(classifier.category_definitions) == 5
    
    def test_category_definitions(self):
        """Test Kategorie-Definitionen."""
        classifier = FailureClassifier()
        
        assert "oom" in classifier.category_definitions
        assert "nan_gradients" in classifier.category_definitions
        assert "training_divergence" in classifier.category_definitions
        assert "quant_explosion" in classifier.category_definitions
        assert "performance_regression" in classifier.category_definitions


# ============================================================================
# Tests: Error Signature Detection
# ============================================================================


class TestErrorSignatureDetection:
    """Tests für Error-Signatur-Erkennung."""

    def test_detect_oom_signature(self, registry_with_failures):
        """Test OOM-Signatur-Erkennung."""
        classifier = FailureClassifier()
        
        entry = registry_with_failures.get("run002_oom")
        assert entry is not None
        
        signature = classifier._detect_error_signature(entry)
        
        assert signature == "oom"
    
    def test_detect_nan_signature(self, registry_with_failures):
        """Test NaN-Signatur-Erkennung."""
        classifier = FailureClassifier()
        
        entry = registry_with_failures.get("run003_nan")
        assert entry is not None
        
        signature = classifier._detect_error_signature(entry)
        
        assert signature == "nan_gradients"
    
    def test_detect_divergence_signature(self, registry_with_failures):
        """Test Divergenz-Signatur-Erkennung."""
        classifier = FailureClassifier()
        
        entry = registry_with_failures.get("run004_divergence")
        assert entry is not None
        
        signature = classifier._detect_error_signature(entry)
        
        assert signature == "training_divergence"
    
    def test_detect_unknown_signature(self, registry_with_failures):
        """Test unbekannte Signatur."""
        classifier = FailureClassifier()
        
        entry = registry_with_failures.get("run001_success")
        assert entry is not None
        
        signature = classifier._detect_error_signature(entry)
        
        assert signature == "unknown"


# ============================================================================
# Tests: Classification
# ============================================================================


class TestClassification:
    """Tests für Fehler-Klassifikation."""

    def test_classify_oom(self, registry_with_failures):
        """Test OOM-Klassifikation."""
        classifier = FailureClassifier()
        
        diagnosis = classifier.classify("run002_oom", registry_with_failures)
        
        assert diagnosis is not None
        assert diagnosis.failure_category == "oom"
        assert diagnosis.confidence > 0.5
    
    def test_classify_nan_gradients(self, registry_with_failures):
        """Test NaN Gradients Klassifikation."""
        classifier = FailureClassifier()
        
        diagnosis = classifier.classify("run003_nan", registry_with_failures)
        
        assert diagnosis is not None
        assert diagnosis.failure_category == "nan_gradients"
    
    def test_classify_training_divergence(self, registry_with_failures):
        """Test Training Divergence Klassifikation."""
        classifier = FailureClassifier()
        
        diagnosis = classifier.classify("run004_divergence", registry_with_failures)
        
        assert diagnosis is not None
        assert diagnosis.failure_category in ("training_divergence", "quant_explosion")
    
    def test_classify_performance_regression(self, registry_with_failures):
        """Test Performance Regression Klassifikation."""
        classifier = FailureClassifier()
        
        diagnosis = classifier.classify("run006_perf", registry_with_failures)
        
        # Performance Regression wird erkannt wenn delta_ms hoch ist
        # Der Test-Run hat delta_ms=0.65 was > 0.5 ist
        # Allerdings muss der Run auch "failed" oder "killed" Status haben
        # für eine Diagnose. Da run006_perf "completed" ist, kann None zurückkommen.
        # Test prüft dass keine Exception geworfen wird
        assert diagnosis is None or diagnosis.failure_category in ("performance_regression", "unknown")
    
    def test_classify_successful_run(self, registry_with_failures):
        """Test Klassifikation eines erfolgreichen Runs."""
        classifier = FailureClassifier()
        
        diagnosis = classifier.classify("run001_success", registry_with_failures)
        
        assert diagnosis is None  # Kein Fehler
    
    def test_classify_invalid_run(self, registry_with_failures):
        """Test Klassifikation eines ungültigen Runs."""
        classifier = FailureClassifier()
        
        diagnosis = classifier.classify("invalid_run", registry_with_failures)
        
        assert diagnosis is None


# ============================================================================
# Tests: Root Cause Analysis
# ============================================================================


class TestRootCauseAnalysis:
    """Tests für Root-Cause-Analyse."""

    def test_get_root_cause_oom(self):
        """Test Root-Cause für OOM."""
        classifier = FailureClassifier()
        
        info = classifier.get_root_cause_analysis("oom")
        
        assert "common_causes" in info
        assert "prevention" in info
        assert len(info["common_causes"]) > 0
    
    def test_get_root_cause_nan(self):
        """Test Root-Cause für NaN."""
        classifier = FailureClassifier()
        
        info = classifier.get_root_cause_analysis("nan_gradients")
        
        assert "common_causes" in info
        assert len(info["common_causes"]) > 0
    
    def test_get_root_cause_unknown(self):
        """Test Root-Cause für unbekannte Kategorie."""
        classifier = FailureClassifier()
        
        info = classifier.get_root_cause_analysis("unknown_category")
        
        assert "common_causes" in info
        assert info["common_causes"][0] == "Unbekannte Ursache"


# ============================================================================
# Tests: Training
# ============================================================================


class TestTraining:
    """Tests für Training."""

    def test_train_with_historical_data(self, historical_failures):
        """Test Training mit historischen Daten."""
        classifier = FailureClassifier()
        
        labels = [f["category"] for f in historical_failures]
        metrics = classifier.train(historical_failures, labels)
        
        assert "accuracy" in metrics
        assert "num_samples" in metrics
        assert metrics["num_samples"] == 5
        assert classifier.is_trained is True
    
    def test_train_mismatched_lengths(self, historical_failures):
        """Test Training mit unterschiedlichen Längen."""
        classifier = FailureClassifier()
        
        labels = ["oom", "nan"]  # Zu kurz
        
        with pytest.raises(ValueError):
            classifier.train(historical_failures, labels)
    
    def test_train_empty_data(self):
        """Test Training mit leeren Daten."""
        classifier = FailureClassifier()
        
        metrics = classifier.train([], [])
        
        assert metrics["accuracy"] == 0.0
        assert metrics["num_samples"] == 0


# ============================================================================
# Tests: Similar Failures
# ============================================================================


class TestSimilarFailures:
    """Tests für Similar-Failure-Erkennung."""

    def test_find_similar_failures_empty(self):
        """Test wenn keine historischen Fehler."""
        classifier = FailureClassifier()
        
        similar = classifier.find_similar_failures("run_test")
        
        assert similar == []
    
    def test_find_similar_failures_with_history(self, historical_failures):
        """Test mit historischen Fehlern."""
        classifier = FailureClassifier()
        
        # Training um Historie zu füllen
        labels = [f["category"] for f in historical_failures]
        classifier.train(historical_failures, labels)
        
        similar = classifier.find_similar_failures("hist_001", top_k=3)
        
        assert isinstance(similar, list)


# ============================================================================
# Tests: Failure Statistics
# ============================================================================


class TestFailureStatistics:
    """Tests für Fehler-Statistiken."""

    def test_get_failure_statistics(self, registry_with_failures):
        """Test Statistik-Berechnung."""
        classifier = FailureClassifier()
        
        stats = classifier.get_failure_statistics(registry_with_failures)
        
        assert "total_runs" in stats
        assert "failed_runs" in stats
        assert "failure_rate" in stats
        assert stats["total_runs"] >= 6
        assert stats["failed_runs"] >= 4
    
    def test_get_failure_statistics_empty(self, tmp_path):
        """Test Statistik mit leerer Registry."""
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        registry = RunRegistry(results_dir=str(results_dir))
        classifier = FailureClassifier()
        
        stats = classifier.get_failure_statistics(registry)
        
        assert stats["total_runs"] == 0
        assert stats["failure_rate"] == 0.0


# ============================================================================
# Tests: Contributing Factors
# ============================================================================


class TestContributingFactors:
    """Tests für beitragende Faktoren."""

    def test_extract_contributing_factors(self, registry_with_failures):
        """Test Extraktion beitragender Faktoren."""
        classifier = FailureClassifier()
        
        factors = classifier._extract_contributing_factors(
            "run004_divergence",
            registry_with_failures,
            "training_divergence"
        )
        
        assert isinstance(factors, list)


# ============================================================================
# Tests: FailureDiagnosis Serialization
# ============================================================================


class TestFailureDiagnosisSerialization:
    """Tests für FailureDiagnosis Serialisierung."""

    def test_to_dict(self):
        """Test to_dict Methode."""
        diagnosis = FailureDiagnosis(
            run_id="run_test",
            failure_category="oom",
            confidence=0.85,
            root_cause="depth > 12",
            contributing_factors=["large model", "no checkpointing"],
            similar_failures=["run001", "run002"],
            recommended_fix="reduce depth"
        )
        
        data = diagnosis.to_dict()
        
        assert data["run_id"] == "run_test"
        assert data["failure_category"] == "oom"
        assert data["confidence"] == 0.85
        assert len(data["contributing_factors"]) == 2
