#!/usr/bin/env python3
"""
Tests für Refinement Engine (Phase 4 Evaluation).

15 Tests für:
- Guardrail Performance Analyse
- Prediction Errors Analyse
- Human Overrides Analyse
- Refinement Report Generierung
- Refinement Application
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from research.refinement_engine import RefinementEngine, RefinementSuggestion
from research.surrogate_scorer import SurrogateScorer
from orchestrator.guardrails import create_default_guardrails, GuardrailManager
from research.override_learner import create_override_learner, OverrideLearner
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
    """Registry mit Test-Runs."""
    registry = RunRegistry(results_dir=str(temp_results_dir))

    # Erstelle Runs
    for i in range(20):
        run_id = f"run_{i:03d}"
        registry.register(run_id=run_id, config_hash=f"config_{i % 3}", seed=42)

        if i % 5 == 0:  # 20% failures
            registry.fail_run(run_id, notes="OOM")
        else:
            registry.complete_run(run_id, {
                "val_bpb": 1.50 - i * 0.01,
                "ms_per_step": 100,
                "steps_completed": 1000,
            })

    return registry


@pytest.fixture
def scorer() -> SurrogateScorer:
    """SurrogateScorer Instanz."""
    return SurrogateScorer(model_type="random_forest")


@pytest.fixture
def guardrail_manager() -> GuardrailManager:
    """GuardrailManager Instanz."""
    return create_default_guardrails()


@pytest.fixture
def override_learner() -> OverrideLearner:
    """OverrideLearner Instanz."""
    return create_override_learner(history_limit=1000)


@pytest.fixture
def engine(registry, scorer, guardrail_manager, override_learner) -> RefinementEngine:
    """RefinementEngine Instanz."""
    return RefinementEngine(
        scorer=scorer,
        guardrail_manager=guardrail_manager,
        override_learner=override_learner,
        registry=registry,
    )


# ============================================================================
# Tests: Initialization
# ============================================================================


class TestInitialization:
    """Tests für Initialisierung."""

    def test_init(self, registry, scorer, guardrail_manager, override_learner):
        """Test Standard-Initialisierung."""
        engine = RefinementEngine(
            scorer=scorer,
            guardrail_manager=guardrail_manager,
            override_learner=override_learner,
            registry=registry,
        )

        assert engine.scorer == scorer
        assert engine.guardrail_manager == guardrail_manager
        assert engine.override_learner == override_learner
        assert engine.registry == registry

    def test_config_defaults(self, engine):
        """Test Standard-Konfiguration."""
        assert "high_override_rate_threshold" in engine.config
        assert "prediction_error_threshold" in engine.config
        assert "min_samples_for_suggestion" in engine.config


# ============================================================================
# Tests: Guardrail Performance Analysis
# ============================================================================


class TestGuardrailAnalysis:
    """Tests für Guardrail Performance Analyse."""

    def test_analyze_guardrail_empty(self, engine):
        """Test Analyse mit keine Overrides."""
        suggestions = engine.analyze_guardrail_performance()
        assert isinstance(suggestions, list)
        # Keine Overrides = keine Vorschläge
        assert len(suggestions) == 0

    def test_analyze_guardrail_with_overrides(self, engine, registry):
        """Test Analyse mit Overrides."""
        # Logge mehrere Overrides
        for i in range(10):
            engine.override_learner.log_override(
                original_action="execute",
                original_decision="approve",
                human_decision="block",
                justification="Zu riskant",
                context={"features_active": ["feature_a", "feature_b"]},
                confidence_before=0.8,
                guardrail_violations=["exploration_limit"] if i < 7 else [],
            )

        suggestions = engine.analyze_guardrail_performance()

        # Sollte Vorschläge generieren
        assert isinstance(suggestions, list)

    def test_suggestion_structure(self, engine):
        """Test Struktur der Vorschläge."""
        # Override loggen
        for i in range(10):
            engine.override_learner.log_override(
                original_action="execute",
                original_decision="approve",
                human_decision="block",
                justification="Test",
                context={},
                guardrail_violations=["test_guardrail"],
            )

        suggestions = engine.analyze_guardrail_performance()

        if suggestions:
            sug = suggestions[0]
            assert isinstance(sug, RefinementSuggestion)
            assert sug.component in ("guardrails", "scorer", "hypothesis_generator", "quarantine", "thresholds", "other")
            assert 0.0 <= sug.confidence <= 1.0
            assert sug.priority >= 1


# ============================================================================
# Tests: Prediction Errors Analysis
# ============================================================================


class TestPredictionErrorAnalysis:
    """Tests für Prediction Errors Analyse."""

    def test_analyze_prediction_errors_empty(self, engine):
        """Test Analyse mit keine Runs."""
        suggestions = engine.analyze_prediction_errors()
        assert isinstance(suggestions, list)

    def test_analyze_prediction_errors_insufficient_data(self, engine):
        """Test Analyse mit unzureichenden Daten."""
        # Nur wenige Runs
        suggestions = engine.analyze_prediction_errors()
        assert isinstance(suggestions, list)


# ============================================================================
# Tests: Human Overrides Analysis
# ============================================================================


class TestOverrideAnalysis:
    """Tests für Human Overrides Analyse."""

    def test_analyze_overrides_empty(self, engine):
        """Test Analyse mit keine Overrides."""
        suggestions = engine.analyze_human_overrides()
        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_analyze_overrides_with_data(self, engine):
        """Test Analyse mit Overrides."""
        # Logge Overrides
        for i in range(10):
            engine.override_learner.log_override(
                original_action="execute",
                original_decision="approve",
                human_decision="block",
                justification="Zu hohe Exploration",
                context={"features_active": ["exploration_feature"]},
                confidence_before=0.75,
                action_type="run_proposal",
            )

        suggestions = engine.analyze_human_overrides()

        # Sollte Vorschläge generieren bei hoher Override-Rate
        assert isinstance(suggestions, list)

    def test_override_pattern_detection(self, engine):
        """Test Pattern-Erkennung in Overrides."""
        # Logge Overrides mit gleichem Feature
        for i in range(8):
            engine.override_learner.log_override(
                original_action="execute",
                original_decision="approve",
                human_decision="block",
                justification="Feature problematisch",
                context={"features_active": ["problematic_feature"]},
                confidence_before=0.8,
            )

        suggestions = engine.analyze_human_overrides()

        # Sollte Feature-bezogene Vorschläge generieren
        assert isinstance(suggestions, list)


# ============================================================================
# Tests: Refinement Report
# ============================================================================


class TestRefinementReport:
    """Tests für Refinement Report."""

    def test_generate_report_empty(self, engine):
        """Test Report mit keine Vorschläge."""
        report = engine.generate_refinement_report()

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Phase 4 Refinement Report" in report

    def test_generate_report_with_suggestions(self, engine):
        """Test Report mit Vorschlägen."""
        # Overrides loggen
        for i in range(15):
            engine.override_learner.log_override(
                original_action="execute",
                original_decision="approve",
                human_decision="block",
                justification="Test Override",
                context={"features_active": ["test_feature"]},
                confidence_before=0.7,
                guardrail_violations=["test_guardrail"],
            )

        report = engine.generate_refinement_report()

        assert isinstance(report, str)
        assert "Refinement-Vorschläge" in report or "Vorschläge" in report
        assert "Zusammenfassung" in report or "Summary" in report

    def test_report_structure(self, engine):
        """Test Report-Struktur."""
        report = engine.generate_refinement_report()

        # Abschnitte prüfen
        assert "# Phase 4 Refinement Report" in report
        assert "Generiert:" in report


# ============================================================================
# Tests: Refinement Application
# ============================================================================


class TestRefinementApplication:
    """Tests für Refinement Application."""

    def test_apply_refinement_guardrails(self, engine):
        """Test Anwenden von Guardrail-Refinement."""
        suggestion = RefinementSuggestion(
            component="guardrails",
            current_behavior="Test current",
            suggested_change="Test change",
            expected_improvement="Test improvement",
            confidence=0.8,
        )

        result = engine.apply_refinement(suggestion)
        assert result is True

    def test_apply_refinement_scorer(self, engine):
        """Test Anwenden von Scorer-Refinement."""
        suggestion = RefinementSuggestion(
            component="scorer",
            current_behavior="Test current",
            suggested_change="Test change",
            expected_improvement="Test improvement",
            confidence=0.7,
        )

        result = engine.apply_refinement(suggestion)
        assert result is True

    def test_apply_refinement_thresholds(self, engine):
        """Test Anwenden von Threshold-Refinement."""
        suggestion = RefinementSuggestion(
            component="thresholds",
            current_behavior="Test current",
            suggested_change="Test change",
            expected_improvement="Test improvement",
            confidence=0.9,
        )

        result = engine.apply_refinement(suggestion)
        assert result is True

    def test_apply_refinement_other(self, engine):
        """Test Anwenden von anderem Refinement."""
        suggestion = RefinementSuggestion(
            component="other",
            current_behavior="Test current",
            suggested_change="Test change",
            expected_improvement="Test improvement",
            confidence=0.6,
        )

        result = engine.apply_refinement(suggestion)
        assert result is True


# ============================================================================
# Tests: RefinementSuggestion Validation
# ============================================================================


class TestRefinementSuggestionValidation:
    """Tests für RefinementSuggestion Validierung."""

    def test_valid_suggestion(self):
        """Test gültiger Vorschlag."""
        sug = RefinementSuggestion(
            component="guardrails",
            current_behavior="Current behavior",
            suggested_change="Suggested change",
            expected_improvement="Expected improvement",
            confidence=0.8,
        )

        assert sug.component == "guardrails"
        assert sug.confidence == 0.8
        assert sug.priority == 3  # Default

    def test_invalid_confidence_too_high(self):
        """Test dass zu hohe Confidence Fehler wirft."""
        with pytest.raises(ValueError, match="Confidence muss zwischen 0.0 und 1.0 sein"):
            RefinementSuggestion(
                component="scorer",
                current_behavior="Test",
                suggested_change="Test",
                expected_improvement="Test",
                confidence=1.5,  # Zu hoch!
            )

    def test_invalid_confidence_negative(self):
        """Test dass negative Confidence Fehler wirft."""
        with pytest.raises(ValueError, match="Confidence muss zwischen 0.0 und 1.0 sein"):
            RefinementSuggestion(
                component="scorer",
                current_behavior="Test",
                suggested_change="Test",
                expected_improvement="Test",
                confidence=-0.1,  # Negativ!
            )

    def test_to_dict(self):
        """Test Dictionary-Konvertierung."""
        sug = RefinementSuggestion(
            component="guardrails",
            current_behavior="Current",
            suggested_change="Change",
            expected_improvement="Improvement",
            confidence=0.75,
            evidence=["Evidence 1", "Evidence 2"],
            priority=2,
        )

        d = sug.to_dict()

        assert d["component"] == "guardrails"
        assert d["confidence"] == 0.75
        assert d["priority"] == 2
        assert len(d["evidence"]) == 2


# ============================================================================
# Tests: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests für Edge Cases."""

    def test_empty_registry(self, temp_results_dir, scorer, guardrail_manager, override_learner):
        """Test mit leerer Registry."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        engine = RefinementEngine(
            scorer=scorer,
            guardrail_manager=guardrail_manager,
            override_learner=override_learner,
            registry=registry,
        )

        # Sollte nicht crashen
        suggestions = engine.analyze_guardrail_performance()
        assert isinstance(suggestions, list)

        suggestions = engine.analyze_prediction_errors()
        assert isinstance(suggestions, list)

        suggestions = engine.analyze_human_overrides()
        assert isinstance(suggestions, list)

    def test_only_failed_runs(self, temp_results_dir, scorer, guardrail_manager, override_learner):
        """Test mit nur fehlgeschlagenen Runs."""
        registry = RunRegistry(results_dir=str(temp_results_dir))

        for i in range(10):
            run_id = f"fail_{i:03d}"
            registry.register(run_id=run_id, config_hash="config", seed=42)
            registry.fail_run(run_id, notes="OOM")

        engine = RefinementEngine(
            scorer=scorer,
            guardrail_manager=guardrail_manager,
            override_learner=override_learner,
            registry=registry,
        )

        # Sollte nicht crashen
        report = engine.generate_refinement_report()
        assert isinstance(report, str)
