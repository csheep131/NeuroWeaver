#!/usr/bin/env python3
"""
Tests für Override Learner.

Phase 4C: Guardrail System & Integration
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Füge Parent-Directory zum Path hinzu für direkte Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.override_learner import (
    OverrideEvent,
    OverrideLearner,
    create_override_learner,
)


class TestOverrideEvent:
    """Tests für OverrideEvent-Klasse."""

    def test_override_creation(self):
        """OverrideEvent erstellen."""
        event = OverrideEvent(
            override_id="override-123",
            original_action="execute_run",
            original_decision="execute",
            human_decision="block",
            justification="Safety concern",
            timestamp=datetime.utcnow(),
            context={"confidence": 0.75},
        )

        assert event.override_id == "override-123"
        assert event.original_action == "execute_run"
        assert event.original_decision == "execute"
        assert event.human_decision == "block"
        assert event.justification == "Safety concern"

    def test_override_to_dict(self):
        """Override zu Dictionary konvertieren."""
        event = OverrideEvent(
            override_id="override-456",
            original_action="promote_run",
            original_decision="approve",
            human_decision="reject",
            justification="Not ready",
            timestamp=datetime.utcnow(),
            context={"run_id": "run001"},
            action_type="promote_candidate",
            confidence_before=0.65,
        )

        result_dict = event.to_dict()

        assert result_dict["override_id"] == "override-456"
        assert result_dict["action_type"] == "promote_candidate"
        assert result_dict["confidence_before"] == 0.65
        assert "context" in result_dict


class TestOverrideLearner:
    """Tests für OverrideLearner-Klasse."""

    @pytest.fixture
    def learner(self):
        """Learner Fixture."""
        return create_override_learner()

    def test_learner_initialization(self, learner):
        """Learner Initialisierung."""
        assert len(learner.overrides) == 0

    def test_log_override(self, learner):
        """Override dokumentieren."""
        event = learner.log_override(
            original_action="execute_run",
            original_decision="execute",
            human_decision="block",
            context={"confidence": 0.75},
            justification="Test justification",
        )

        assert event.override_id is not None
        assert event.original_action == "execute_run"
        assert len(learner.overrides) == 1

    def test_log_override_with_action_type(self, learner):
        """Override mit Action-Type dokumentieren."""
        event = learner.log_override(
            original_action="execute_run",
            original_decision="execute",
            human_decision="block",
            context={},
            action_type="execute_smoke",
            confidence_before=0.75,
        )

        assert event.action_type == "execute_smoke"
        assert event.confidence_before == 0.75

    def test_log_override_with_guardrail_violations(self, learner):
        """Override mit Guardrail-Verletzungen."""
        event = learner.log_override(
            original_action="execute_run",
            original_decision="execute",
            human_decision="block",
            context={},
            guardrail_violations=["Budget exceeded", "Low confidence"],
        )

        assert len(event.guardrail_violations) == 2

    def test_analyze_override_patterns_empty(self, learner):
        """Muster-Analyse wenn keine Overrides."""
        patterns = learner.analyze_override_patterns()

        assert patterns["total_overrides"] == 0
        assert patterns["most_overridden_actions"] == []
        assert patterns["common_justifications"] == []

    def test_analyze_override_patterns_with_data(self, learner):
        """Muster-Analyse mit Daten."""
        for i in range(5):
            learner.log_override(
                original_action="execute_run",
                original_decision="execute",
                human_decision="block",
                context={},
                action_type="execute_smoke",
            )

        patterns = learner.analyze_override_patterns()

        assert patterns["total_overrides"] == 5
        assert len(patterns["most_overridden_actions"]) > 0
        assert patterns["most_overridden_actions"][0]["action"] == "execute_run"

    def test_analyze_override_patterns_justifications(self, learner):
        """Muster-Analyse für Begründungen."""
        for i in range(3):
            learner.log_override(
                original_action="action1",
                original_decision="execute",
                human_decision="block",
                context={},
                justification="Safety concern",
            )
            learner.log_override(
                original_action="action2",
                original_decision="execute",
                human_decision="block",
                context={},
                justification="Budget issue",
            )

        patterns = learner.analyze_override_patterns()

        assert len(patterns["common_justifications"]) > 0

    def test_suggest_threshold_adjustments_empty(self, learner):
        """Threshold-Vorschläge wenn keine Overrides."""
        suggestions = learner.suggest_threshold_adjustments()

        assert len(suggestions) == 0

    def test_suggest_threshold_adjustments_strict(self, learner):
        """Vorschlag für zu strenge Thresholds."""
        # Simuliere viele Blocks bei einer Action
        for i in range(10):
            learner.log_override(
                original_action="exploration_run",
                original_decision="execute",
                human_decision="block",
                context={"total_actions_of_type": 12},
                action_type="exploration_run",
                confidence_before=0.8,
            )

        suggestions = learner.suggest_threshold_adjustments()

        assert len(suggestions) > 0
        assert any(s["type"] == "threshold_too_strict" for s in suggestions)

    def test_suggest_threshold_adjustments_loose(self, learner):
        """Vorschlag für zu lockere Thresholds."""
        # Simuliere viele Executes bei einer Action
        for i in range(10):
            learner.log_override(
                original_action="risky_run",
                original_decision="block",
                human_decision="execute",
                context={"total_actions_of_type": 12},
                action_type="risky_run",
            )

        suggestions = learner.suggest_threshold_adjustments()

        assert len(suggestions) > 0
        assert any(s["type"] == "threshold_too_loose" for s in suggestions)

    def test_suggest_threshold_adjustments_confidence(self, learner):
        """Vorschlag für Confidence-Threshold."""
        # Simuliere Blocks bei niedriger Confidence
        for i in range(6):
            learner.log_override(
                original_action="low_conf_run",
                original_decision="execute",
                human_decision="block",
                context={},
                confidence_before=0.4,
            )

        suggestions = learner.suggest_threshold_adjustments()

        assert len(suggestions) > 0
        assert any(s["type"] == "confidence_threshold_too_low" for s in suggestions)

    def test_calibrate_confidence(self, learner):
        """Confidence-Kalibrierung."""
        result = learner.calibrate_confidence(
            predicted_confidence=0.9,
            actual_success_rate=0.6,
        )

        assert "calibration_factor" in result
        assert "recommendation" in result
        assert result["calibration_factor"] < 1.0  # Confidence war zu hoch

    def test_calibrate_confidence_too_low(self, learner):
        """Kalibrierung wenn Confidence zu niedrig."""
        result = learner.calibrate_confidence(
            predicted_confidence=0.5,
            actual_success_rate=0.8,
        )

        assert result["calibration_factor"] > 1.0
        assert "erhöhen" in result["recommendation"]

    def test_calibrate_confidence_well_calibrated(self, learner):
        """Kalibrierung wenn gut kalibriert."""
        result = learner.calibrate_confidence(
            predicted_confidence=0.8,
            actual_success_rate=0.78,
        )

        assert 0.9 <= result["calibration_factor"] <= 1.1
        assert "gut kalibriert" in result["recommendation"]

    def test_get_calibration_statistics_empty(self, learner):
        """Kalibrierungs-Statistiken wenn keine Daten."""
        stats = learner.get_calibration_statistics()

        assert stats["data_points"] == 0
        assert stats["calibration_quality"] == "unknown"

    def test_get_calibration_statistics_with_data(self, learner):
        """Kalibrierungs-Statistiken mit Daten."""
        learner.calibrate_confidence(0.8, 0.75)
        learner.calibrate_confidence(0.7, 0.65)
        learner.calibrate_confidence(0.9, 0.85)

        stats = learner.get_calibration_statistics()

        assert stats["data_points"] == 3
        assert "avg_calibration_factor" in stats
        assert "calibration_quality" in stats

    def test_get_override_statistics_empty(self, learner):
        """Override-Statistiken wenn keine Overrides."""
        stats = learner.get_override_statistics(hours=24)

        assert stats["total_overrides"] == 0

    def test_get_override_statistics_with_data(self, learner):
        """Override-Statistiken mit Daten."""
        learner.log_override(
            original_action="action1",
            original_decision="execute",
            human_decision="block",
            context={},
            action_type="type_a",
            confidence_before=0.7,
        )
        learner.log_override(
            original_action="action2",
            original_decision="execute",
            human_decision="block",
            context={},
            action_type="type_a",
            confidence_before=0.6,
        )
        learner.log_override(
            original_action="action3",
            original_decision="block",
            human_decision="execute",
            context={},
            action_type="type_b",
        )

        stats = learner.get_override_statistics(hours=24)

        assert stats["total_overrides"] == 3
        assert "by_decision" in stats
        assert stats["by_decision"]["block"] == 2
        assert stats["by_decision"]["execute"] == 1

    def test_get_overrides_by_action_type(self, learner):
        """Overrides nach Action-Type filtern."""
        learner.log_override(
            original_action="action1",
            original_decision="execute",
            human_decision="block",
            context={},
            action_type="type_a",
        )
        learner.log_override(
            original_action="action2",
            original_decision="execute",
            human_decision="block",
            context={},
            action_type="type_b",
        )
        learner.log_override(
            original_action="action3",
            original_decision="execute",
            human_decision="block",
            context={},
            action_type="type_a",
        )

        type_a_overrides = learner.get_overrides_by_action_type("type_a")

        assert len(type_a_overrides) == 2

    def test_clear_history(self, learner):
        """History löschen."""
        for i in range(5):
            learner.log_override(
                original_action=f"action{i}",
                original_decision="execute",
                human_decision="block",
                context={},
            )

        learner.clear_history()

        assert len(learner.overrides) == 0

    def test_history_limit_enforced(self):
        """History-Limit wird durchgesetzt."""
        learner = create_override_learner(history_limit=10)

        for i in range(20):
            learner.log_override(
                original_action=f"action{i}",
                original_decision="execute",
                human_decision="block",
                context={},
            )

        assert len(learner.overrides) <= 10


class TestCreateOverrideLearner:
    """Tests für create_override_learner Funktion."""

    def test_create_learner(self):
        """Learner erstellen."""
        learner = create_override_learner()

        assert isinstance(learner, OverrideLearner)
        assert len(learner.overrides) == 0

    def test_create_learner_with_custom_limit(self):
        """Learner mit benutzerdefiniertem Limit."""
        learner = create_override_learner(history_limit=50)

        assert learner._history_limit == 50
