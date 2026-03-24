#!/usr/bin/env python3
"""
Tests für Guardrail System.

Phase 4C: Guardrail System & Integration
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Füge Parent-Directory zum Path hinzu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.guardrails import (
    AutonomyLevel,
    GuardrailType,
    Guardrail,
    AutonomyConfig,
    GuardrailManager,
    create_default_guardrails,
)


class TestGuardrail:
    """Tests für Guardrail-Klasse."""

    def test_guardrail_check_passes(self):
        """Guardrail-Check besteht wenn Wert unter Threshold."""
        guardrail = Guardrail(
            guardrail_type=GuardrailType.BUDGET,
            name="Test Budget",
            description="Test description",
            threshold=100.0,
            is_hard_limit=True,
            action_on_violation="block",
        )

        passed, message = guardrail.check(50.0)

        assert passed is True
        assert message is None

    def test_guardrail_check_violation(self):
        """Guardrail-Check fällt durch wenn Wert über Threshold."""
        guardrail = Guardrail(
            guardrail_type=GuardrailType.BUDGET,
            name="Test Budget",
            description="Test description",
            threshold=100.0,
            is_hard_limit=True,
            action_on_violation="block",
        )

        passed, message = guardrail.check(150.0)

        assert passed is False
        assert message is not None
        assert "Test Budget" in message
        assert "150.00" in message
        assert "100.00" in message

    def test_guardrail_check_at_threshold(self):
        """Guardrail-Check besteht genau am Threshold."""
        guardrail = Guardrail(
            guardrail_type=GuardrailType.BUDGET,
            name="Test Budget",
            description="Test description",
            threshold=100.0,
            is_hard_limit=True,
            action_on_violation="block",
        )

        passed, message = guardrail.check(100.0)

        assert passed is True
        assert message is None

    def test_guardrail_with_ratio_threshold(self):
        """Guardrail mit Ratio-Threshold (0-1)."""
        guardrail = Guardrail(
            guardrail_type=GuardrailType.EXPLORATION,
            name="Exploration Ratio",
            description="Max exploration ratio",
            threshold=0.5,
            is_hard_limit=False,
            action_on_violation="warn",
        )

        # Unter Threshold
        passed, _ = guardrail.check(0.3)
        assert passed is True

        # Über Threshold
        passed, message = guardrail.check(0.7)
        assert passed is False
        assert message is not None


class TestAutonomyConfig:
    """Tests für AutonomyConfig-Klasse."""

    def test_default_config(self):
        """Default Konfiguration erstellen."""
        config = AutonomyConfig(
            level=AutonomyLevel.SUPERVISED,
            guardrails=[],
            allowed_actions=["propose_runs", "execute_smoke"],
            requires_approval=["submit_bundle"],
        )

        assert config.level == AutonomyLevel.SUPERVISED
        assert len(config.guardrails) == 0
        assert "propose_runs" in config.allowed_actions
        assert "submit_bundle" in config.requires_approval

    def test_config_with_guardrails(self):
        """Konfiguration mit Guardrails."""
        guardrail = Guardrail(
            guardrail_type=GuardrailType.BUDGET,
            name="Budget",
            description="Budget limit",
            threshold=100.0,
            is_hard_limit=True,
            action_on_violation="block",
        )

        config = AutonomyConfig(
            level=AutonomyLevel.AUTONOMOUS,
            guardrails=[guardrail],
        )

        assert config.level == AutonomyLevel.AUTONOMOUS
        assert len(config.guardrails) == 1
        assert config.guardrails[0].name == "Budget"


class TestGuardrailManager:
    """Tests für GuardrailManager-Klasse."""

    def test_manager_initialization(self):
        """Manager Initialisierung."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        assert manager.config.level == AutonomyLevel.SUPERVISED
        assert len(manager.guardrails) == 5

    def test_check_action_allowed(self):
        """Erlaubte Aktion besteht Guardrail-Check."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        context = {
            "budget_used": 45.0,
            "confidence": 0.78,
            "exploration_ratio": 0.3,
            "safety_score": 0.9,
        }

        allowed, violations = manager.check_action("propose_runs", context)

        assert allowed is True
        assert len(violations) == 0

    def test_check_action_blocked_by_budget(self):
        """Aktion blockiert bei Budget-Überschreitung."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        context = {
            "budget_used": 120.0,  # Über Threshold von 100
            "confidence": 0.78,
            "exploration_ratio": 0.3,
            "safety_score": 0.9,
        }

        allowed, violations = manager.check_action("propose_runs", context)

        assert allowed is False
        assert len(violations) > 0
        assert "Budget" in violations[0]

    def test_check_action_blocked_by_confidence(self):
        """Aktion blockiert bei zu niedriger Confidence."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        context = {
            "budget_used": 45.0,
            "confidence": 0.4,  # Unter Threshold von 0.6
            "exploration_ratio": 0.3,
            "safety_score": 0.9,
        }

        allowed, violations = manager.check_action("propose_runs", context)

        assert allowed is False
        assert len(violations) > 0
        assert "Confidence" in violations[0] or "confidence" in violations[0].lower()

    def test_check_action_not_in_allowed_list(self):
        """Aktion blockiert wenn nicht in allowed_actions."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        allowed, violations = manager.check_action("unknown_action", {})

        assert allowed is False
        assert len(violations) > 0
        assert "nicht in der Liste" in violations[0]

    def test_get_required_approvals(self):
        """Benötigte Approvals abrufen."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        approvals = manager.get_required_approvals("submit_bundle")

        assert "technical_lead" in approvals
        assert "project_owner" in approvals

    def test_get_required_approvals_empty(self):
        """Keine Approvals für normale Aktionen."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        approvals = manager.get_required_approvals("propose_runs")

        assert approvals == []

    def test_create_approval_request(self):
        """Approval-Anfrage erstellen."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        context = {
            "confidence": 0.72,
            "budget_used": 0.45,
        }

        request = manager.create_approval_request("submit_bundle", context)

        assert "request_id" in request
        assert request["action"] == "submit_bundle"
        assert "requires_approval_from" in request
        assert len(request["requires_approval_from"]) > 0
        assert "reason" in request

    def test_create_approval_request_risk_level_high(self):
        """High Risk-Level für kritische Aktionen."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        request = manager.create_approval_request("submit_bundle", {})

        assert request["risk_level"] == "high"

    def test_create_approval_request_risk_level_low(self):
        """Low Risk-Level bei hoher Confidence."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        context = {"confidence": 0.9}
        request = manager.create_approval_request("propose_runs", context)

        assert request["risk_level"] == "low"

    def test_add_guardrail(self):
        """Neue Guardrail hinzufügen."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        initial_count = len(manager.guardrails)

        new_guardrail = Guardrail(
            guardrail_type=GuardrailType.BUDGET,
            name="Custom Budget",
            description="Custom budget limit",
            threshold=200.0,
            is_hard_limit=True,
            action_on_violation="block",
        )

        manager.add_guardrail(new_guardrail)

        assert len(manager.guardrails) == initial_count + 1

    def test_remove_guardrail(self):
        """Guardrail entfernen."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        initial_count = len(manager.guardrails)

        removed = manager.remove_guardrail("Weekly GPU Budget")

        assert removed is True
        assert len(manager.guardrails) == initial_count - 1

    def test_remove_guardrail_not_found(self):
        """Nicht-existente Guardrail entfernen."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        removed = manager.remove_guardrail("NonExistentGuardrail")

        assert removed is False

    def test_get_guardrail_status(self):
        """Guardrail-Status abrufen."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)

        status = manager.get_guardrail_status()

        assert "autonomy_level" in status
        assert status["autonomy_level"] == "supervised"
        assert "total_guardrails" in status
        assert status["total_guardrails"] == 5
        assert "guardrails_by_type" in status

    def test_soft_limit_violation_warns(self):
        """Soft-Limit Verletzung warnt nur."""
        config = AutonomyConfig(
            level=AutonomyLevel.SUPERVISED,
            guardrails=[
                Guardrail(
                    guardrail_type=GuardrailType.EXPLORATION,
                    name="Exploration",
                    description="Max exploration",
                    threshold=0.5,
                    is_hard_limit=False,
                    action_on_violation="warn",
                )
            ],
            allowed_actions=["propose_runs"],
        )
        manager = GuardrailManager(config)

        context = {"exploration_ratio": 0.7}
        allowed, violations = manager.check_action("propose_runs", context)

        # Soft-Limit blockiert nicht automatisch
        assert allowed is True
        assert len(violations) > 0


class TestCreateDefaultGuardrails:
    """Tests für create_default_guardrails Funktion."""

    def test_default_guardrails_count(self):
        """Default Guardrails erstellen."""
        config = create_default_guardrails()

        assert len(config.guardrails) == 5

    def test_default_autonomy_level(self):
        """Default Autonomie-Level."""
        config = create_default_guardrails()

        assert config.level == AutonomyLevel.SUPERVISED

    def test_default_has_budget_guardrail(self):
        """Default Config hat Budget Guardrail."""
        config = create_default_guardrails()

        budget_guardrails = [
            g for g in config.guardrails
            if g.guardrail_type == GuardrailType.BUDGET
        ]

        assert len(budget_guardrails) == 1
        assert budget_guardrails[0].threshold == 100.0

    def test_default_has_submission_guardrail(self):
        """Default Config hat Submission Guardrail."""
        config = create_default_guardrails()

        submission_guardrails = [
            g for g in config.guardrails
            if g.guardrail_type == GuardrailType.SUBMISSION
        ]

        assert len(submission_guardrails) == 1

    def test_default_allowed_actions(self):
        """Default erlaubte Aktionen."""
        config = create_default_guardrails()

        assert "propose_runs" in config.allowed_actions
        assert "execute_smoke" in config.allowed_actions
        assert "promote_candidate" in config.allowed_actions

    def test_default_requires_approval(self):
        """Default Approval-pflichtige Aktionen."""
        config = create_default_guardrails()

        assert "submit_bundle" in config.requires_approval
        assert "execute_expensive_run" in config.requires_approval
