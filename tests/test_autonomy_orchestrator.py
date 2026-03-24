#!/usr/bin/env python3
"""
Tests für Autonomy Orchestrator.

Phase 4C: Guardrail System & Integration
"""

from __future__ import annotations

import pytest
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Füge Parent-Directory zum Path hinzu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports nach dem Hinzufügen zum Path
from orchestrator.guardrails import (
    AutonomyLevel,
    create_default_guardrails,
    GuardrailManager,
)
from orchestrator.autonomy_orchestrator import (
    AutonomousAction,
    AutonomyOrchestrator,
    create_autonomy_orchestrator,
)


class TestAutonomousAction:
    """Tests für AutonomousAction-Klasse."""

    def test_action_creation(self):
        """AutonomousAction erstellen."""
        action = AutonomousAction(
            action_id="test-123",
            action_type="propose_run",
            timestamp=datetime.utcnow(),
            status="pending",
            confidence=0.75,
        )

        assert action.action_id == "test-123"
        assert action.action_type == "propose_run"
        assert action.status == "pending"
        assert action.confidence == 0.75
        assert action.result is None

    def test_action_to_dict(self):
        """Action zu Dictionary konvertieren."""
        action = AutonomousAction(
            action_id="test-456",
            action_type="execute_smoke",
            timestamp=datetime.utcnow(),
            status="completed",
            confidence=0.85,
            guardrail_checks={"initial_check": True},
            result={"success": True},
        )

        result_dict = action.to_dict()

        assert result_dict["action_id"] == "test-456"
        assert result_dict["action_type"] == "execute_smoke"
        assert result_dict["status"] == "completed"
        assert "guardrail_checks" in result_dict
        assert "result" in result_dict


class TestAutonomyOrchestrator:
    """Tests für AutonomyOrchestrator-Klasse."""

    @pytest.fixture
    def orchestrator(self):
        """Orchestrator Fixture."""
        return create_autonomy_orchestrator()

    def test_orchestrator_initialization(self):
        """Orchestrator Initialisierung."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)
        orchestrator = AutonomyOrchestrator(manager)

        assert orchestrator.guardrail_manager is manager

    @pytest.mark.asyncio
    async def test_propose_action_allowed(self, orchestrator):
        """Erlaubte Aktion vorschlagen."""
        context = {
            "confidence": 0.75,
            "budget_used": 45.0,
            "exploration_ratio": 0.3,
        }

        action = await orchestrator.propose_action("propose_runs", context)

        assert action.action_type == "propose_runs"
        assert action.status in ("pending", "awaiting_approval")
        assert action.confidence == 0.75

    @pytest.mark.asyncio
    async def test_propose_action_blocked_by_guardrail(self, orchestrator):
        """Aktion durch Guardrail blockiert."""
        context = {
            "confidence": 0.4,  # Unter Threshold von 0.6
        }

        action = await orchestrator.propose_action("propose_runs", context)

        assert action.status == "blocked"
        assert len(action.violation_messages) > 0

    @pytest.mark.asyncio
    async def test_propose_action_requires_approval(self, orchestrator):
        """Aktion benötigt Approval."""
        context = {
            "confidence": 0.75,
        }

        action = await orchestrator.propose_action("submit_bundle", context)

        assert action.status == "awaiting_approval"
        assert action.approval_status is not None
        assert "request_id" in action.approval_status

    @pytest.mark.asyncio
    async def test_execute_action(self, orchestrator):
        """Aktion ausführen."""
        context = {"confidence": 0.75}

        # Zuerst vorschlagen
        action = await orchestrator.propose_action("propose_runs", context)

        # Dann ausführen
        result_action = await orchestrator.execute_action(action)

        assert result_action.action_id == action.action_id
        assert result_action.status in ("completed", "blocked", "pending")

    @pytest.mark.asyncio
    async def test_execute_blocked_action(self, orchestrator):
        """Blockierte Aktion kann nicht ausgeführt werden."""
        context = {"confidence": 0.4}

        action = await orchestrator.propose_action("propose_runs", context)
        assert action.status == "blocked"

        result_action = await orchestrator.execute_action(action)

        assert result_action.status == "blocked"

    def test_approve_action(self, orchestrator):
        """Aktion genehmigen."""
        context = {"confidence": 0.75}

        # Async Teil
        loop = asyncio.get_event_loop()
        action = loop.run_until_complete(
            orchestrator.propose_action("submit_bundle", context)
        )

        # Sync Approval
        success = orchestrator.approve_action(
            action.action_id,
            approved_by="test_user",
            justification="Test justification",
        )

        assert success is True

        # Status prüfen
        updated_action = orchestrator.get_action(action.action_id)
        assert updated_action is not None
        assert updated_action.approval_status["status"] == "approved"

    def test_reject_action(self, orchestrator):
        """Aktion ablehnen."""
        context = {"confidence": 0.75}

        loop = asyncio.get_event_loop()
        action = loop.run_until_complete(
            orchestrator.propose_action("submit_bundle", context)
        )

        success = orchestrator.reject_action(
            action.action_id,
            rejected_by="test_user",
            reason="Test reason",
        )

        assert success is True

        updated_action = orchestrator.get_action(action.action_id)
        assert updated_action is not None
        assert updated_action.approval_status["status"] == "rejected"
        assert updated_action.status == "blocked"

    def test_approve_nonexistent_action(self, orchestrator):
        """Nicht-existente Aktion genehmigen."""
        success = orchestrator.approve_action("nonexistent-id", "user")

        assert success is False

    def test_get_action(self, orchestrator):
        """Einzelne Aktion abrufen."""
        context = {"confidence": 0.75}

        loop = asyncio.get_event_loop()
        action = loop.run_until_complete(
            orchestrator.propose_action("propose_runs", context)
        )

        retrieved = orchestrator.get_action(action.action_id)

        assert retrieved is not None
        assert retrieved.action_id == action.action_id

    def test_get_action_not_found(self, orchestrator):
        """Nicht-existente Aktion abrufen."""
        retrieved = orchestrator.get_action("nonexistent-id")

        assert retrieved is None

    def test_get_action_history(self, orchestrator):
        """Action-Historie abrufen."""
        context = {"confidence": 0.75}

        loop = asyncio.get_event_loop()

        # Mehrere Aktionen erstellen
        for i in range(5):
            loop.run_until_complete(
                orchestrator.propose_action("propose_runs", context)
            )

        history = orchestrator.get_action_history(limit=10)

        assert len(history) == 5

    def test_get_action_history_limited(self, orchestrator):
        """Begrenzte Action-Historie."""
        context = {"confidence": 0.75}

        loop = asyncio.get_event_loop()

        for i in range(10):
            loop.run_until_complete(
                orchestrator.propose_action("propose_runs", context)
            )

        history = orchestrator.get_action_history(limit=3)

        assert len(history) == 3

    def test_get_statistics_empty(self, orchestrator):
        """Statistiken wenn keine Aktionen."""
        stats = orchestrator.get_statistics()

        assert stats["total_actions"] == 0
        assert stats["auto_executed"] == 0
        assert stats["human_approved"] == 0
        assert stats["blocked_by_guardrails"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_statistics_with_actions(self, orchestrator):
        """Statistiken mit Aktionen."""
        context = {"confidence": 0.75}

        loop = asyncio.get_event_loop()

        # Aktionen erstellen
        for i in range(10):
            loop.run_until_complete(
                orchestrator.propose_action("propose_runs", context)
            )

        stats = orchestrator.get_statistics()

        assert stats["total_actions"] == 10
        assert "success_rate" in stats

    def test_get_pending_actions(self, orchestrator):
        """Ausstehende Aktionen abrufen."""
        context = {"confidence": 0.75}

        loop = asyncio.get_event_loop()

        action1 = loop.run_until_complete(
            orchestrator.propose_action("propose_runs", context)
        )
        action2 = loop.run_until_complete(
            orchestrator.propose_action("execute_smoke", context)
        )

        pending = orchestrator.get_pending_actions()

        assert len(pending) >= 2

    def test_get_actions_by_status(self, orchestrator):
        """Aktionen nach Status filtern."""
        context_blocked = {"confidence": 0.4}
        context_allowed = {"confidence": 0.75}

        loop = asyncio.get_event_loop()

        loop.run_until_complete(
            orchestrator.propose_action("propose_runs", context_blocked)
        )
        loop.run_until_complete(
            orchestrator.propose_action("execute_smoke", context_allowed)
        )

        blocked = orchestrator.get_actions_by_status("blocked")
        assert len(blocked) >= 1

    def test_clear_history(self, orchestrator):
        """History löschen."""
        context = {"confidence": 0.75}

        loop = asyncio.get_event_loop()

        for i in range(5):
            loop.run_until_complete(
                orchestrator.propose_action("propose_runs", context)
            )

        orchestrator.clear_history()

        stats = orchestrator.get_statistics()
        assert stats["total_actions"] == 0

    def test_history_limit_enforced(self):
        """History-Limit wird durchgesetzt."""
        config = create_default_guardrails()
        manager = GuardrailManager(config)
        orchestrator = AutonomyOrchestrator(manager, action_history_limit=10)

        context = {"confidence": 0.75}

        loop = asyncio.get_event_loop()

        # Mehr Aktionen als Limit
        for i in range(20):
            loop.run_until_complete(
                orchestrator.propose_action("propose_runs", context)
            )

        stats = orchestrator.get_statistics()
        assert stats["total_actions"] <= 15  # Etwas Spielraum für interne Aktionen


class TestCreateAutonomyOrchestrator:
    """Tests für create_autonomy_orchestrator Funktion."""

    def test_create_with_default_level(self):
        """Mit default Autonomie-Level erstellen."""
        orchestrator = create_autonomy_orchestrator()

        assert orchestrator is not None
        assert orchestrator.guardrail_manager.config.level == AutonomyLevel.SUPERVISED

    def test_create_with_custom_level(self):
        """Mit benutzerdefiniertem Autonomie-Level erstellen."""
        orchestrator = create_autonomy_orchestrator(
            autonomy_level=AutonomyLevel.AUTONOMOUS
        )

        assert orchestrator.guardrail_manager.config.level == AutonomyLevel.AUTONOMOUS

    def test_create_returns_orchestrator(self):
        """Erstellt AutonomyOrchestrator Instanz."""
        orchestrator = create_autonomy_orchestrator()

        assert isinstance(orchestrator, AutonomyOrchestrator)
