#!/usr/bin/env python3
"""
Tests für Approval Interface.

Phase 4C: Guardrail System & Integration
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Füge Parent-Directory zum Path hinzu
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.guardrails import create_default_guardrails, GuardrailManager
from orchestrator.autonomy_orchestrator import (
    AutonomyOrchestrator,
    AutonomousAction,
    create_autonomy_orchestrator,
)
from orchestrator.approval_interface import (
    ApprovalRequest,
    ApprovalInterface,
    create_approval_interface,
)


class TestApprovalRequest:
    """Tests für ApprovalRequest-Klasse."""

    def test_request_creation(self):
        """ApprovalRequest erstellen."""
        now = datetime.utcnow()
        request = ApprovalRequest(
            request_id="req-123",
            action_type="submit_bundle",
            description="Test submission",
            risk_level="high",
            confidence=0.75,
            guardrail_status={"initial_check": True},
            created_at=now,
            expires_at=now + timedelta(hours=24),
            status="pending",
        )

        assert request.request_id == "req-123"
        assert request.action_type == "submit_bundle"
        assert request.risk_level == "high"
        assert request.status == "pending"

    def test_request_to_dict(self):
        """Request zu Dictionary konvertieren."""
        now = datetime.utcnow()
        request = ApprovalRequest(
            request_id="req-456",
            action_type="execute_smoke",
            description="Test",
            risk_level="low",
            confidence=0.9,
            guardrail_status={},
            created_at=now,
            expires_at=now + timedelta(hours=24),
            status="pending",
        )

        result_dict = request.to_dict()

        assert result_dict["request_id"] == "req-456"
        assert result_dict["action_type"] == "execute_smoke"
        assert "created_at" in result_dict
        assert "expires_at" in result_dict

    def test_request_is_expired(self):
        """Prüfen ob Request abgelaufen."""
        past = datetime.utcnow() - timedelta(hours=2)
        request = ApprovalRequest(
            request_id="req-expired",
            action_type="test",
            description="Test",
            risk_level="low",
            confidence=0.9,
            guardrail_status={},
            created_at=past,
            expires_at=past + timedelta(hours=1),  # Already expired
            status="pending",
        )

        assert request.is_expired() is True

    def test_request_not_expired(self):
        """Request nicht abgelaufen."""
        now = datetime.utcnow()
        request = ApprovalRequest(
            request_id="req-active",
            action_type="test",
            description="Test",
            risk_level="low",
            confidence=0.9,
            guardrail_status={},
            created_at=now,
            expires_at=now + timedelta(hours=24),
            status="pending",
        )

        assert request.is_expired() is False

    def test_request_is_pending(self):
        """Prüfen ob Request pending."""
        now = datetime.utcnow()
        request = ApprovalRequest(
            request_id="req-pending",
            action_type="test",
            description="Test",
            risk_level="low",
            confidence=0.9,
            guardrail_status={},
            created_at=now,
            expires_at=now + timedelta(hours=24),
            status="pending",
        )

        assert request.is_pending() is True

    def test_request_not_pending_approved(self):
        """Approved Request ist nicht pending."""
        now = datetime.utcnow()
        request = ApprovalRequest(
            request_id="req-approved",
            action_type="test",
            description="Test",
            risk_level="low",
            confidence=0.9,
            guardrail_status={},
            created_at=now,
            expires_at=now + timedelta(hours=24),
            status="approved",
        )

        assert request.is_pending() is False


class TestApprovalInterface:
    """Tests für ApprovalInterface-Klasse."""

    @pytest.fixture
    def interface(self):
        """Interface Fixture."""
        orchestrator = create_autonomy_orchestrator()
        return create_approval_interface(orchestrator)

    def test_interface_initialization(self, interface):
        """Interface Initialisierung."""
        assert interface.orchestrator is not None

    def test_create_approval_request(self, interface):
        """Approval-Anfrage erstellen."""
        action = AutonomousAction(
            action_id="action-123",
            action_type="submit_bundle",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.75,
        )

        request = interface.create_approval_request(action)

        assert request.request_id is not None
        assert request.action_type == "submit_bundle"
        assert request.status == "pending"
        assert request.action_id == "action-123"

    def test_create_approval_request_with_description(self, interface):
        """Approval-Anfrage mit Beschreibung."""
        action = AutonomousAction(
            action_id="action-456",
            action_type="execute_smoke",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.85,
        )

        request = interface.create_approval_request(
            action, description="Custom description"
        )

        assert request.description == "Custom description"

    def test_get_pending_approvals(self, interface):
        """Ausstehende Approvals abrufen."""
        action = AutonomousAction(
            action_id="action-pending",
            action_type="submit_bundle",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.75,
        )

        interface.create_approval_request(action)

        pending = interface.get_pending_approvals()

        assert len(pending) >= 1
        assert pending[0].action_id == "action-pending"

    def test_get_request(self, interface):
        """Einzelne Anfrage abrufen."""
        action = AutonomousAction(
            action_id="action-get",
            action_type="test",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.75,
        )

        created_request = interface.create_approval_request(action)

        retrieved = interface.get_request(created_request.request_id)

        assert retrieved is not None
        assert retrieved.request_id == created_request.request_id

    def test_get_request_not_found(self, interface):
        """Nicht-existente Anfrage abrufen."""
        retrieved = interface.get_request("nonexistent-id")

        assert retrieved is None

    def test_approve_request(self, interface):
        """Anfrage genehmigen."""
        action = AutonomousAction(
            action_id="action-approve",
            action_type="submit_bundle",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.75,
        )

        request = interface.create_approval_request(action)

        success = interface.approve(
            request.request_id, justification="Test justification"
        )

        assert success is True

        updated = interface.get_request(request.request_id)
        assert updated is not None
        assert updated.status == "approved"
        assert updated.justification == "Test justification"

    def test_approve_nonexistent_request(self, interface):
        """Nicht-existente Anfrage genehmigen."""
        success = interface.approve("nonexistent-id", "Justification")

        assert success is False

    def test_approve_already_approved_request(self, interface):
        """Bereits genehmigte Anfrage erneut genehmigen."""
        action = AutonomousAction(
            action_id="action-double",
            action_type="test",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.75,
        )

        request = interface.create_approval_request(action)
        interface.approve(request.request_id)

        success = interface.approve(request.request_id, "Second approval")

        assert success is False

    def test_reject_request(self, interface):
        """Anfrage ablehnen."""
        action = AutonomousAction(
            action_id="action-reject",
            action_type="submit_bundle",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.75,
        )

        request = interface.create_approval_request(action)

        success = interface.reject(request.request_id, reason="Test reason")

        assert success is True

        updated = interface.get_request(request.request_id)
        assert updated is not None
        assert updated.status == "rejected"
        assert updated.rejection_reason == "Test reason"

    def test_reject_nonexistent_request(self, interface):
        """Nicht-existente Anfrage ablehnen."""
        success = interface.reject("nonexistent-id", "Reason")

        assert success is False

    def test_bulk_approve(self, interface):
        """Bulk-Approval."""
        request_ids = []

        for i in range(5):
            action = AutonomousAction(
                action_id=f"action-bulk-{i}",
                action_type="propose_runs",
                timestamp=datetime.utcnow(),
                status="awaiting_approval",
                confidence=0.85,  # High confidence = low risk
            )
            request = interface.create_approval_request(action)
            request_ids.append(request.request_id)

        results = interface.bulk_approve(request_ids, risk_threshold="low")

        assert len(results) == 5
        assert all(success for success in results.values())

    def test_bulk_approve_mixed_risk(self, interface):
        """Bulk-Approval mit gemischtem Risk-Level."""
        request_ids = []

        # Low risk request
        action_low = AutonomousAction(
            action_id="action-low",
            action_type="propose_runs",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.9,
        )
        request_low = interface.create_approval_request(action_low)
        request_ids.append(request_low.request_id)

        # High risk request
        action_high = AutonomousAction(
            action_id="action-high",
            action_type="submit_bundle",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.5,
        )
        request_high = interface.create_approval_request(action_high)
        request_ids.append(request_high.request_id)

        results = interface.bulk_approve(request_ids, risk_threshold="low")

        # Low risk should pass, high risk should fail
        assert results.get(request_low.request_id) is True
        assert results.get(request_high.request_id) is False

    def test_get_approval_statistics(self, interface):
        """Approval-Statistiken."""
        stats = interface.get_approval_statistics()

        assert "pending" in stats
        assert "approved_today" in stats
        assert "rejected_today" in stats
        assert "avg_approval_time" in stats
        assert "total_requests" in stats

    def test_get_requests_by_risk_level(self, interface):
        """Anfragen nach Risk-Level filtern."""
        # Create low risk request
        action_low = AutonomousAction(
            action_id="action-low-risk",
            action_type="propose_runs",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.9,
        )
        request_low = interface.create_approval_request(action_low)

        # Create high risk request
        action_high = AutonomousAction(
            action_id="action-high-risk",
            action_type="submit_bundle",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.4,
        )
        request_high = interface.create_approval_request(action_high)

        low_risk_requests = interface.get_requests_by_risk_level("low")
        high_risk_requests = interface.get_requests_by_risk_level("high")

        assert len(low_risk_requests) >= 1
        assert len(high_risk_requests) >= 1

    def test_get_expired_requests(self, interface):
        """Abgelaufene Anfragen."""
        # Create expired request
        past = datetime.utcnow() - timedelta(hours=25)
        request = ApprovalRequest(
            request_id="req-expired-test",
            action_type="test",
            description="Test",
            risk_level="low",
            confidence=0.9,
            guardrail_status={},
            created_at=past,
            expires_at=past + timedelta(hours=1),
            status="pending",
        )
        interface._requests[request.request_id] = request

        expired = interface.get_expired_requests()

        assert any(r.request_id == "req-expired-test" for r in expired)

    def test_cleanup_expired(self, interface):
        """Abgelaufene Anfragen bereinigen."""
        # Create expired request
        past = datetime.utcnow() - timedelta(hours=25)
        request = ApprovalRequest(
            request_id="req-cleanup",
            action_type="test",
            description="Test",
            risk_level="low",
            confidence=0.9,
            guardrail_status={},
            created_at=past,
            expires_at=past + timedelta(hours=1),
            status="pending",
        )
        interface._requests[request.request_id] = request

        cleaned = interface.cleanup_expired()

        assert cleaned >= 1

        updated = interface.get_request("req-cleanup")
        assert updated is not None
        assert updated.status == "rejected"

    def test_get_all_requests(self, interface):
        """Alle Anfragen abrufen."""
        for i in range(3):
            action = AutonomousAction(
                action_id=f"action-all-{i}",
                action_type="test",
                timestamp=datetime.utcnow(),
                status="awaiting_approval",
                confidence=0.75,
            )
            interface.create_approval_request(action)

        all_requests = interface.get_all_requests()

        assert len(all_requests) >= 3

    def test_clear_history(self, interface):
        """History löschen."""
        action = AutonomousAction(
            action_id="action-clear",
            action_type="test",
            timestamp=datetime.utcnow(),
            status="awaiting_approval",
            confidence=0.75,
        )
        interface.create_approval_request(action)

        interface.clear_history()

        all_requests = interface.get_all_requests()
        assert len(all_requests) == 0


class TestCreateApprovalInterface:
    """Tests für create_approval_interface Funktion."""

    def test_create_interface(self):
        """Interface erstellen."""
        orchestrator = create_autonomy_orchestrator()
        interface = create_approval_interface(orchestrator)

        assert isinstance(interface, ApprovalInterface)
        assert interface.orchestrator is orchestrator

    def test_create_with_custom_expiry(self):
        """Interface mit benutzerdefinierter Expiry-Zeit."""
        orchestrator = create_autonomy_orchestrator()
        interface = create_approval_interface(
            orchestrator, approval_expiry_hours=48
        )

        assert interface._approval_expiry_hours == 48
