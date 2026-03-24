#!/usr/bin/env python3
"""
Human-on-the-loop Interface für NeuroWeave.

Dashboard mit Approval-Workflow für autonome Aktionen.

Phase 4C: Guardrail System & Integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional
import uuid

from orchestrator.autonomy_orchestrator import AutonomyOrchestrator, AutonomousAction


@dataclass
class ApprovalRequest:
    """Eine Freigabe-Anfrage."""

    request_id: str
    action_type: str
    description: str
    risk_level: Literal["low", "medium", "high"]
    confidence: float
    guardrail_status: Dict[str, bool]
    created_at: datetime
    expires_at: datetime
    status: Literal["pending", "approved", "rejected"]
    action_id: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejected_by: Optional[str] = None
    rejected_at: Optional[datetime] = None
    justification: Optional[str] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Konvertiere zu Dictionary."""
        return {
            "request_id": self.request_id,
            "action_type": self.action_type,
            "description": self.description,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "guardrail_status": dict(self.guardrail_status),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status,
            "action_id": self.action_id,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejected_by": self.rejected_by,
            "rejected_at": self.rejected_at.isoformat() if self.rejected_at else None,
            "justification": self.justification,
            "rejection_reason": self.rejection_reason,
        }

    def is_expired(self) -> bool:
        """Prüfen ob Anfrage abgelaufen ist."""
        return datetime.utcnow() > self.expires_at

    def is_pending(self) -> bool:
        """Prüfen ob Anfrage noch ausstehend ist."""
        return self.status == "pending" and not self.is_expired()


class ApprovalInterface:
    """
    Interface für Human-Freigaben.

    Features:
    - Pending Approvals auflisten
    - Approval/Rejection mit Begründung
    - Bulk-Approval für Low-Risk Actions
    - Expiry-Handling
    """

    def __init__(
        self,
        orchestrator: AutonomyOrchestrator,
        approval_expiry_hours: int = 24,
    ) -> None:
        """
        Initialisiere ApprovalInterface.

        Args:
            orchestrator: Der AutonomyOrchestrator
            approval_expiry_hours: Stunden nach denen Approvals verfallen
        """
        self._orchestrator = orchestrator
        self._approval_expiry_hours = approval_expiry_hours
        self._requests: Dict[str, ApprovalRequest] = {}

    @property
    def orchestrator(self) -> AutonomyOrchestrator:
        """Orchestrator zurückgeben."""
        return self._orchestrator

    def create_approval_request(
        self,
        action: AutonomousAction,
        description: Optional[str] = None,
    ) -> ApprovalRequest:
        """
        Neue Approval-Anfrage erstellen.

        Args:
            action: Die Aktion die genehmigt werden muss
            description: Optionale Beschreibung

        Returns:
            Die erstellte ApprovalRequest
        """
        request_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Bestimme Risk-Level aus Action-Kontext
        risk_level = self._determine_risk_level(action)

        # Erstelle Beschreibung falls nicht angegeben
        if description is None:
            description = (
                f"Autonome Aktion '{action.action_type}' "
                f"(Confidence: {action.confidence:.1%})"
            )

        request = ApprovalRequest(
            request_id=request_id,
            action_type=action.action_type,
            description=description,
            risk_level=risk_level,
            confidence=action.confidence,
            guardrail_status=dict(action.guardrail_checks),
            created_at=now,
            expires_at=now + timedelta(hours=self._approval_expiry_hours),
            status="pending",
            action_id=action.action_id,
        )

        self._requests[request_id] = request
        return request

    def _determine_risk_level(self, action: AutonomousAction) -> Literal[
        "low", "medium", "high"
    ]:
        """
        Risk-Level für eine Aktion bestimmen.

        Args:
            action: Die Aktion

        Returns:
            Risk-Level
        """
        # High-Risk Aktionen
        high_risk_types = ["submit_bundle", "override_guardrail"]
        if action.action_type in high_risk_types:
            return "high"

        # Risk-Level basierend auf Confidence
        if action.confidence < 0.5:
            return "high"
        elif action.confidence < 0.7:
            return "medium"

        # Risk-Level basierend auf Guardrail-Checks
        failed_checks = sum(
            1 for passed in action.guardrail_checks.values() if not passed
        )
        if failed_checks > 0:
            return "medium"

        return "low"

    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """
        Ausstehende Freigaben.

        Returns:
            Liste der pending Approval-Anfragen
        """
        pending = []
        for request in self._requests.values():
            if request.is_pending():
                pending.append(request)

        # Sortiere nach created_at (neueste zuerst)
        pending.sort(key=lambda x: x.created_at, reverse=True)
        return pending

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """
        Einzelne Anfrage abrufen.

        Args:
            request_id: ID der Anfrage

        Returns:
            ApprovalRequest oder None
        """
        return self._requests.get(request_id)

    def approve(
        self, request_id: str, justification: Optional[str] = None
    ) -> bool:
        """
        Freigabe erteilen.

        Args:
            request_id: Anfrage-ID
            justification: Optionale Begründung

        Returns:
            True wenn erfolgreich genehmigt
        """
        request = self._requests.get(request_id)
        if request is None:
            return False

        if not request.is_pending():
            return False

        # Update Request-Status
        request.status = "approved"
        request.approved_by = "human_user"  # Wird von außen gesetzt
        request.approved_at = datetime.utcnow()
        request.justification = justification

        # Update Action im Orchestrator
        if request.action_id:
            self._orchestrator.approve_action(
                request.action_id,
                approved_by=request.approved_by,
                justification=justification,
            )

        return True

    def reject(self, request_id: str, reason: str) -> bool:
        """
        Freigabe ablehnen.

        Args:
            request_id: Anfrage-ID
            reason: Ablehnungsgrund

        Returns:
            True wenn erfolgreich abgelehnt
        """
        request = self._requests.get(request_id)
        if request is None:
            return False

        if not request.is_pending():
            return False

        # Update Request-Status
        request.status = "rejected"
        request.rejected_by = "human_user"  # Wird von außen gesetzt
        request.rejected_at = datetime.utcnow()
        request.rejection_reason = reason

        # Update Action im Orchestrator
        if request.action_id:
            self._orchestrator.reject_action(
                request.action_id,
                rejected_by=request.rejected_by,
                reason=reason,
            )

        return True

    def bulk_approve(
        self, request_ids: List[str], risk_threshold: str = "low"
    ) -> Dict[str, bool]:
        """
        Bulk-Freigabe für Low-Risk Actions.

        Args:
            request_ids: Liste von Anfrage-IDs
            risk_threshold: Max Risk-Level für Bulk-Approval

        Returns:
            Dictionary mit request_id -> success Status
        """
        results: Dict[str, bool] = {}

        # Definiere Risk-Level Reihenfolge
        risk_order = {"low": 0, "medium": 1, "high": 2}
        max_risk_level = risk_order.get(risk_threshold, 0)

        for request_id in request_ids:
            request = self._requests.get(request_id)
            if request is None:
                results[request_id] = False
                continue

            # Prüfe Risk-Level
            request_risk = risk_order.get(request.risk_level, 2)
            if request_risk > max_risk_level:
                results[request_id] = False
                continue

            # Genehmige
            success = self.approve(request_id)
            results[request_id] = success

        return results

    def get_approval_statistics(self) -> Dict:
        """
        Freigabe-Statistiken.

        Returns:
            Dictionary mit Statistiken
        """
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        requests = list(self._requests.values())

        pending_count = sum(1 for r in requests if r.is_pending())
        approved_today = sum(
            1
            for r in requests
            if r.status == "approved"
            and r.approved_at
            and r.approved_at >= today_start
        )
        rejected_today = sum(
            1
            for r in requests
            if r.status == "rejected"
            and r.rejected_at
            and r.rejected_at >= today_start
        )

        # Berechne durchschnittliche Approval-Zeit
        approval_times = []
        for r in requests:
            if r.status == "approved" and r.approved_at:
                delta = r.approved_at - r.created_at
                approval_times.append(delta.total_seconds())

        avg_approval_time = "N/A"
        if approval_times:
            avg_seconds = sum(approval_times) / len(approval_times)
            hours = int(avg_seconds // 3600)
            minutes = int((avg_seconds % 3600) // 60)
            avg_approval_time = f"{hours}h {minutes}m"

        return {
            "pending": pending_count,
            "approved_today": approved_today,
            "rejected_today": rejected_today,
            "avg_approval_time": avg_approval_time,
            "total_requests": len(requests),
        }

    def get_requests_by_risk_level(
        self, risk_level: str
    ) -> List[ApprovalRequest]:
        """
        Anfragen nach Risk-Level filtern.

        Args:
            risk_level: Risk-Level zum Filtern

        Returns:
            Liste der Anfragen mit dem Risk-Level
        """
        return [
            r for r in self._requests.values()
            if r.risk_level == risk_level
        ]

    def get_expired_requests(self) -> List[ApprovalRequest]:
        """
        Abgelaufene Anfragen.

        Returns:
            Liste der abgelaufenen Anfragen
        """
        return [r for r in self._requests.values() if r.is_expired()]

    def cleanup_expired(self) -> int:
        """
        Abgelaufene Anfragen bereinigen.

        Returns:
            Anzahl bereinigter Anfragen
        """
        expired_ids = [
            rid for rid, req in self._requests.items()
            if req.is_expired()
        ]

        for request_id in expired_ids:
            request = self._requests[request_id]
            if request.status == "pending":
                request.status = "rejected"
                request.rejection_reason = "Expired (timeout)"

        return len(expired_ids)

    def get_all_requests(self) -> List[ApprovalRequest]:
        """
        Alle Anfragen.

        Returns:
            Liste aller Approval-Anfragen
        """
        return list(self._requests.values())

    def clear_history(self) -> None:
        """Alle Anfragen löschen."""
        self._requests.clear()


def create_approval_interface(
    orchestrator: AutonomyOrchestrator,
    approval_expiry_hours: int = 24,
) -> ApprovalInterface:
    """
    Factory-Funktion für ApprovalInterface.

    Args:
        orchestrator: Der AutonomyOrchestrator
        approval_expiry_hours: Stunden nach denen Approvals verfallen

    Returns:
        Konfiguriertes ApprovalInterface
    """
    return ApprovalInterface(
        orchestrator=orchestrator,
        approval_expiry_hours=approval_expiry_hours,
    )
