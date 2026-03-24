#!/usr/bin/env python3
"""
Autonomy Orchestrator für NeuroWeave.

Zentrale Steuerung der autonomen Aktionen mit Guardrail-Integration.

Phase 4C: Guardrail System & Integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import uuid
import asyncio

from orchestrator.guardrails import GuardrailManager, AutonomyLevel


@dataclass
class AutonomousAction:
    """Eine autonome Aktion."""

    action_id: str
    action_type: Literal[
        "propose_run", "execute_smoke", "promote_candidate", "kill_run"
    ]
    timestamp: datetime
    status: Literal[
        "pending", "executing", "completed", "blocked", "awaiting_approval"
    ]
    confidence: float
    guardrail_checks: Dict[str, bool] = field(default_factory=dict)
    approval_status: Optional[Dict] = None
    result: Optional[Dict] = None
    violation_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Konvertiere zu Dictionary."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "confidence": self.confidence,
            "guardrail_checks": dict(self.guardrail_checks),
            "approval_status": self.approval_status,
            "result": self.result,
            "violation_messages": list(self.violation_messages),
        }


class AutonomyOrchestrator:
    """
    Zentrale Steuerung autonomer Aktionen.

    Workflow:
    1. Action vorschlagen (z.B. "execute run001")
    2. Guardrails prüfen
    3. Bei Bedarf Human-Approval einholen
    4. Action ausführen
    5. Result dokumentieren
    """

    def __init__(
        self,
        guardrail_manager: GuardrailManager,
        action_history_limit: int = 1000,
    ) -> None:
        """
        Initialisiere AutonomyOrchestrator.

        Args:
            guardrail_manager: Manager für Guardrail-Checks
            action_history_limit: Maximale Anzahl gespeicherter Aktionen
        """
        self._guardrail_manager = guardrail_manager
        self._action_history_limit = action_history_limit
        self._actions: Dict[str, AutonomousAction] = {}
        self._action_queue: List[str] = []

    @property
    def guardrail_manager(self) -> GuardrailManager:
        """GuardrailManager zurückgeben."""
        return self._guardrail_manager

    async def propose_action(
        self, action_type: str, context: Dict
    ) -> AutonomousAction:
        """
        Aktion vorschlagen und Guardrails prüfen.

        Args:
            action_type: Typ der Aktion ("propose_run", "execute_smoke", ...)
            context: Kontext für Guardrail-Checks

        Returns:
            AutonomousAction mit Status
        """
        action_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Erstelle Action im pending Status
        action = AutonomousAction(
            action_id=action_id,
            action_type=action_type,  # type: ignore
            timestamp=timestamp,
            status="pending",
            confidence=context.get("confidence", 0.0),
        )

        # Führe Guardrail-Check durch
        allowed, violation_messages = self._guardrail_manager.check_action(
            action_type, context
        )

        action.violation_messages = violation_messages

        # Speichere Guardrail-Check Ergebnisse
        action.guardrail_checks["initial_check"] = allowed

        if not allowed:
            action.status = "blocked"
            self._store_action(action)
            return action

        # Prüfe ob Approval benötigt wird
        if action_type in self._guardrail_manager.config.requires_approval:
            action.status = "awaiting_approval"
            approval_request = self._guardrail_manager.create_approval_request(
                action_type, context
            )
            action.approval_status = {
                "request_id": approval_request["request_id"],
                "status": "pending",
                "requires_approval_from": approval_request["requires_approval_from"],
            }
            self._store_action(action)
            return action

        # Action ist bereit zur Ausführung
        action.status = "pending"
        self._store_action(action)
        self._action_queue.append(action_id)

        return action

    async def execute_action(self, action: AutonomousAction) -> AutonomousAction:
        """
        Aktion ausführen.

        Steps:
        1. Guardrail-Check wiederholen (frische Daten)
        2. Bei Hard-Limit-Verletzung: blockieren
        3. Bei Soft-Limit-Verletzung: warnen und ausführen
        4. Action ausführen
        5. Result speichern

        Args:
            action: Die auszuführende Aktion

        Returns:
            Aktualisierte AutonomousAction
        """
        # Hole aktuelle Version der Action
        current_action = self._actions.get(action.action_id)
        if current_action is None:
            raise ValueError(f"Action {action.action_id} nicht gefunden")

        if current_action.status == "blocked":
            return current_action

        if current_action.status == "awaiting_approval":
            if not self._is_approved(current_action):
                return current_action

        # Wiederhole Guardrail-Check mit frischen Daten
        context = {
            "confidence": current_action.confidence,
        }
        allowed, violation_messages = self._guardrail_manager.check_action(
            current_action.action_type, context
        )

        current_action.guardrail_checks["pre_execution_check"] = allowed

        # Prüfe auf Hard-Limit Verletzung
        has_hard_violation = False
        for guardrail in self._guardrail_manager.guardrails:
            if guardrail.is_hard_limit:
                for msg in violation_messages:
                    if guardrail.name in msg:
                        has_hard_violation = True
                        break

        if has_hard_violation:
            current_action.status = "blocked"
            current_action.violation_messages.extend(violation_messages)
            return current_action

        # Führe Action aus
        current_action.status = "executing"

        try:
            # Simuliere Action-Ausführung
            result = await self._execute_action_internal(current_action)
            current_action.result = result
            current_action.status = "completed"
        except Exception as e:
            current_action.status = "blocked"
            current_action.result = {"error": str(e)}

        return current_action

    async def _execute_action_internal(
        self, action: AutonomousAction
    ) -> Dict[str, Any]:
        """
        Interne Action-Ausführung.

        Args:
            action: Die auszuführende Aktion

        Returns:
            Ergebnis der Ausführung
        """
        # Placeholder für tatsächliche Ausführung
        # Wird von子类 oder Callbacks überschrieben
        await asyncio.sleep(0)  # Yield control

        return {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "executed_at": datetime.utcnow().isoformat(),
            "success": True,
        }

    async def request_approval(self, action: AutonomousAction) -> bool:
        """
        Human-Freigabe einholen.

        Args:
            action: Die Aktion die genehmigt werden muss

        Returns:
            True wenn freigegeben, False wenn abgelehnt
        """
        current_action = self._actions.get(action.action_id)
        if current_action is None:
            raise ValueError(f"Action {action.action_id} nicht gefunden")

        if current_action.status != "awaiting_approval":
            raise ValueError(
                f"Action {action.action_id} ist nicht im awaiting_approval Status"
            )

        # Placeholder für tatsächliche Approval-Logik
        # Wird durch ApprovalInterface implementiert
        await asyncio.sleep(0)  # Yield control

        # Standard: Approval wird als pending markiert
        # Tatsächliche Entscheidung kommt von ApprovalInterface
        return False

    def approve_action(
        self, action_id: str, approved_by: str, justification: Optional[str] = None
    ) -> bool:
        """
        Aktion genehmigen.

        Args:
            action_id: ID der Aktion
            approved_by: Name/Rolle des Genehmigers
            justification: Optionale Begründung

        Returns:
            True wenn erfolgreich genehmigt
        """
        action = self._actions.get(action_id)
        if action is None:
            return False

        if action.status != "awaiting_approval":
            return False

        action.approval_status = {
            "status": "approved",
            "approved_by": approved_by,
            "approved_at": datetime.utcnow().isoformat(),
            "justification": justification,
        }
        action.status = "pending"
        self._action_queue.append(action_id)

        return True

    def reject_action(
        self, action_id: str, rejected_by: str, reason: str
    ) -> bool:
        """
        Aktion ablehnen.

        Args:
            action_id: ID der Aktion
            rejected_by: Name/Rolle des Ablehners
            reason: Ablehnungsgrund

        Returns:
            True wenn erfolgreich abgelehnt
        """
        action = self._actions.get(action_id)
        if action is None:
            return False

        if action.status != "awaiting_approval":
            return False

        action.approval_status = {
            "status": "rejected",
            "rejected_by": rejected_by,
            "rejected_at": datetime.utcnow().isoformat(),
            "reason": reason,
        }
        action.status = "blocked"

        return True

    def _is_approved(self, action: AutonomousAction) -> bool:
        """Prüfen ob Action genehmigt ist."""
        if action.approval_status is None:
            return False
        return action.approval_status.get("status") == "approved"

    def _store_action(self, action: AutonomousAction) -> None:
        """
        Action speichern und History-Limit einhalten.

        Args:
            action: Die zu speichernde Aktion
        """
        self._actions[action.action_id] = action

        # History-Limit einhalten (älteste entfernen)
        while len(self._actions) > self._action_history_limit:
            oldest_id = min(
                self._actions.keys(),
                key=lambda x: self._actions[x].timestamp,
            )
            del self._actions[oldest_id]

    def get_action_history(self, limit: int = 50) -> List[AutonomousAction]:
        """
        Historie der Aktionen.

        Args:
            limit: Maximale Anzahl zurückgegebener Aktionen

        Returns:
            Liste der neuesten Aktionen
        """
        sorted_actions = sorted(
            self._actions.values(), key=lambda x: x.timestamp, reverse=True
        )
        return sorted_actions[:limit]

    def get_action(self, action_id: str) -> Optional[AutonomousAction]:
        """
        Einzelne Aktion abrufen.

        Args:
            action_id: ID der Aktion

        Returns:
            AutonomousAction oder None
        """
        return self._actions.get(action_id)

    def get_statistics(self) -> Dict:
        """
        Autonomie-Statistiken.

        Returns:
            Dictionary mit Statistiken
        """
        actions = list(self._actions.values())

        if not actions:
            return {
                "total_actions": 0,
                "auto_executed": 0,
                "human_approved": 0,
                "blocked_by_guardrails": 0,
                "success_rate": 0.0,
            }

        total = len(actions)
        completed = sum(1 for a in actions if a.status == "completed")
        blocked = sum(1 for a in actions if a.status == "blocked")
        approved = sum(
            1 for a in actions if a.approval_status
            and a.approval_status.get("status") == "approved"
        )

        success_rate = completed / total if total > 0 else 0.0

        return {
            "total_actions": total,
            "auto_executed": completed,
            "human_approved": approved,
            "blocked_by_guardrails": blocked,
            "success_rate": success_rate,
        }

    def get_pending_actions(self) -> List[AutonomousAction]:
        """
        Ausstehende Aktionen.

        Returns:
            Liste der pending Aktionen
        """
        return [
            a for a in self._actions.values()
            if a.status in ("pending", "awaiting_approval")
        ]

    def get_actions_by_status(
        self, status: str
    ) -> List[AutonomousAction]:
        """
        Aktionen nach Status filtern.

        Args:
            status: Status zum Filtern

        Returns:
            Liste der Aktionen mit dem Status
        """
        return [a for a in self._actions.values() if a.status == status]

    def clear_history(self) -> None:
        """Action-History löschen."""
        self._actions.clear()
        self._action_queue.clear()


def create_autonomy_orchestrator(
    autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED,
) -> AutonomyOrchestrator:
    """
    Factory-Funktion für AutonomyOrchestrator.

    Args:
        autonomy_level: Gewünschtes Autonomie-Level

    Returns:
        Konfigurierter AutonomyOrchestrator
    """
    from .guardrails import create_default_guardrails

    config = create_default_guardrails()
    config.level = autonomy_level

    guardrail_manager = GuardrailManager(config)
    orchestrator = AutonomyOrchestrator(guardrail_manager)

    return orchestrator
