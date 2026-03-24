#!/usr/bin/env python3
"""
Guardrail System für NeuroWeave.

Definition der Autonomie-Grenzen für sichere autonome Operationen.

Phase 4C: Guardrail System & Integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Literal, Optional, Tuple
import uuid


class AutonomyLevel(Enum):
    """Stufen der Autonomie."""

    MANUAL = "manual"  # Alle Aktionen benötigen Human-Freigabe
    ASSISTED = "assisted"  # Vorschläge mit Human-Review
    SUPERVISED = "supervised"  # Auto innerhalb Guardrails, Human-Alert
    AUTONOMOUS = "autonomous"  # Voll-auto mit Human-Oversight


class GuardrailType(Enum):
    """Typen von Guardrails."""

    BUDGET = "budget"  # Ressourcen-Limits
    EXPLORATION = "exploration"  # Exploration-Limits
    CONFIDENCE = "confidence"  # Mindest-Confidence für Auto-Aktionen
    SAFETY = "safety"  # Sicherheits-kritische Aktionen
    SUBMISSION = "submission"  # Submission-Freigabe


@dataclass
class Guardrail:
    """Eine einzelne Guardrail."""

    guardrail_type: GuardrailType
    name: str
    description: str
    threshold: float  # Schwellenwert
    is_hard_limit: bool  # True = darf nicht überschritten werden
    action_on_violation: Literal["block", "warn", "alert_human"]

    def check(self, value: float) -> Tuple[bool, Optional[str]]:
        """
        Prüfen ob Wert Guardrail verletzt.

        Args:
            value: Der zu prüfende Wert

        Returns:
            Tuple aus (passed_violation, violation_message)
            - passed_violation=True: Guardrail wurde NICHT verletzt
            - passed_violation=False: Guardrail wurde verletzt
            - violation_message=None wenn keine Verletzung
        """
        # Für CONFIDENCE und SAFETY: höher ist besser (value muss >= threshold sein)
        if self.guardrail_type in (GuardrailType.CONFIDENCE, GuardrailType.SAFETY):
            if value >= self.threshold:
                return (True, None)
            violation_msg = (
                f"Guardrail '{self.name}' verletzt: "
                f"Wert {value:.2f} ist unter dem Minimum {self.threshold:.2f}"
            )
            return (False, violation_msg)

        # Für SUBMISSION: immer blockieren (threshold=1.0, value wird ignoriert)
        if self.guardrail_type == GuardrailType.SUBMISSION:
            # Submission Guardrail blockiert immer für manuelle Freigabe
            return (False, f"Guardrail '{self.name}': Submission erfordert Human-Freigabe")

        # Für alle anderen (BUDGET, EXPLORATION, etc.): niedriger ist besser
        # value muss <= threshold sein
        if value <= self.threshold:
            return (True, None)

        violation_msg = (
            f"Guardrail '{self.name}' verletzt: "
            f"Wert {value:.2f} überschreitet Threshold {self.threshold:.2f}"
        )
        return (False, violation_msg)


@dataclass
class AutonomyConfig:
    """Konfiguration für Autonomie-Level."""

    level: AutonomyLevel
    guardrails: List[Guardrail] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=list)
    requires_approval: List[str] = field(default_factory=list)


class GuardrailManager:
    """
    Verwaltet Guardrails für sichere Autonomie.

    Default Guardrails:
    1. BUDGET: Max GPU-hours/Woche
    2. EXPLORATION: Max 50% explorative Runs
    3. CONFIDENCE: Min 60% für Auto-Aktionen
    4. SAFETY: Keine teuren Remote-Runs ohne Approval
    5. SUBMISSION: Submission immer mit Human-Freigabe
    """

    def __init__(self, config: AutonomyConfig) -> None:
        """
        Initialisiere GuardrailManager.

        Args:
            config: Autonomie-Konfiguration mit Guardrails
        """
        self._config = config
        self._guardrails_by_type: Dict[GuardrailType, List[Guardrail]] = {}
        self._group_guardrails()

    def _group_guardrails(self) -> None:
        """Gruppiere Guardrails nach Typ für schnellen Zugriff."""
        self._guardrails_by_type = {}
        for guardrail in self._config.guardrails:
            guardrail_list = self._guardrails_by_type.get(guardrail.guardrail_type, [])
            guardrail_list.append(guardrail)
            self._guardrails_by_type[guardrail.guardrail_type] = guardrail_list

    @property
    def config(self) -> AutonomyConfig:
        """Konfiguration zurückgeben."""
        return self._config

    @property
    def guardrails(self) -> List[Guardrail]:
        """Alle Guardrails zurückgeben."""
        return list(self._config.guardrails)

    def check_action(self, action: str, context: Dict) -> Tuple[bool, List[str]]:
        """
        Prüfen ob Aktion erlaubt ist.

        Args:
            action: "execute_run", "promote_run", "submit_bundle", ...
            context: {"budget_used": ..., "confidence": ..., "is_exploration": ...}

        Returns:
            Tuple aus (allowed, violation_messages)
            - allowed=True: Aktion ist erlaubt
            - allowed=False: Aktion ist blockiert
            - violation_messages: Liste der Verletzungsnachrichten
        """
        violation_messages: List[str] = []

        # Prüfe ob Aktion im allowed_actions Liste ist
        if self._config.allowed_actions and action not in self._config.allowed_actions:
            violation_messages.append(
                f"Aktion '{action}' ist nicht in der Liste erlaubter Aktionen"
            )
            return (False, violation_messages)

        # Prüfe alle Guardrails
        for guardrail in self._config.guardrails:
            value = self._extract_context_value(guardrail, context)
            if value is None:
                continue

            passed, violation_msg = guardrail.check(value)
            if not passed and violation_msg:
                violation_messages.append(violation_msg)

                # Bei Hard-Limit sofort blockieren
                if guardrail.is_hard_limit:
                    return (False, violation_messages)

        # Prüfe ob Aktion Approval benötigt
        if action in self._config.requires_approval:
            # Aktion ist erlaubt, benötigt aber Approval
            # Wird separat durch request_approval behandelt
            pass

        is_allowed = len(violation_messages) == 0
        return (is_allowed, violation_messages)

    def _extract_context_value(
        self, guardrail: Guardrail, context: Dict
    ) -> Optional[float]:
        """
        Extrahiere den relevanten Wert aus dem Kontext für eine Guardrail.

        Args:
            guardrail: Die Guardrail für die der Wert extrahiert wird
            context: Der Kontext mit Werten

        Returns:
            Der extrahierte Wert oder None wenn nicht vorhanden
        """
        mapping: Dict[GuardrailType, str] = {
            GuardrailType.BUDGET: "budget_used",
            GuardrailType.EXPLORATION: "exploration_ratio",
            GuardrailType.CONFIDENCE: "confidence",
            GuardrailType.SAFETY: "safety_score",
            GuardrailType.SUBMISSION: "submission_ready",
        }

        context_key = mapping.get(guardrail.guardrail_type)
        if not context_key:
            return None

        value = context.get(context_key)
        if value is None:
            return None

        return float(value)

    def get_required_approvals(self, action: str) -> List[str]:
        """
        Benötigte Freigaben für Aktion.

        Args:
            action: Die Aktion für die Freigaben benötigt werden

        Returns:
            Liste von Rollen die genehmigen müssen, z.B.
            ["budget_owner", "technical_lead"] oder []
        """
        approval_mapping: Dict[str, List[str]] = {
            "submit_bundle": ["technical_lead", "project_owner"],
            "execute_expensive_run": ["budget_owner"],
            "promote_to_submission": ["technical_lead"],
            "override_guardrail": ["safety_officer"],
        }

        return list(approval_mapping.get(action, []))

    def create_approval_request(self, action: str, context: Dict) -> Dict:
        """
        Freigabe-Anfrage erstellen.

        Args:
            action: Die Aktion die genehmigt werden muss
            context: Kontext für die Entscheidung

        Returns:
            Dictionary mit Anfrage-Informationen
        """
        request_id = str(uuid.uuid4())

        # Bestimme Risk-Level basierend auf Aktion und Kontext
        risk_level = self._determine_risk_level(action, context)

        # Bestimme benötigte Approvals
        requires_approval_from = self.get_required_approvals(action)

        # Erstelle Begründung
        reason = self._create_approval_reason(action, context)

        return {
            "request_id": request_id,
            "action": action,
            "reason": reason,
            "risk_level": risk_level,
            "requires_approval_from": requires_approval_from,
            "context": dict(context),
            "created_at": datetime.utcnow().isoformat(),
        }

    def _determine_risk_level(
        self, action: str, context: Dict
    ) -> Literal["low", "medium", "high"]:
        """
        Risk-Level für eine Aktion bestimmen.

        Args:
            action: Die Aktion
            context: Kontext-Informationen

        Returns:
            Risk-Level als String
        """
        # High-Risk Aktionen
        high_risk_actions = ["submit_bundle", "override_guardrail", "execute_expensive_run"]
        if action in high_risk_actions:
            return "high"

        # Medium-Risk basierend auf Kontext
        confidence = context.get("confidence", 1.0)
        if confidence < 0.7:
            return "medium"

        budget_used = context.get("budget_used", 0.0)
        if budget_used > 0.8:
            return "medium"

        return "low"

    def _create_approval_reason(self, action: str, context: Dict) -> str:
        """
        Begründung für Approval-Anfrage erstellen.

        Args:
            action: Die Aktion
            context: Kontext-Informationen

        Returns:
            Begründungstext
        """
        reasons: Dict[str, str] = {
            "submit_bundle": "Submission Bundle erfordert Human-Freigabe gemäß Guardrail",
            "execute_expensive_run": "Teurer Run überschreitet Budget-Threshold",
            "promote_to_submission": "Promotion zu Submission erfordert Review",
            "override_guardrail": "Guardrail-Override erfordert Safety-Approval",
        }

        base_reason = reasons.get(action, f"Aktion '{action}' erfordert Human-Freigabe")

        # Füge Kontext-Details hinzu
        details = []
        if "confidence" in context:
            details.append(f"Confidence: {context['confidence']:.1%}")
        if "budget_used" in context:
            details.append(f"Budget verwendet: {context['budget_used']:.1%}")

        if details:
            return f"{base_reason} ({', '.join(details)})"

        return base_reason

    def add_guardrail(self, guardrail: Guardrail) -> None:
        """
        Neue Guardrail hinzufügen.

        Args:
            guardrail: Die hinzuzufügende Guardrail
        """
        self._config.guardrails.append(guardrail)
        self._group_guardrails()

    def remove_guardrail(self, guardrail_name: str) -> bool:
        """
        Guardrail entfernen.

        Args:
            guardrail_name: Name der zu entfernenden Guardrail

        Returns:
            True wenn entfernt, False wenn nicht gefunden
        """
        for i, guardrail in enumerate(self._config.guardrails):
            if guardrail.name == guardrail_name:
                self._config.guardrails.pop(i)
                self._group_guardrails()
                return True
        return False

    def get_guardrail_status(self) -> Dict:
        """
        Status aller Guardrails zurückgeben.

        Returns:
            Dictionary mit Status-Informationen
        """
        status = {
            "autonomy_level": self._config.level.value,
            "total_guardrails": len(self._config.guardrails),
            "guardrails_by_type": {},
        }

        for guardrail_type, guardrails in self._guardrails_by_type.items():
            type_status = []
            for guardrail in guardrails:
                type_status.append(
                    {
                        "name": guardrail.name,
                        "threshold": guardrail.threshold,
                        "is_hard_limit": guardrail.is_hard_limit,
                        "action_on_violation": guardrail.action_on_violation,
                    }
                )
            status["guardrails_by_type"][guardrail_type.value] = type_status

        return status


def create_default_guardrails() -> AutonomyConfig:
    """
    Default Guardrails für NeuroWeave erstellen.

    Returns:
        AutonomyConfig mit vorkonfigurierten Guardrails
    """
    guardrails = [
        Guardrail(
            guardrail_type=GuardrailType.BUDGET,
            name="Weekly GPU Budget",
            description="Maximale GPU-hours pro Woche",
            threshold=100.0,  # 100 GPU-hours
            is_hard_limit=True,
            action_on_violation="block",
        ),
        Guardrail(
            guardrail_type=GuardrailType.EXPLORATION,
            name="Exploration Ratio",
            description="Maximaler Anteil explorativer Runs",
            threshold=0.5,  # 50%
            is_hard_limit=False,
            action_on_violation="warn",
        ),
        Guardrail(
            guardrail_type=GuardrailType.CONFIDENCE,
            name="Minimum Confidence",
            description="Mindest-Confidence für Auto-Aktionen",
            threshold=0.6,  # 60%
            is_hard_limit=True,
            action_on_violation="block",
        ),
        Guardrail(
            guardrail_type=GuardrailType.SAFETY,
            name="Safety Score",
            description="Mindest-Safety-Score für Remote-Runs",
            threshold=0.8,  # 80%
            is_hard_limit=True,
            action_on_violation="block",
        ),
        Guardrail(
            guardrail_type=GuardrailType.SUBMISSION,
            name="Submission Readiness",
            description="Submission immer mit Human-Freigabe",
            threshold=1.0,  # Immer blockieren für manuelle Freigabe
            is_hard_limit=True,
            action_on_violation="alert_human",
        ),
    ]

    allowed_actions = [
        "propose_runs",
        "execute_smoke",
        "promote_candidate",
        "kill_run",
        "analyze_results",
    ]

    requires_approval = [
        "submit_bundle",
        "execute_expensive_run",
        "promote_to_submission",
        "override_guardrail",
    ]

    return AutonomyConfig(
        level=AutonomyLevel.SUPERVISED,
        guardrails=guardrails,
        allowed_actions=allowed_actions,
        requires_approval=requires_approval,
    )
