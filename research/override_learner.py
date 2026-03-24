#!/usr/bin/env python3
"""
Override Learner für NeuroWeave.

System lernt wenn Human überschreibt.

Phase 4C: Guardrail System & Integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import uuid
from collections import defaultdict


@dataclass
class OverrideEvent:
    """Ein Human-Override."""

    override_id: str
    original_action: str
    original_decision: str  # "execute", "block", "approve"
    human_decision: str  # "block", "execute", "reject"
    justification: Optional[str]
    timestamp: datetime
    context: Dict  # Features, Guardrails, Confidence
    action_type: Optional[str] = None
    confidence_before: Optional[float] = None
    guardrail_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Konvertiere zu Dictionary."""
        return {
            "override_id": self.override_id,
            "original_action": self.original_action,
            "original_decision": self.original_decision,
            "human_decision": self.human_decision,
            "justification": self.justification,
            "timestamp": self.timestamp.isoformat(),
            "context": dict(self.context),
            "action_type": self.action_type,
            "confidence_before": self.confidence_before,
            "guardrail_violations": list(self.guardrail_violations),
        }


class OverrideLearner:
    """
    Lernt aus Human-Overrides.

    Ziel:
    - Thresholds anpassen wenn Human häufig überschreibt
    - Pattern erkennen ("Human blockt immer Feature X")
    - Confidence-Kalibrierung
    """

    def __init__(self, history_limit: int = 1000) -> None:
        """
        Initialisiere OverrideLearner.

        Args:
            history_limit: Maximale Anzahl gespeicherter Override-Events
        """
        self._overrides: List[OverrideEvent] = []
        self._history_limit = history_limit
        self._confidence_calibration_data: List[Dict] = []

    @property
    def overrides(self) -> List[OverrideEvent]:
        """Alle Override-Events zurückgeben."""
        return list(self._overrides)

    def log_override(
        self,
        original_action: str,
        original_decision: str,
        human_decision: str,
        context: Dict,
        justification: Optional[str] = None,
        action_type: Optional[str] = None,
        confidence_before: Optional[float] = None,
        guardrail_violations: Optional[List[str]] = None,
    ) -> OverrideEvent:
        """
        Override dokumentieren.

        Args:
            original_action: Ursprüngliche Aktion
            original_decision: Ursprüngliche System-Entscheidung
            human_decision: Menschliche Entscheidung
            context: Kontext-Informationen
            justification: Optionale Begründung
            action_type: Typ der Aktion
            confidence_before: Confidence vor dem Override
            guardrail_violations: Guardrail-Verletzungen

        Returns:
            Das dokumentierte Override-Event
        """
        override_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        event = OverrideEvent(
            override_id=override_id,
            original_action=original_action,
            original_decision=original_decision,
            human_decision=human_decision,
            justification=justification,
            timestamp=timestamp,
            context=dict(context),
            action_type=action_type,
            confidence_before=confidence_before,
            guardrail_violations=guardrail_violations or [],
        )

        self._overrides.append(event)
        self._enforce_history_limit()

        # Speichere für Confidence-Kalibrierung
        if confidence_before is not None:
            self._confidence_calibration_data.append({
                "predicted_confidence": confidence_before,
                "human_decision": human_decision,
                "timestamp": timestamp,
            })

        return event

    def _enforce_history_limit(self) -> None:
        """Älteste Overrides entfernen wenn Limit erreicht."""
        while len(self._overrides) > self._history_limit:
            self._overrides.pop(0)

    def analyze_override_patterns(self) -> Dict:
        """
        Override-Muster erkennen.

        Returns:
            Dictionary mit erkannten Mustern
        """
        if not self._overrides:
            return {
                "most_overridden_actions": [],
                "common_justifications": [],
                "override_rate_by_action_type": {},
                "total_overrides": 0,
                "time_range_hours": 0,
            }

        # Zähle Overrides nach Action
        action_counts: Dict[str, int] = defaultdict(int)
        for override in self._overrides:
            action_counts[override.original_action] += 1

        # Top überwundene Aktionen
        most_overridden = sorted(
            action_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Häufigste Begründungen
        justification_counts: Dict[str, int] = defaultdict(int)
        for override in self._overrides:
            if override.justification:
                # Normalisiere Begründung (erste 50 Zeichen)
                norm_justification = override.justification[:50].strip()
                justification_counts[norm_justification] += 1

        common_justifications = sorted(
            justification_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Override-Rate nach Action-Type
        action_type_counts: Dict[str, int] = defaultdict(int)
        action_type_total: Dict[str, int] = defaultdict(int)

        for override in self._overrides:
            if override.action_type:
                action_type_counts[override.action_type] += 1
                # Schätzung: Kontext enthält Info über Gesamtaktionen
                action_type_total[override.action_type] = override.context.get(
                    "total_actions_of_type", 1
                )

        override_rate_by_type = {}
        for action_type, count in action_type_counts.items():
            total = action_type_total.get(action_type, 1)
            override_rate_by_type[action_type] = count / total if total > 0 else 0

        # Zeitbereich berechnen
        if self._overrides:
            oldest = min(o.timestamp for o in self._overrides)
            newest = max(o.timestamp for o in self._overrides)
            time_range_hours = (newest - oldest).total_seconds() / 3600
        else:
            time_range_hours = 0

        return {
            "most_overridden_actions": [
                {"action": action, "count": count}
                for action, count in most_overridden
            ],
            "common_justifications": [
                {"justification": just, "count": count}
                for just, count in common_justifications
            ],
            "override_rate_by_action_type": override_rate_by_type,
            "total_overrides": len(self._overrides),
            "time_range_hours": time_range_hours,
        }

    def suggest_threshold_adjustments(self) -> List[Dict]:
        """
        Vorschläge für Threshold-Anpassungen.

        Example:
            "Human blockt 80% von 'exploration' Actions → Exploration-Threshold zu streng"

        Returns:
            Liste von Vorschlägen
        """
        suggestions = []

        if not self._overrides:
            return suggestions

        # Analysiere Overrides nach Entscheidungstyp
        block_overrides = [
            o for o in self._overrides
            if o.original_decision == "execute" and o.human_decision == "block"
        ]

        execute_overrides = [
            o for o in self._overrides
            if o.original_decision == "block" and o.human_decision == "execute"
        ]

        # Prüfe auf systematische Muster
        action_block_counts: Dict[str, int] = defaultdict(int)
        for override in block_overrides:
            action_key = override.action_type or override.original_action
            action_block_counts[action_key] += 1

        # Vorschläge für zu strenge Thresholds
        for action, count in action_block_counts.items():
            total_actions = len([
                o for o in self._overrides
                if (o.action_type or o.original_action) == action
            ])

            if total_actions > 0:
                block_rate = count / total_actions
                if block_rate > 0.7:  # >70% Block-Rate
                    suggestions.append({
                        "type": "threshold_too_strict",
                        "action": action,
                        "block_rate": block_rate,
                        "suggestion": (
                            f"Human blockt {block_rate:.0%} von '{action}' Actions → "
                            "Threshold möglicherweise zu streng"
                        ),
                        "recommendation": "Threshold um 10-20% lockern",
                    })

        # Vorschläge für zu lockere Thresholds
        action_execute_counts: Dict[str, int] = defaultdict(int)
        for override in execute_overrides:
            action_key = override.action_type or override.original_action
            action_execute_counts[action_key] += 1

        for action, count in action_execute_counts.items():
            total_actions = len([
                o for o in self._overrides
                if (o.action_type or o.original_action) == action
            ])

            if total_actions > 0:
                execute_rate = count / total_actions
                if execute_rate > 0.7:  # >70% Execute-Rate
                    suggestions.append({
                        "type": "threshold_too_loose",
                        "action": action,
                        "execute_rate": execute_rate,
                        "suggestion": (
                            f"Human erlaubt {execute_rate:.0%} von '{action}' Actions → "
                            "Threshold möglicherweise zu locker"
                        ),
                        "recommendation": "Threshold um 10-20% verschärfen",
                    })

        # Confidence-basierte Vorschläge
        low_confidence_blocks = [
            o for o in block_overrides
            if o.confidence_before and o.confidence_before < 0.5
        ]

        if len(low_confidence_blocks) > 5:
            suggestions.append({
                "type": "confidence_threshold_too_low",
                "count": len(low_confidence_blocks),
                "suggestion": (
                    f"{len(low_confidence_blocks)} Blocks bei Confidence <50% → "
                    "Mindest-Confidence möglicherweise zu niedrig"
                ),
                "recommendation": "Mindest-Confidence von 0.5 auf 0.6 erhöhen",
            })

        return suggestions

    def calibrate_confidence(
        self,
        predicted_confidence: float,
        actual_success_rate: float,
    ) -> Dict:
        """
        Confidence-Kalibrierung.

        Wenn predicted_confidence=90% aber actual_success_rate=60% → Confidence zu hoch

        Args:
            predicted_confidence: Vom System vorhergesagte Confidence
            actual_success_rate: Tatsächliche Erfolgsrate

        Returns:
            Kalibrierungs-Ergebnis
        """
        # Speichere für spätere Analyse
        self._confidence_calibration_data.append({
            "predicted_confidence": predicted_confidence,
            "actual_success_rate": actual_success_rate,
            "timestamp": datetime.utcnow(),
        })

        # Berechne Kalibrierungs-Faktor
        if predicted_confidence > 0:
            calibration_factor = actual_success_rate / predicted_confidence
        else:
            calibration_factor = 1.0

        # Bestimme Kalibrierungs-Empfehlung
        if calibration_factor < 0.8:
            recommendation = "Confidence zu hoch - um 20% reduzieren"
            adjustment = -0.2
        elif calibration_factor > 1.2:
            recommendation = "Confidence zu niedrig - um 20% erhöhen"
            adjustment = 0.2
        else:
            recommendation = "Confidence gut kalibriert"
            adjustment = 0.0

        return {
            "predicted_confidence": predicted_confidence,
            "actual_success_rate": actual_success_rate,
            "calibration_factor": calibration_factor,
            "recommendation": recommendation,
            "adjustment": adjustment,
            "data_points": len(self._confidence_calibration_data),
        }

    def get_calibration_statistics(self) -> Dict:
        """
        Statistik zur Confidence-Kalibrierung.

        Returns:
            Dictionary mit Kalibrierungs-Statistiken
        """
        if not self._confidence_calibration_data:
            return {
                "data_points": 0,
                "avg_calibration_factor": 1.0,
                "calibration_quality": "unknown",
            }

        # Berechne durchschnittlichen Kalibrierungs-Faktor
        factors = []
        for data in self._confidence_calibration_data:
            pred = data["predicted_confidence"]
            actual = data["actual_success_rate"]
            if pred > 0:
                factors.append(actual / pred)

        if not factors:
            return {
                "data_points": 0,
                "avg_calibration_factor": 1.0,
                "calibration_quality": "unknown",
            }

        avg_factor = sum(factors) / len(factors)

        # Bestimme Kalibrierungs-Qualität
        if 0.9 <= avg_factor <= 1.1:
            quality = "good"
        elif 0.7 <= avg_factor < 0.9 or 1.1 < avg_factor <= 1.3:
            quality = "moderate"
        else:
            quality = "poor"

        return {
            "data_points": len(self._confidence_calibration_data),
            "avg_calibration_factor": avg_factor,
            "calibration_quality": quality,
            "min_factor": min(factors),
            "max_factor": max(factors),
        }

    def get_override_statistics(self, hours: int = 24) -> Dict:
        """
        Override-Statistiken für Zeitraum.

        Args:
            hours: Zeitraum in Stunden

        Returns:
            Dictionary mit Statistiken
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=hours)

        recent_overrides = [
            o for o in self._overrides
            if o.timestamp >= cutoff
        ]

        if not recent_overrides:
            return {
                "total_overrides": 0,
                "by_decision": {},
                "by_action_type": {},
                "time_range_hours": hours,
            }

        # Nach Entscheidung gruppieren
        by_decision: Dict[str, int] = defaultdict(int)
        for override in recent_overrides:
            by_decision[override.human_decision] += 1

        # Nach Action-Type gruppieren
        by_action_type: Dict[str, int] = defaultdict(int)
        for override in recent_overrides:
            action_type = override.action_type or "unknown"
            by_action_type[action_type] += 1

        # Durchschnittliche Confidence vor Override
        confidences = [
            o.confidence_before for o in recent_overrides
            if o.confidence_before is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "total_overrides": len(recent_overrides),
            "by_decision": dict(by_decision),
            "by_action_type": dict(by_action_type),
            "avg_confidence_before": avg_confidence,
            "time_range_hours": hours,
        }

    def get_overrides_by_action_type(
        self, action_type: str
    ) -> List[OverrideEvent]:
        """
        Overrides nach Action-Type filtern.

        Args:
            action_type: Action-Type zum Filtern

        Returns:
            Liste der Overrides
        """
        return [
            o for o in self._overrides
            if o.action_type == action_type
        ]

    def clear_history(self) -> None:
        """Override-History löschen."""
        self._overrides.clear()
        self._confidence_calibration_data.clear()


def create_override_learner(history_limit: int = 1000) -> OverrideLearner:
    """
    Factory-Funktion für OverrideLearner.

    Args:
        history_limit: Maximale Anzahl gespeicherter Override-Events

    Returns:
        Konfigurierter OverrideLearner
    """
    return OverrideLearner(history_limit=history_limit)
