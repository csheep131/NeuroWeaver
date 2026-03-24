#!/usr/bin/env python3
"""
Rollback Manager für NeuroWeave Phase 4B.

Automatisches Recovery-System bei Fehlern.

Prinzipien:
- Inkrementelles Rollback (nur problematische Teile)
- Learning from Rollbacks (Dokumentation)
- Fallback auf letzte stabile Konfiguration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.registry import RunEntry, RunRegistry
from research.failure_classifier import FailureDiagnosis


@dataclass
class RollbackPlan:
    """
    Plan für Rollback.

    Attributes:
        failed_run_id: ID des fehlgeschlagenen Runs
        rollback_target: Parent-Run oder letzte stabile Konfiguration
        changes_to_revert: Liste der zurückzunehmenden Features
        changes_to_keep: Erfolgreiche Teile die behalten werden
        estimated_recovery_time: Geschätzte Zeit für Recovery
        success_probability: Erfolgswahrscheinlichkeit des Rollbacks
    """

    failed_run_id: str
    rollback_target: str
    changes_to_revert: List[str]
    changes_to_keep: List[str]
    estimated_recovery_time: str
    success_probability: float

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung."""
        return {
            "failed_run_id": self.failed_run_id,
            "rollback_target": self.rollback_target,
            "changes_to_revert": self.changes_to_revert,
            "changes_to_keep": self.changes_to_keep,
            "estimated_recovery_time": self.estimated_recovery_time,
            "success_probability": self.success_probability,
        }


@dataclass
class RollbackRecord:
    """
    Dokumentation eines durchgeführten Rollbacks.

    Attributes:
        rollback_id: Eindeutige ID des Rollbacks
        failed_run_id: ID des fehlgeschlagenen Runs
        target_run_id: ID des Ziel-Runs nach Rollback
        plan: Der ausgeführte Rollback-Plan
        outcome: Ergebnis ("success", "partial", "failed")
        actual_recovery_time: Tatsächliche Zeit für Recovery
        lessons_learned: Erkenntnisse für zukünftige Rollbacks
        timestamp: Zeitpunkt des Rollbacks
    """

    rollback_id: str
    failed_run_id: str
    target_run_id: str
    plan: RollbackPlan
    outcome: Literal["success", "partial", "failed"]
    actual_recovery_time: Optional[str]
    lessons_learned: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung."""
        return {
            "rollback_id": self.rollback_id,
            "failed_run_id": self.failed_run_id,
            "target_run_id": self.target_run_id,
            "plan": self.plan.to_dict(),
            "outcome": self.outcome,
            "actual_recovery_time": self.actual_recovery_time,
            "lessons_learned": self.lessons_learned,
            "timestamp": self.timestamp,
        }


class RollbackManager:
    """
    Automatisches Recovery bei Fehlern.

    Verwaltet Rollback-Pläne, führt Rollbacks durch und dokumentiert
    Ergebnisse für kontinuierliche Verbesserung.

    Example:
        registry = RunRegistry()
        manager = RollbackManager(registry)
        
        diagnosis = failure_classifier.classify("run017", registry)
        if diagnosis:
            plan = manager.create_rollback_plan("run017", diagnosis)
            new_run_id = manager.execute_rollback(plan)
    """

    def __init__(self, registry: RunRegistry, rollback_log_path: str = "results/rollback_log.json"):
        """
        Initialisiere RollbackManager.

        Args:
            registry: RunRegistry für Datenzugriff
            rollback_log_path: Pfad zum Rollback-Log
        """
        self.registry = registry
        self.rollback_log_path = Path(rollback_log_path)
        self._rollback_history: List[RollbackRecord] = []
        self._load_history()

    def _load_history(self) -> None:
        """Lade Rollback-Historie von Disk."""
        if self.rollback_log_path.exists():
            with open(self.rollback_log_path, "r") as f:
                data = json.load(f)
                for record_data in data:
                    plan_data = record_data.get("plan", {})
                    plan = RollbackPlan(
                        failed_run_id=plan_data.get("failed_run_id", ""),
                        rollback_target=plan_data.get("rollback_target", ""),
                        changes_to_revert=plan_data.get("changes_to_revert", []),
                        changes_to_keep=plan_data.get("changes_to_keep", []),
                        estimated_recovery_time=plan_data.get("estimated_recovery_time", ""),
                        success_probability=plan_data.get("success_probability", 0.0),
                    )
                    record = RollbackRecord(
                        rollback_id=record_data.get("rollback_id", ""),
                        failed_run_id=record_data.get("failed_run_id", ""),
                        target_run_id=record_data.get("target_run_id", ""),
                        plan=plan,
                        outcome=record_data.get("outcome", "success"),
                        actual_recovery_time=record_data.get("actual_recovery_time"),
                        lessons_learned=record_data.get("lessons_learned", []),
                        timestamp=record_data.get("timestamp", ""),
                    )
                    self._rollback_history.append(record)

    def _save_history(self) -> None:
        """Speichere Rollback-Historie auf Disk."""
        self.rollback_log_path.parent.mkdir(parents=True, exist_ok=True)
        data = [record.to_dict() for record in self._rollback_history]
        with open(self.rollback_log_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_rollback_id(self) -> str:
        """Generiere eindeutige Rollback-ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"rb_{timestamp}_{len(self._rollback_history) + 1:03d}"

    def _extract_features_from_run(self, run_id: str) -> List[str]:
        """
        Extrahiere Features aus einem Run.

        Args:
            run_id: Run-ID

        Returns:
            Liste von Feature-Namen
        """
        entry = self.registry.get(run_id)
        if entry is None:
            return []

        # Features aus Config extrahieren (wenn verfügbar)
        config_path = Path("configs") / "runs" / f"{run_id}.yaml"
        if config_path.exists():
            # Vereinfachte Feature-Extraktion
            # In Produktion: Config laden und Features parsen
            pass

        # Fallback: Features aus delta_bpb und delta_ms ableiten
        features = []

        if entry.delta_bpb is not None:
            if entry.delta_bpb > 0.1:
                features.append("unstable_feature")
            if entry.delta_bpb > 0.5:
                features.append("high_impact_feature")

        if entry.delta_ms is not None:
            if entry.delta_ms > 0.3:
                features.append("slow_feature")

        return features

    def _identify_problematic_features(
        self,
        failed_run_id: str,
        diagnosis: FailureDiagnosis
    ) -> List[str]:
        """
        Identifiziere problematische Features basierend auf Diagnose.

        Args:
            failed_run_id: ID des fehlgeschlagenen Runs
            diagnosis: FailureDiagnosis

        Returns:
            Liste von problematischen Features
        """
        problematic = []

        # Contributing Factors analysieren
        for factor in diagnosis.contributing_factors:
            factor_lower = factor.lower()
            if "seed" in factor_lower:
                problematic.append("seed_sensitivity")
            if "parent" in factor_lower:
                problematic.append("parent_lineage")
            if "bpb" in factor_lower:
                problematic.append("performance_degradation")
            if "step" in factor_lower or "time" in factor_lower:
                problematic.append("slowdown")

        # Root Cause analysieren
        root_cause_lower = diagnosis.root_cause.lower()
        if "learning rate" in root_cause_lower:
            problematic.append("learning_rate")
        if "depth" in root_cause_lower:
            problematic.append("model_depth")
        if "batch" in root_cause_lower:
            problematic.append("batch_size")
        if "quant" in root_cause_lower:
            problematic.append("quantization")
        if "activation" in root_cause_lower:
            problematic.append("activation_type")
        if "gradient" in root_cause_lower:
            problematic.append("gradient_clipping")

        return list(set(problematic))

    def create_rollback_plan(
        self,
        failed_run_id: str,
        failure_diagnosis: FailureDiagnosis
    ) -> RollbackPlan:
        """
        Rollback-Plan für fehlgeschlagenen Run erstellen.

        Strategie:
        1. Identifiziere problematische Features (aus FailureDiagnosis)
        2. Finde letzte stabile Konfiguration (Parent oder früher)
        3. Behalte erfolgreiche Teile bei

        Args:
            failed_run_id: ID des fehlgeschlagenen Runs
            failure_diagnosis: Diagnose des Fehlers

        Returns:
            RollbackPlan für die Recovery
        """
        failed_entry = self.registry.get(failed_run_id)
        if failed_entry is None:
            raise ValueError(f"Run '{failed_run_id}' nicht gefunden")

        # 1. Problematische Features identifizieren
        changes_to_revert = self._identify_problematic_features(
            failed_run_id, failure_diagnosis
        )

        # 2. Letzte stabile Konfiguration finden
        last_stable = self.get_last_stable_configuration(failed_run_id)

        if last_stable is None:
            # Fallback: Parent verwenden
            if failed_entry.parent_run_id:
                last_stable = failed_entry.parent_run_id
            else:
                # Kein Parent verfügbar: Root-Config als Fallback
                last_stable = "base_config"

        # 3. Changes to keep bestimmen (erfolgreiche Teile)
        changes_to_keep = []

        # Wenn Parent existiert und erfolgreich war, behalte Parent-Features
        if failed_entry.parent_run_id:
            parent = self.registry.get(failed_entry.parent_run_id)
            if parent and parent.status == "completed":
                # Parent-Features behalten
                parent_features = self._extract_features_from_run(parent.run_id)
                changes_to_keep = [f for f in parent_features if f not in changes_to_revert]

        # 4. Erfolgswahrscheinlichkeit schätzen
        success_probability = self._estimate_success_probability(
            failed_run_id, last_stable, failure_diagnosis
        )

        # 5. Geschätzte Recovery-Zeit
        estimated_time = self._estimate_recovery_time(failure_diagnosis.failure_category)

        return RollbackPlan(
            failed_run_id=failed_run_id,
            rollback_target=last_stable,
            changes_to_revert=changes_to_revert,
            changes_to_keep=changes_to_keep,
            estimated_recovery_time=estimated_time,
            success_probability=success_probability,
        )

    def _estimate_success_probability(
        self,
        failed_run_id: str,
        rollback_target: str,
        diagnosis: FailureDiagnosis
    ) -> float:
        """
        Schätze Erfolgswahrscheinlichkeit des Rollbacks.

        Args:
            failed_run_id: ID des fehlgeschlagenen Runs
            rollback_target: Ziel des Rollbacks
            diagnosis: FailureDiagnosis

        Returns:
            Wahrscheinlichkeit zwischen 0 und 1
        """
        # Basis-Wahrscheinlichkeit
        base_prob = 0.7

        # Anpassung basierend auf Diagnose-Konfidenz
        confidence_bonus = diagnosis.confidence * 0.15

        # Anpassung basierend auf Similar Failures
        if diagnosis.similar_failures:
            # Ähnliche Fehler erfolgreich recovered?
            similar_success = sum(
                1 for record in self._rollback_history
                if record.failed_run_id in diagnosis.similar_failures
                and record.outcome == "success"
            )
            if similar_success > 0:
                base_prob += 0.1

        # Begrenzen auf [0.3, 0.95]
        probability = min(0.95, max(0.3, base_prob + confidence_bonus))

        return probability

    def _estimate_recovery_time(self, failure_category: str) -> str:
        """
        Schätze Recovery-Zeit basierend auf Fehlerkategorie.

        Args:
            failure_category: Fehlerkategorie

        Returns:
            Geschätzte Zeit (z.B. "30min", "2h")
        """
        time_estimates = {
            "oom": "1h",  # Config-Änderung + Run
            "nan_gradients": "45min",  # LR-Anpassung + Run
            "training_divergence": "1h",  # Feature-Removal + Run
            "quant_explosion": "30min",  # Quant-Config zurücksetzen
            "performance_regression": "45min",  # Feature-Optimierung
        }

        return time_estimates.get(failure_category, "1h")

    def execute_rollback(self, plan: RollbackPlan) -> str:
        """
        Rollback ausführen.

        Erstellt einen neuen Run basierend auf der Rollback-Konfiguration.

        Args:
            plan: Rollback-Plan

        Returns:
            run_id des neuen Runs (Rollback-Ziel)
        """
        start_time = datetime.now()

        # Neuen Run registrieren
        rollback_id = self._generate_rollback_id()
        target_config_hash = f"rollback_{plan.rollback_target}"

        # Neuen Run im Registry eintragen
        new_entry = self.registry.register(
            run_id=rollback_id,
            config_hash=target_config_hash,
            parent_run_id=plan.rollback_target if plan.rollback_target != "base_config" else None,
        )

        # Run als "pending" markieren mit Rollback-Info
        self.registry.update(
            rollback_id,
            notes=f"Rollback von {plan.failed_run_id}. "
                  f"Revert: {', '.join(plan.changes_to_revert)}. "
                  f"Keep: {', '.join(plan.changes_to_keep)}",
            tags=["rollback", f"from:{plan.failed_run_id}"],
        )

        # Rollback dokumentieren
        # Hinweis: outcome wird später aktualisiert wenn Run abgeschlossen
        record = RollbackRecord(
            rollback_id=rollback_id,
            failed_run_id=plan.failed_run_id,
            target_run_id=rollback_id,
            plan=plan,
            outcome="success",  # Wird später aktualisiert
            actual_recovery_time=None,
            lessons_learned=[],
            timestamp=start_time.isoformat(),
        )

        self._rollback_history.append(record)
        self._save_history()

        return rollback_id

    def get_last_stable_configuration(self, run_id: str) -> Optional[str]:
        """
        Letzte stabile Konfiguration in Lineage finden.

        Durchsuche Parent-Historie rückwärts bis stabiler Run gefunden.

        Args:
            run_id: ID des Runs

        Returns:
            run_id der letzten stabilen Konfiguration oder None
        """
        entry = self.registry.get(run_id)
        if entry is None:
            return None

        # Lineage durchsuchen
        lineage = self.registry.get_lineage(run_id)

        # Rückwärts durch Lineage suchen (von aktuell zu alt)
        for ancestor in reversed(lineage):
            if ancestor.status == "completed":
                # Prüfe ob Metriken valide
                if ancestor.val_bpb is not None:
                    # Prüfe ob delta_bpb akzeptabel
                    if ancestor.delta_bpb is None or ancestor.delta_bpb <= 0:
                        return ancestor.run_id

        # Fallback: Parent wenn vorhanden
        if entry.parent_run_id:
            parent = self.registry.get(entry.parent_run_id)
            if parent and parent.status == "completed":
                return parent.run_id

        return None

    def log_rollback(
        self,
        plan: RollbackPlan,
        outcome: Literal["success", "partial", "failed"],
        actual_recovery_time: Optional[str] = None,
        lessons_learned: Optional[List[str]] = None
    ) -> None:
        """
        Rollback dokumentieren für Learning.

        Args:
            plan: Ausgeführter Rollback-Plan
            outcome: "success" | "partial" | "failed"
            actual_recovery_time: Tatsächliche Zeit für Recovery
            lessons_learned: Erkenntnisse für zukünftige Rollbacks
        """
        # Finde entsprechenden Record
        for record in self._rollback_history:
            if record.plan.failed_run_id == plan.failed_run_id:
                record.outcome = outcome
                record.actual_recovery_time = actual_recovery_time
                if lessons_learned:
                    record.lessons_learned = lessons_learned
                break

        self._save_history()

    def get_rollback_statistics(self) -> Dict[str, Any]:
        """
        Rollback-Statistiken.

        Returns:
            {
                "total_rollbacks": N,
                "success_rate": X%,
                "most_common_cause": "...",
                "average_recovery_time": "Xh"
            }
        """
        if not self._rollback_history:
            return {
                "total_rollbacks": 0,
                "success_rate": 0.0,
                "most_common_cause": None,
                "average_recovery_time": None,
            }

        total = len(self._rollback_history)
        successes = sum(1 for r in self._rollback_history if r.outcome == "success")
        success_rate = successes / total if total > 0 else 0.0

        # Häufigste Ursache
        cause_counts: Dict[str, int] = {}
        for record in self._rollback_history:
            for cause in record.plan.changes_to_revert:
                cause_counts[cause] = cause_counts.get(cause, 0) + 1

        most_common_cause = None
        if cause_counts:
            most_common_cause = max(cause_counts.items(), key=lambda x: x[1])[0]

        # Durchschnittliche Recovery-Zeit (vereinfacht)
        recovery_times = [
            r.actual_recovery_time
            for r in self._rollback_history
            if r.actual_recovery_time is not None
        ]

        avg_recovery = None
        if recovery_times:
            # Vereinfachte Berechnung (in Produktion: Zeit parsen und mitteln)
            avg_recovery = f"{len(recovery_times)}h"  # Placeholder

        return {
            "total_rollbacks": total,
            "success_rate": success_rate,
            "success_count": successes,
            "partial_count": sum(1 for r in self._rollback_history if r.outcome == "partial"),
            "failed_count": sum(1 for r in self._rollback_history if r.outcome == "failed"),
            "most_common_cause": most_common_cause,
            "average_recovery_time": avg_recovery,
        }

    def get_rollback_history(
        self,
        limit: int = 10,
        outcome_filter: Optional[str] = None
    ) -> List[RollbackRecord]:
        """
        Rollback-Historie abrufen.

        Args:
            limit: Maximale Anzahl zurückgegebener Einträge
            outcome_filter: Optionaler Filter nach Outcome

        Returns:
            Liste von RollbackRecords
        """
        history = self._rollback_history.copy()

        if outcome_filter:
            history = [r for r in history if r.outcome == outcome_filter]

        # Nach Timestamp sortieren (neueste zuerst)
        history.sort(key=lambda r: r.timestamp, reverse=True)

        return history[:limit]

    def get_rollback_recommendations(
        self,
        failed_run_id: str
    ) -> List[Dict[str, Any]]:
        """
        Empfehlungen für Rollback basierend auf Historie.

        Args:
            failed_run_id: ID des fehlgeschlagenen Runs

        Returns:
            Liste von Empfehlungen
        """
        entry = self.registry.get(failed_run_id)
        if entry is None:
            return []

        recommendations = []

        # Ähnliche historische Rollbacks suchen
        for record in self._rollback_history:
            if record.outcome == "success":
                # Prüfe ob ähnliche Features betroffen
                plan = record.plan
                matching_reverts = [
                    f for f in plan.changes_to_revert
                    if f in self._extract_features_from_run(failed_run_id)
                ]

                if matching_reverts:
                    recommendations.append({
                        "based_on_rollback": record.rollback_id,
                        "success_outcome": True,
                        "recommended_reverts": matching_reverts,
                        "target_config": plan.rollback_target,
                        "success_probability": plan.success_probability,
                    })

        return recommendations[:5]  # Top 5 Empfehlungen
