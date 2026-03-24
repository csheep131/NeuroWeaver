#!/usr/bin/env python3
"""
Run Quarantine Manager für NeuroWeave Phase 4B.

Self-protection gegen bekannte problematische Features und Feature-Kombinationen.

Regeln:
- Feature in 3+ Lineages problematisch → Quarantäne
- Quarantäne läuft nach N erfolgreichen Runs anderer Features aus
- Human Override möglich
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from core.registry import RunRegistry


@dataclass
class QuarantineEntry:
    """
    Ein quarantäner Run/Feature.

    Attributes:
        target: Feature-Name oder Feature-Kombination ("film+xsa")
        reason: Begründung für Quarantäne
        quarantine_type: Typ der Quarantäne
        triggered_by: Run-IDs die Quarantäne ausgelöst haben
        quarantine_start: Beginn der Quarantäne
        quarantine_duration: Anzahl Runs bis Auslauf
        remaining_runs: Verbleibende Runs bis Auslauf
        context_filter: Nur in bestimmten Kontexten blockiert
    """

    target: str
    reason: str
    quarantine_type: Literal["feature", "combination", "context_specific"]
    triggered_by: List[str]
    quarantine_start: str
    quarantine_duration: int
    remaining_runs: int
    context_filter: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung."""
        return {
            "target": self.target,
            "reason": self.reason,
            "quarantine_type": self.quarantine_type,
            "triggered_by": self.triggered_by,
            "quarantine_start": self.quarantine_start,
            "quarantine_duration": self.quarantine_duration,
            "remaining_runs": self.remaining_runs,
            "context_filter": self.context_filter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuarantineEntry":
        """Erstelle QuarantineEntry aus Dictionary."""
        return cls(
            target=data.get("target", ""),
            reason=data.get("reason", ""),
            quarantine_type=data.get("quarantine_type", "feature"),
            triggered_by=data.get("triggered_by", []),
            quarantine_start=data.get("quarantine_start", ""),
            quarantine_duration=data.get("quarantine_duration", 5),
            remaining_runs=data.get("remaining_runs", 5),
            context_filter=data.get("context_filter"),
        )


class RunQuarantineManager:
    """
    Verwaltet Quarantäne für Features/Kombinationen.

    Überwacht Feature-Performance und setzt problematische Features
    automatisch in Quarantäne zum Schutz des Systems.

    Example:
        manager = RunQuarantineManager(quarantine_threshold=3, quarantine_duration=5)
        
        # Prüfen vor Run-Start
        is_blocked, reason = manager.check_quarantine(["film", "xsa"], context="default")
        if is_blocked:
            print(f"Feature-Kombination blockiert: {reason}")
        
        # Nach erfolgreichem Run
        manager.tick("run001")
    """

    def __init__(
        self,
        quarantine_threshold: int = 3,
        quarantine_duration: int = 5,
        quarantine_log_path: str = "results/quarantine_log.json"
    ):
        """
        Initialisiere RunQuarantineManager.

        Args:
            quarantine_threshold: Anzahl Fehler für Quarantäne-Auslösung
            quarantine_duration: Anzahl erfolgreicher Runs bis Quarantäne ausläuft
            quarantine_log_path: Pfad zum Quarantäne-Log
        """
        self.quarantine_threshold = quarantine_threshold
        self.quarantine_duration = quarantine_duration
        self.quarantine_log_path = Path(quarantine_log_path)

        self._quarantine_entries: Dict[str, QuarantineEntry] = {}
        self._feature_failure_counts: Dict[str, List[str]] = {}  # feature -> [run_ids]
        self._successful_run_count: int = 0

        self._load_state()

    def _load_state(self) -> None:
        """Lade Quarantäne-Status von Disk."""
        if self.quarantine_log_path.exists():
            with open(self.quarantine_log_path, "r") as f:
                data = json.load(f)
                for target, entry_data in data.get("entries", {}).items():
                    self._quarantine_entries[target] = QuarantineEntry.from_dict(entry_data)
                self._feature_failure_counts = data.get("failure_counts", {})
                self._successful_run_count = data.get("successful_run_count", 0)

    def _save_state(self) -> None:
        """Speichere Quarantäne-Status auf Disk."""
        self.quarantine_log_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": {k: v.to_dict() for k, v in self._quarantine_entries.items()},
            "failure_counts": self._feature_failure_counts,
            "successful_run_count": self._successful_run_count,
        }
        with open(self.quarantine_log_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_feature_key(self, features: List[str]) -> str:
        """
        Generiere konsistenten Key für Feature oder Feature-Kombination.

        Args:
            features: Liste von Features

        Returns:
            Sortierter Feature-Key (z.B. "film+xsa")
        """
        return "+".join(sorted(features))

    def _extract_features_from_run(
        self,
        run_id: str,
        registry: RunRegistry
    ) -> List[str]:
        """
        Extrahiere Features aus einem Run.

        Args:
            run_id: Run-ID
            registry: RunRegistry

        Returns:
            Liste von Feature-Namen
        """
        entry = registry.get(run_id)
        if entry is None:
            return []

        # Features aus Config extrahieren (wenn verfügbar)
        config_path = Path("configs") / "runs" / f"{run_id}.yaml"
        if config_path.exists():
            # In Produktion: Config laden und Features parsen
            pass

        # Fallback: Features aus Tags oder Notes extrahieren
        features = []
        if entry.tags:
            for tag in entry.tags:
                if tag not in ("rollback", "quarantine"):
                    features.append(tag)

        return features

    def check_quarantine(
        self,
        proposed_features: List[str],
        context: str = "default"
    ) -> Tuple[bool, Optional[str]]:
        """
        Prüfen ob Feature-Kombination quarantänt ist.

        Args:
            proposed_features: Liste der geplanten Features
            context: Optionaler Kontext (z.B. "low_budget", "high_performance")

        Returns:
            (is_quarantined, reason)
        """
        if not proposed_features:
            return False, None

        # Einzelne Features prüfen
        for feature in proposed_features:
            if feature in self._quarantine_entries:
                entry = self._quarantine_entries[feature]

                # Kontext-Filter prüfen
                if entry.context_filter and entry.context_filter != context:
                    continue  # Quarantäne gilt nicht für diesen Kontext

                return True, f"Feature '{feature}' ist in Quarantäne: {entry.reason}"

        # Feature-Kombinationen prüfen
        combo_key = self._generate_feature_key(proposed_features)
        if combo_key in self._quarantine_entries:
            entry = self._quarantine_entries[combo_key]

            # Kontext-Filter prüfen
            if entry.context_filter and entry.context_filter != context:
                pass  # Quarantäne gilt nicht für diesen Kontext
            else:
                return True, f"Feature-Kombination '{combo_key}' ist in Quarantäne: {entry.reason}"

        # Kontext-spezifische Quarantänen prüfen
        context_key = f"{combo_key}@{context}"
        if context_key in self._quarantine_entries:
            entry = self._quarantine_entries[context_key]
            return True, f"Feature-Kombination im Kontext '{context}' blockiert: {entry.reason}"

        return False, None

    def add_quarantine(
        self,
        feature: str,
        triggered_by: List[str],
        context: Optional[str] = None,
        reason: Optional[str] = None,
        duration: Optional[int] = None
    ) -> QuarantineEntry:
        """
        Quarantäne hinzufügen.

        Args:
            feature: Feature-Name oder Kombination ("film+xsa")
            triggered_by: Run-IDs die Quarantäne ausgelöst haben
            context: Optionaler Kontext (nur in diesem Kontext blockiert)
            reason: Begründung für Quarantäne
            duration: Überschreibt default quarantine_duration

        Returns:
            Erstellte QuarantineEntry
        """
        # Bestehende Quarantäne aktualisieren oder neue erstellen
        target_key = feature
        if context:
            target_key = f"{feature}@{context}"

        quarantine_type: Literal["feature", "combination", "context_specific"]
        if "+" in feature:
            quarantine_type = "combination"
        elif context:
            quarantine_type = "context_specific"
        else:
            quarantine_type = "feature"

        default_reason = f"Automatische Quarantäne nach {len(triggered_by)} Fehlern"
        entry = QuarantineEntry(
            target=feature,
            reason=reason or default_reason,
            quarantine_type=quarantine_type,
            triggered_by=triggered_by.copy(),
            quarantine_start=datetime.now().isoformat(),
            quarantine_duration=duration or self.quarantine_duration,
            remaining_runs=duration or self.quarantine_duration,
            context_filter=context,
        )

        self._quarantine_entries[target_key] = entry
        self._save_state()

        return entry

    def release_quarantine(self, feature: str, context: Optional[str] = None) -> bool:
        """
        Quarantäne vorzeitig aufheben.

        Args:
            feature: Feature-Name oder Kombination
            context: Optionaler Kontext

        Returns:
            True wenn Quarantäne entfernt wurde, False wenn nicht existiert
        """
        target_key = feature
        if context:
            target_key = f"{feature}@{context}"

        if target_key in self._quarantine_entries:
            del self._quarantine_entries[target_key]
            self._save_state()
            return True

        return False

    def tick(self, successful_run_id: str, registry: RunRegistry) -> None:
        """
        Nach erfolgreichem Run: Quarantäne-Zähler dekrementieren.

        Wird bei jedem erfolgreichen Run aufgerufen.

        Args:
            successful_run_id: ID des erfolgreichen Runs
            registry: RunRegistry für Datenzugriff
        """
        self._successful_run_count += 1

        # Alle Quarantäne-Zähler dekrementieren
        expired = []
        for target_key, entry in self._quarantine_entries.items():
            entry.remaining_runs -= 1
            if entry.remaining_runs <= 0:
                expired.append(target_key)

        # Abgelaufene Quarantänen entfernen
        for target_key in expired:
            del self._quarantine_entries[target_key]

        # Failure-Counts bereinigen (alte Einträge entfernen)
        self._cleanup_failure_counts(registry)

        self._save_state()

    def _cleanup_failure_counts(self, registry: RunRegistry) -> None:
        """
        Bereinige Failure-Counts von alten Einträgen.

        Entfernt Runs die älter als N Runs sind.
        """
        # Behalte nur letzte N Runs pro Feature
        max_history = self.quarantine_threshold * 2

        for feature in list(self._feature_failure_counts.keys()):
            run_ids = self._feature_failure_counts[feature]
            if len(run_ids) > max_history:
                self._feature_failure_counts[feature] = run_ids[-max_history:]

    def get_quarantine_list(self) -> List[QuarantineEntry]:
        """
        Aktive Quarantänen auflisten.

        Returns:
            Liste aller aktiven QuarantineEntries
        """
        return list(self._quarantine_entries.values())

    def human_override(
        self,
        feature: str,
        justification: str,
        context: Optional[str] = None
    ) -> bool:
        """
        Manuelle Freigabe durch Human.

        Args:
            feature: Freigegebenes Feature
            justification: Begründung für Override
            context: Optionaler Kontext

        Returns:
            True wenn Override erfolgreich, False wenn Quarantäne nicht existiert
        """
        target_key = feature
        if context:
            target_key = f"{feature}@{context}"

        if target_key in self._quarantine_entries:
            entry = self._quarantine_entries[target_key]
            entry.reason += f" | Human Override: {justification}"
            # Quarantäne nicht entfernen, aber dokumentieren
            self._save_state()
            return True

        return False

    def report_failure(
        self,
        run_id: str,
        features: List[str],
        failure_reason: str,
        context: str = "default"
    ) -> Optional[QuarantineEntry]:
        """
        Melde einen fehlgeschlagenen Run für Quarantäne-Prüfung.

        Args:
            run_id: ID des fehlgeschlagenen Runs
            features: Features des Runs
            failure_reason: Grund des Fehlers
            context: Kontext des Runs

        Returns:
            QuarantineEntry wenn Quarantäne ausgelöst, sonst None
        """
        # Failure-Counts aktualisieren
        for feature in features:
            if feature not in self._feature_failure_counts:
                self._feature_failure_counts[feature] = []
            self._feature_failure_counts[feature].append(run_id)

        # Prüfen ob Quarantäne-Schwellwert erreicht
        new_quarantines = []

        # Einzelne Features prüfen
        for feature in features:
            failure_count = len(self._feature_failure_counts.get(feature, []))
            if failure_count >= self.quarantine_threshold:
                # Quarantäne auslösen
                if feature not in self._quarantine_entries:
                    entry = self.add_quarantine(
                        feature=feature,
                        triggered_by=self._feature_failure_counts[feature],
                        reason=f"Automatische Quarantäne: {failure_count} Fehler in Folge. {failure_reason}",
                        context=context if context != "default" else None,
                    )
                    new_quarantines.append(entry)

        # Feature-Kombinationen prüfen (Paare)
        for i, f1 in enumerate(features):
            for f2 in features[i + 1:]:
                combo_key = self._generate_feature_key([f1, f2])
                combo_failures = self._count_combo_failures(f1, f2)

                if combo_failures >= self.quarantine_threshold:
                    if combo_key not in self._quarantine_entries:
                        entry = self.add_quarantine(
                            feature=combo_key,
                            triggered_by=[run_id],  # Vereinfacht
                            reason=f"Feature-Kombination problematisch: {combo_failures} Fehler",
                            quarantine_type="combination",
                        )
                        new_quarantines.append(entry)

        return new_quarantines[0] if new_quarantines else None

    def _count_combo_failures(self, f1: str, f2: str) -> int:
        """
        Zähle Fehler für Feature-Kombination.

        Args:
            f1: Feature 1
            f2: Feature 2

        Returns:
            Anzahl der Fehler bei Kombination
        """
        # Vereinfachte Zählung: beide Features müssen in failure_counts sein
        f1_failures = set(self._feature_failure_counts.get(f1, []))
        f2_failures = set(self._feature_failure_counts.get(f2, []))

        # Schnittmenge = Runs wo beide Features aktiv waren
        return len(f1_failures & f2_failures)

    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Statistiken über Features und Quarantänen.

        Returns:
            Dictionary mit Statistiken
        """
        # Feature-Ranking nach Fehleranzahl
        feature_ranking = []
        for feature, run_ids in self._feature_failure_counts.items():
            feature_ranking.append({
                "feature": feature,
                "failure_count": len(run_ids),
                "is_quarantined": feature in self._quarantine_entries,
            })

        feature_ranking.sort(key=lambda x: x["failure_count"], reverse=True)

        # Quarantäne-Statistiken
        quarantine_counts = {
            "total": len(self._quarantine_entries),
            "feature": sum(1 for e in self._quarantine_entries.values() if e.quarantine_type == "feature"),
            "combination": sum(1 for e in self._quarantine_entries.values() if e.quarantine_type == "combination"),
            "context_specific": sum(1 for e in self._quarantine_entries.values() if e.quarantine_type == "context_specific"),
        }

        return {
            "total_features_tracked": len(self._feature_failure_counts),
            "active_quarantines": quarantine_counts,
            "feature_ranking": feature_ranking[:10],  # Top 10
            "total_successful_runs": self._successful_run_count,
        }

    def export_quarantine_report(self, output_path: str) -> str:
        """
        Exportiere Quarantäne-Report.

        Args:
            output_path: Pfad zur Ausgabedatei

        Returns:
            Pfad zur exportierten Datei
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_feature_statistics(),
            "active_quarantines": [e.to_dict() for e in self.get_quarantine_list()],
            "quarantine_threshold": self.quarantine_threshold,
            "quarantine_duration": self.quarantine_duration,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        return str(output_file)

    def is_feature_tracked(self, feature: str) -> bool:
        """
        Prüfen ob Feature getrackt wird.

        Args:
            feature: Feature-Name

        Returns:
            True wenn Feature getrackt wird
        """
        return feature in self._feature_failure_counts

    def get_feature_failure_history(self, feature: str) -> List[str]:
        """
        Historie der Fehler für ein Feature.

        Args:
            feature: Feature-Name

        Returns:
            Liste von Run-IDs
        """
        return self._feature_failure_counts.get(feature, []).copy()

    def reset_all(self) -> None:
        """
        Setze alle Quarantänen zurück.

        Vorsicht: Entfernt alle Quarantänen und Failure-Counts.
        """
        self._quarantine_entries.clear()
        self._feature_failure_counts.clear()
        self._successful_run_count = 0
        self._save_state()
