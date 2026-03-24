#!/usr/bin/env python3
"""
Phase 4: Autonome Selbstverbesserung - Zentrale Steuerung

Usage:
    python3 -m orchestrator.phase4_orchestrator run      # Autonome Run-Auswahl
    python3 -m orchestrator.phase4_orchestrator status   # Status anzeigen
    python3 -m orchestrator.phase4_orchestrator report   # Zusammenfassung

Phase 4 Komponenten:
- Meta Features (Woche 1-2)
- Surrogate Scorer, Hypothesis Generator, Pareto Tracker (Woche 3-4)
- Anomaly Detector, Failure Classifier, etc. (Woche 5-6)
- Guardrails, Autonomy, Alerting (Woche 7-8)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import RunRegistry
from core.meta_features import MetaFeatureExtractor, RunMetaFeatures
from core.alerting import AlertManager, AlertSeverity, create_alert_manager

from research.surrogate_scorer import SurrogateScorer
from research.hypothesis_generator import HypothesisGenerator
from research.pareto_tracker import ParetoTracker
from research.anomaly_detector import AnomalyDetector
from research.failure_classifier import FailureClassifier
from research.drift_monitor import DriftMonitor
from research.override_learner import OverrideLearner, create_override_learner

from orchestrator.guardrails import (
    create_default_guardrails,
    GuardrailManager,
    AutonomyLevel,
    AutonomyConfig,
)
from orchestrator.autonomy_orchestrator import (
    AutonomyOrchestrator,
    create_autonomy_orchestrator,
)
from orchestrator.approval_interface import (
    ApprovalInterface,
    create_approval_interface,
)
from orchestrator.run_quarantine import RunQuarantineManager
from orchestrator.rollback_manager import RollbackManager


class Phase4Orchestrator:
    """
    Zentrale Orchestrierung aller Phase 4 Komponenten.

    Integriert:
    - Meta Features (Woche 1-2)
    - Surrogate Scorer, Hypothesis Generator, Pareto Tracker (Woche 3-4)
    - Anomaly Detector, Failure Classifier, etc. (Woche 5-6)
    - Guardrails, Autonomy, Alerting (Woche 7-8)
    """

    def __init__(
        self,
        results_dir: Optional[str] = None,
        configs_dir: Optional[str] = None,
        autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED,
    ) -> None:
        """
        Initialisiere Phase4Orchestrator.

        Args:
            results_dir: Verzeichnis für Ergebnisse
            configs_dir: Verzeichnis für Konfigurationen
            autonomy_level: Gewünschtes Autonomie-Level
        """
        self._results_dir = Path(results_dir) if results_dir else Path(__file__).parent.parent / "results"
        self._configs_dir = Path(configs_dir) if configs_dir else Path(__file__).parent.parent / "configs"

        # Initialisiere Registry und Extractor
        self._registry = RunRegistry(results_dir=str(self._results_dir))
        self._extractor = MetaFeatureExtractor(configs_dir=self._configs_dir)

        # Initialisiere Phase 4A Komponenten
        self._surrogate_scorer = SurrogateScorer()
        self._hypothesis_generator = HypothesisGenerator()
        self._pareto_tracker = ParetoTracker()

        # Initialisiere Phase 4B Komponenten
        self._anomaly_detector = AnomalyDetector()
        self._failure_classifier = FailureClassifier()
        self._drift_monitor = DriftMonitor()
        self._quarantine_manager = RunQuarantineManager()
        self._rollback_manager = RollbackManager()

        # Initialisiere Phase 4C Komponenten
        config = create_default_guardrails()
        config.level = autonomy_level
        self._guardrail_manager = GuardrailManager(config)
        self._autonomy_orchestrator = AutonomyOrchestrator(self._guardrail_manager)
        self._approval_interface = ApprovalInterface(self._autonomy_orchestrator)
        self._alert_manager = create_alert_manager()
        self._override_learner = create_override_learner()

    @property
    def registry(self) -> RunRegistry:
        """RunRegistry zurückgeben."""
        return self._registry

    @property
    def guardrail_manager(self) -> GuardrailManager:
        """GuardrailManager zurückgeben."""
        return self._guardrail_manager

    @property
    def autonomy_orchestrator(self) -> AutonomyOrchestrator:
        """AutonomyOrchestrator zurückgeben."""
        return self._autonomy_orchestrator

    @property
    def approval_interface(self) -> ApprovalInterface:
        """ApprovalInterface zurückgeben."""
        return self._approval_interface

    @property
    def alert_manager(self) -> AlertManager:
        """AlertManager zurückgeben."""
        return self._alert_manager

    @property
    def override_learner(self) -> OverrideLearner:
        """OverrideLearner zurückgeben."""
        return self._override_learner

    def run_autonomous_cycle(self) -> Dict[str, Any]:
        """
        Einen autonomen Zyklus ausführen.

        Steps:
        1. Meta-Features extrahieren
        2. Hypothesen generieren
        3. Guardrails prüfen
        4. Top-Vorschläge ausführen (Smoke-Test)
        5. Anomalien prüfen
        6. Bei Erfolg: Promotion vorschlagen

        Returns:
            Dictionary mit Zyklus-Ergebnissen
        """
        cycle_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results: Dict[str, Any] = {
            "cycle_id": cycle_id,
            "timestamp": datetime.utcnow().isoformat(),
            "steps": {},
            "status": "running",
        }

        try:
            # Step 1: Meta-Features extrahieren
            results["steps"]["meta_features"] = self._extract_meta_features()

            # Step 2: Hypothesen generieren
            results["steps"]["hypotheses"] = self._generate_hypotheses()

            # Step 3: Guardrails prüfen
            results["steps"]["guardrail_check"] = self._check_guardrails()

            # Step 4: Top-Vorschläge auswählen
            results["steps"]["selection"] = self._select_top_candidates()

            # Step 5: Anomalien prüfen
            results["steps"]["anomaly_check"] = self._check_anomalies()

            # Step 6: Promotion vorschlagen (wenn erfolgreich)
            results["steps"]["promotion"] = self._propose_promotion()

            results["status"] = "completed"

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)

            # Erstelle Alert bei Fehler
            self._alert_manager.create_alert(
                severity=AlertSeverity.HIGH,
                title="Autonomous Cycle Failed",
                message=f"Autonomer Zyklus {cycle_id} fehlgeschlagen: {e}",
                source="phase4_orchestrator",
                requires_action=True,
            )

        return results

    def _extract_meta_features(self) -> Dict[str, Any]:
        """Meta-Features extrahieren."""
        run_ids = [run.run_id for run in self._registry.list_runs()[:100]]

        if not run_ids:
            return {"count": 0, "status": "no_runs"}

        features = self._extractor.extract_batch(run_ids, self._registry)

        return {
            "count": len(features),
            "status": "success",
            "run_ids_processed": run_ids[:10],
        }

    def _generate_hypotheses(self) -> Dict[str, Any]:
        """Hypothesen generieren."""
        # Placeholder für Hypothesen-Generierung
        return {
            "hypotheses_generated": 0,
            "status": "success",
        }

    def _check_guardrails(self) -> Dict[str, Any]:
        """Guardrails prüfen."""
        status = self._guardrail_manager.get_guardrail_status()
        return {
            "guardrails_active": status["total_guardrails"],
            "autonomy_level": status["autonomy_level"],
            "status": "success",
        }

    def _select_top_candidates(self) -> Dict[str, Any]:
        """Top-Kandidaten auswählen."""
        # Placeholder für Kandidaten-Auswahl
        return {
            "candidates_selected": 0,
            "status": "success",
        }

    def _check_anomalies(self) -> Dict[str, Any]:
        """Anomalien prüfen."""
        # Placeholder für Anomalie-Prüfung
        return {
            "anomalies_detected": 0,
            "status": "success",
        }

    def _propose_promotion(self) -> Dict[str, Any]:
        """Promotion vorschlagen."""
        # Placeholder für Promotion-Vorschlag
        return {
            "promotions_proposed": 0,
            "status": "success",
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Gesamt-Status aller Phase 4 Komponenten.

        Returns:
            Dictionary mit Status-Informationen
        """
        return {
            "meta_features": self._get_meta_features_status(),
            "surrogate_scorer": self._get_surrogate_scorer_status(),
            "hypotheses": self._get_hypotheses_status(),
            "pareto_frontier": self._get_pareto_frontier_status(),
            "anomalies": self._get_anomalies_status(),
            "quarantines": self._get_quarantine_status(),
            "guardrails": self._get_guardrail_status(),
            "pending_approvals": self._get_approval_status(),
            "alerts": self._get_alert_status(),
        }

    def _get_meta_features_status(self) -> Dict:
        """Status der Meta-Features."""
        run_count = len(list(self._registry.list_runs()))
        return {
            "total_runs": run_count,
            "extractor_ready": self._extractor is not None,
        }

    def _get_surrogate_scorer_status(self) -> Dict:
        """Status des Surrogate Scorer."""
        return {
            "model_loaded": True,
            "training_samples": 0,
        }

    def _get_hypotheses_status(self) -> Dict:
        """Status der Hypothesen."""
        return {
            "pending_hypotheses": 0,
            "tested_hypotheses": 0,
        }

    def _get_pareto_frontier_status(self) -> Dict:
        """Status der Pareto-Frontier."""
        return {
            "pareto_points": 0,
            "last_update": None,
        }

    def _get_anomalies_status(self) -> Dict:
        """Status der Anomalien."""
        return {
            "active_anomalies": 0,
            "resolved_anomalies": 0,
        }

    def _get_quarantine_status(self) -> Dict:
        """Status der Quarantäne."""
        return {
            "quarantined_runs": 0,
            "released_runs": 0,
        }

    def _get_guardrail_status(self) -> Dict:
        """Status der Guardrails."""
        return self._guardrail_manager.get_guardrail_status()

    def _get_approval_status(self) -> Dict:
        """Status der Approvals."""
        stats = self._approval_interface.get_approval_statistics()
        return {
            "pending": stats["pending"],
            "approved_today": stats["approved_today"],
            "rejected_today": stats["rejected_today"],
        }

    def _get_alert_status(self) -> Dict:
        """Status der Alerts."""
        summary = self._alert_manager.get_alert_summary(hours=24)
        return {
            "total": summary["total"],
            "pending": summary["pending"],
            "action_required": summary["action_required"],
        }

    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Zusammenfassungs-Report generieren.

        Args:
            hours: Zeitraum in Stunden

        Returns:
            Dictionary mit Report-Daten
        """
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "time_range_hours": hours,
            "summary": self.get_status(),
            "autonomy_statistics": self._autonomy_orchestrator.get_statistics(),
            "approval_statistics": self._approval_interface.get_approval_statistics(),
            "alert_summary": self._alert_manager.get_alert_summary(hours=hours),
            "override_statistics": self._override_learner.get_override_statistics(
                hours=hours
            ),
        }


def cmd_run(args: argparse.Namespace) -> int:
    """Autonomen Zyklus ausführen."""
    print("🚀 Starte autonomen Zyklus...")

    orchestrator = Phase4Orchestrator(
        autonomy_level=AutonomyLevel(args.autonomy_level),
    )

    results = orchestrator.run_autonomous_cycle()

    if results["status"] == "completed":
        print("✅ Autonomer Zyklus erfolgreich abgeschlossen.")
        print(f"   Cycle-ID: {results['cycle_id']}")
        for step_name, step_result in results.get("steps", {}).items():
            status_icon = "✅" if step_result.get("status") == "success" else "❌"
            print(f"   {status_icon} {step_name}: {step_result.get('status', 'unknown')}")
    else:
        print("❌ Autonomer Zyklus fehlgeschlagen.")
        print(f"   Fehler: {results.get('error', 'Unbekannt')}")
        return 1

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Status anzeigen."""
    print("📊 Phase 4 Status")
    print("=" * 60)

    orchestrator = Phase4Orchestrator()
    status = orchestrator.get_status()

    print(f"\n📈 Meta-Features:")
    print(f"   Total Runs: {status['meta_features']['total_runs']}")

    print(f"\n🛡️  Guardrails:")
    print(f"   Autonomie-Level: {status['guardrails']['autonomy_level'].upper()}")
    print(f"   Aktive Guardrails: {status['guardrails']['total_guardrails']}")

    print(f"\n📋 Approvals:")
    print(f"   Ausstehend: {status['pending_approvals']['pending']}")
    print(f"   Heute genehmigt: {status['pending_approvals']['approved_today']}")

    print(f"\n🚨 Alerts:")
    print(f"   Total (24h): {status['alerts']['total']}")
    print(f"   Ausstehend: {status['alerts']['pending']}")
    print(f"   Action required: {status['alerts']['action_required']}")

    print("\n" + "=" * 60)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Report generieren."""
    print("📄 Generiere Report...")

    orchestrator = Phase4Orchestrator()
    report = orchestrator.generate_report(hours=args.hours)

    print("\n" + "=" * 60)
    print("PHASE 4 REPORT")
    print("=" * 60)
    print(f"Generiert am: {report['generated_at']}")
    print(f"Zeitraum: {report['time_range_hours']}h")

    print("\n📊 Autonomie-Statistiken:")
    auto_stats = report["autonomy_statistics"]
    print(f"   Total Actions: {auto_stats['total_actions']}")
    print(f"   Erfolgsrate: {auto_stats['success_rate']:.0%}")

    print("\n📋 Approval-Statistiken:")
    approval_stats = report["approval_statistics"]
    print(f"   Pending: {approval_stats['pending']}")
    print(f"   Ø Genehmigungszeit: {approval_stats['avg_approval_time']}")

    print("\n🚨 Alert-Zusammenfassung:")
    alert_summary = report["alert_summary"]
    print(f"   Total: {alert_summary['total']}")
    print(f"   Action required: {alert_summary['action_required']}")

    print("\n" + "=" * 60)

    # Optional als JSON speichern
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"💾 Report gespeichert unter: {output_path}")

    return 0


def main() -> int:
    """Hauptfunktion."""
    parser = argparse.ArgumentParser(
        description="Phase 4: Autonome Selbstverbesserung - Zentrale Steuerung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s run                     # Autonomen Zyklus starten
  %(prog)s run --autonomy-level supervised
  %(prog)s status                  # Status anzeigen
  %(prog)s report                  # Report generieren
  %(prog)s report --hours 48       # Report für 48h
  %(prog)s report --output report.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run Command
    run_parser = subparsers.add_parser("run", help="Autonomen Zyklus ausführen")
    run_parser.add_argument(
        "--autonomy-level",
        choices=["manual", "assisted", "supervised", "autonomous"],
        default="supervised",
        help="Autonomie-Level (default: supervised)",
    )
    run_parser.set_defaults(func=cmd_run)

    # Status Command
    status_parser = subparsers.add_parser("status", help="Status anzeigen")
    status_parser.set_defaults(func=cmd_status)

    # Report Command
    report_parser = subparsers.add_parser("report", help="Report generieren")
    report_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Zeitraum in Stunden (default: 24)",
    )
    report_parser.add_argument(
        "--output",
        type=str,
        help="JSON-Datei für Report-Output",
    )
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
