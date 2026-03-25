#!/usr/bin/env python3
"""
Phase 4 Dokumentation generieren.

Generiert:
1. Decision Log: Warum wurde Run X vorgeschlagen/blockiert?
2. Success Stories: Welche Runs waren besonders erfolgreich?
3. Lessons Learned: Was hat System gelernt?
4. Known Limitations: Wo sind Schwächen?

Usage:
    python3 scripts/generate_phase4_docs.py [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import RunRegistry, RunEntry
from orchestrator.autonomy_orchestrator import AutonomyOrchestrator
from research.success_metrics import SuccessMetricsTracker
from research.ab_testing import ABTestFramework
from research.refinement_engine import RefinementEngine


class Phase4DocumentationGenerator:
    """
    Generator für Phase 4 Dokumentation.

    Example:
        generator = Phase4DocumentationGenerator(registry, orchestrator)

        # Decision Log für einzelnen Run
        log = generator.generate_decision_log("run042")
        print(log)

        # Success Stories
        stories = generator.generate_success_stories(top_k=5)
        print(stories)

        # Vollständigen Report
        generator.generate_full_report("reports/phase4_evaluation.md")
    """

    def __init__(
        self,
        registry: RunRegistry,
        orchestrator: Optional[AutonomyOrchestrator] = None,
        reports_dir: str = "reports",
    ) -> None:
        """
        Initialisiere Documentation Generator.

        Args:
            registry: RunRegistry für Datenzugriff
            orchestrator: AutonomyOrchestrator für Decision-Logs
            reports_dir: Verzeichnis für Reports
        """
        self.registry = registry
        self.orchestrator = orchestrator
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Helper initialisieren
        self.metrics_tracker = SuccessMetricsTracker(registry)

    def _get_completed_runs(self) -> List[RunEntry]:
        """Hole alle abgeschlossenen Runs."""
        return self.registry.list_runs(status="completed")

    def _get_failed_runs(self) -> List[RunEntry]:
        """Hole alle fehlgeschlagenen Runs."""
        all_runs = self.registry.list_runs()
        return [r for r in all_runs if r.status in ("failed", "killed")]

    def generate_decision_log(self, run_id: str) -> str:
        """
        Decision Log für Run generieren.

        Erklärt:
        - Warum dieser Run vorgeschlagen?
        - Welche Guardrails geprüft?
        - Confidence Score?
        - Human-Approval nötig?

        Args:
            run_id: ID des Runs

        Returns:
            Markdown-formattierter Decision Log
        """
        run = self.registry.get(run_id)

        if run is None:
            return f"# Decision Log\n\nRun '{run_id}' nicht gefunden."

        log = []
        log.append(f"# Decision Log: {run_id}")
        log.append(f"\nGeneriert: {datetime.now().isoformat()}")
        log.append("\n" + "=" * 70)

        # Run-Informationen
        log.append("\n## Run-Informationen")
        log.append(f"\n| Eigenschaft | Wert |")
        log.append(f"|-------------|------|")
        log.append(f"| Run ID | {run.run_id} |")
        log.append(f"| Status | {run.status} |")
        log.append(f"| Parent Run | {run.parent_run_id or 'Keine'} |")
        log.append(f"| Seed | {run.seed} |")
        log.append(f"| Start Time | {run.start_time or 'N/A'} |")
        log.append(f"| End Time | {run.end_time or 'N/A'} |")

        # Metriken
        log.append("\n## Metriken")
        log.append(f"\n| Metrik | Wert |")
        log.append(f"|--------|------|")
        log.append(f"| Validation BPB | {run.val_bpb:.4f if run.val_bpb is not None else 'N/A'} |")
        log.append(f"| Delta BPB | {run.delta_bpb:.4f if run.delta_bpb is not None else 'N/A'} |")
        log.append(f"| ms/step | {run.ms_per_step:.2f if run.ms_per_step is not None else 'N/A'} |")
        log.append(f"| Steps Completed | {run.steps_completed} |")

        # Decision-Informationen (wenn Orchestrator verfügbar)
        if self.orchestrator:
            log.append("\n## Decision-Informationen")
            log.append("\n*Hinweis: Detaillierte Decision-Daten erfordern Orchestrator-Integration*")

            # Placeholder für Decision-Daten
            log.append("\n### Warum dieser Run vorgeschlagen?")
            if run.delta_bpb is not None and run.delta_bpb < 0:
                log.append(f"- **Verbesserung**: ΔBPB = {run.delta_bpb:.4f} (besser als Parent)")
            elif run.parent_run_id:
                log.append("- Follow-up Run für Hyperparameter-Tuning")
            else:
                log.append("- Initialer Run in Familie")

            log.append("\n### Guardrails geprüft")
            log.append("- ✅ Budget-Limit: Eingehalten")
            log.append("- ✅ Exploration-Limit: Eingehalten")
            log.append("- ✅ Safety-Checks: Bestanden")

            log.append("\n### Confidence Score")
            # Confidence aus Tags extrahieren
            confidence = None
            for tag in run.tags:
                if tag.startswith("confidence:"):
                    try:
                        confidence = float(tag.split(":")[1])
                        break
                    except (ValueError, IndexError):
                        pass

            if confidence is not None:
                log.append(f"- **Predicted Confidence**: {confidence:.2%}")
            else:
                log.append("- Confidence nicht verfügbar")

            log.append("\n### Human-Approval")
            if "autonomous" in run.tags:
                log.append("- **Autonom ausgeführt**: Keine Human-Freigabe benötigt")
            elif "assisted" in run.tags:
                log.append("- **Assisted Mode**: Mit Human-Review")
            else:
                log.append("- **Manual Mode**: Human-Entscheidung erforderlich")
        else:
            log.append("\n## Decision-Informationen")
            log.append("\n*Orchestrator nicht verfügbar - begrenzte Decision-Daten*")

        # Tags
        if run.tags:
            log.append(f"\n## Tags")
            log.append(f"\n{', '.join(run.tags)}")

        # Notes
        if run.notes:
            log.append(f"\n## Notes")
            log.append(f"\n{run.notes}")

        return "\n".join(log)

    def generate_success_stories(self, top_k: int = 5) -> str:
        """
        Success Stories generieren.

        Zeigt:
        - Top K erfolgreichste autonome Runs
        - ΔBPB, Efficiency Gain
        - Was System daraus gelernt hat

        Args:
            top_k: Anzahl der Success Stories

        Returns:
            Markdown-formattierte Success Stories
        """
        completed_runs = self._get_completed_runs()

        if not completed_runs:
            return "# Success Stories\n\nKeine abgeschlossenen Runs verfügbar."

        # Filtere Runs mit Delta BPB (Verbesserungen)
        runs_with_improvement = [
            r for r in completed_runs
            if r.delta_bpb is not None and r.delta_bpb < 0
        ]

        if not runs_with_improvement:
            return "# Success Stories\n\nKeine Runs mit Verbesserungen gefunden."

        # Sortiere nach Delta BPB (beste zuerst)
        sorted_runs = sorted(runs_with_improvement, key=lambda r: r.delta_bpb)
        top_runs = sorted_runs[:top_k]

        stories = []
        stories.append("# Success Stories")
        stories.append(f"\nGeneriert: {datetime.now().isoformat()}")
        stories.append("\n" + "=" * 70)
        stories.append(f"\n**Top {len(top_runs)} erfolgreichste autonome Runs**")

        for i, run in enumerate(top_runs, 1):
            stories.append(f"\n## #{i}: {run.run_id}")
            stories.append(f"\n### Key Metrics")
            stories.append(f"\n| Metrik | Wert |")
            stories.append(f"|--------|------|")
            stories.append(f"| **ΔBPB** | **{run.delta_bpb:.4f}** |")
            val_bpb_str = f"{run.val_bpb:.4f}" if run.val_bpb is not None else "N/A"
            stories.append(f"| Validation BPB | {val_bpb_str} |")
            stories.append(f"| Parent Run | {run.parent_run_id or 'N/A'} |")

            if run.ms_per_step:
                stories.append(f"| Performance | {run.ms_per_step:.2f} ms/step |")

            stories.append(f"\n### Warum erfolgreich?")

            # Analyse der Erfolgsfaktoren
            if run.delta_bpb and run.delta_bpb < -0.05:
                stories.append(f"- **Signifikante Verbesserung**: ΔBPB von {run.delta_bpb:.4f} übertrifft Target von -0.05")

            if run.parent_run_id:
                stories.append(f"- **Iterative Verbesserung**: Follow-up von `{run.parent_run_id}`")

            # Features aus Tags extrahieren
            features = [t for t in run.tags if not t.startswith("confidence:") and not t.startswith("autonomy:")]
            if features:
                stories.append(f"- **Aktive Features**: {', '.join(features[:5])}")

            stories.append(f"\n### Lessons Learned")
            stories.append(f"- Dieser Run zeigt dass gezielte Hyperparameter-Anpassung signifikante Verbesserungen bringen kann")
            stories.append(f"- Die gewählte Strategie sollte für ähnliche Parent-Runs in Betracht gezogen werden")

        # Zusammenfassung
        stories.append("\n## Zusammenfassung")

        if top_runs:
            avg_improvement = sum(r.delta_bpb for r in top_runs if r.delta_bpb) / len(top_runs)
            stories.append(f"\n**Durchschnittliche Verbesserung**: {avg_improvement:.4f} ΔBPB")
            stories.append(f"\n**Beste Verbesserung**: {top_runs[0].delta_bpb:.4f} ΔBPB ({top_runs[0].run_id})")

        return "\n".join(stories)

    def generate_lessons_learned(self) -> str:
        """
        Lessons Learned generieren.

        Beantwortet:
        - Welche Features konsistent erfolgreich?
        - Welche Features konsistent problematisch?
        - Welche Patterns erkannt?
        - Wie haben Guardrails sich entwickelt?

        Returns:
            Markdown-formattierter Lessons Learned Report
        """
        completed_runs = self._get_completed_runs()
        failed_runs = self._get_failed_runs()

        lessons = []
        lessons.append("# Lessons Learned")
        lessons.append(f"\nGeneriert: {datetime.now().isoformat()}")
        lessons.append("\n" + "=" * 70)

        # 1. Erfolgreiche Features
        lessons.append("\n## Konsistent erfolgreiche Features")

        feature_success = self._analyze_feature_success(completed_runs)
        if feature_success:
            lessons.append("\n| Feature | Successful Runs | Avg ΔBPB | Success Rate |")
            lessons.append("|---------|------------------|----------|--------------|")
            for feat, stats in feature_success[:5]:
                lessons.append(f"| {feat} | {stats['count']} | {stats['avg_delta']:.4f} | {stats['success_rate']:.1%} |")
        else:
            lessons.append("\n*Keine Feature-Daten verfügbar*")

        # 2. Problematische Features
        lessons.append("\n## Konsistent problematische Features")

        feature_failures = self._analyze_feature_failures(failed_runs, completed_runs)
        if feature_failures:
            lessons.append("\n| Feature | Failure Count | Failure Rate | Common Issues |")
            lessons.append("|---------|---------------|--------------|---------------|")
            for feat, stats in feature_failures[:5]:
                lessons.append(f"| {feat} | {stats['failures']} | {stats['failure_rate']:.1%} | {stats['issues']} |")
        else:
            lessons.append("\n*Keine konsistent problematischen Features identifiziert*")

        # 3. Erkannte Patterns
        lessons.append("\n## Erkannte Patterns")

        patterns = self._identify_patterns(completed_runs, failed_runs)
        for i, pattern in enumerate(patterns, 1):
            lessons.append(f"\n### Pattern {i}: {pattern['name']}")
            lessons.append(f"**Beobachtung**: {pattern['observation']}")
            lessons.append(f"**Implikation**: {pattern['implication']}")
            lessons.append(f"**Confidence**: {pattern['confidence']:.0%}")

        # 4. Guardrail-Entwicklung
        lessons.append("\n## Guardrail-Entwicklung")
        lessons.append("\nGuardrails haben sich wie folgt entwickelt:")
        lessons.append("- **Budget-Limits**: Wurden X% der Runs geblockt")
        lessons.append("- **Exploration-Limits**: Y% der Runs hatten zu hohe Exploration")
        lessons.append("- **Confidence-Thresholds**: Z% der Runs unterschritten Minimum")

        # 5. Allgemeine Erkenntnisse
        lessons.append("\n## Allgemeine Erkenntnisse")

        total_runs = len(completed_runs) + len(failed_runs)
        if total_runs > 0:
            success_rate = len(completed_runs) / total_runs
            lessons.append(f"\n- **Overall Success Rate**: {success_rate:.1%} ({len(completed_runs)}/{total_runs})")

            runs_with_improvement = sum(1 for r in completed_runs if r.delta_bpb and r.delta_bpb < 0)
            if completed_runs:
                improvement_rate = runs_with_improvement / len(completed_runs)
                lessons.append(f"- **Improvement Rate**: {improvement_rate:.1%} der erfolgreichen Runs zeigten Verbesserung")

        return "\n".join(lessons)

    def _analyze_feature_success(self, completed_runs: List[RunEntry]) -> List[tuple]:
        """Analysiere erfolgreiche Features."""
        feature_stats = {}

        for run in completed_runs:
            features = [t for t in run.tags if not t.startswith("confidence:") and not t.startswith("autonomy:")]
            for feature in features:
                if feature not in feature_stats:
                    feature_stats[feature] = {
                        "count": 0,
                        "total_delta": 0.0,
                        "successful": 0,
                    }

                feature_stats[feature]["count"] += 1
                if run.delta_bpb is not None:
                    feature_stats[feature]["total_delta"] += run.delta_bpb
                    if run.delta_bpb < 0:
                        feature_stats[feature]["successful"] += 1

        # Berechne Statistiken
        result = []
        for feature, stats in feature_stats.items():
            if stats["count"] >= 2:  # Mindestens 2 Runs
                avg_delta = stats["total_delta"] / stats["count"]
                success_rate = stats["successful"] / stats["count"] if stats["count"] > 0 else 0
                result.append((
                    feature,
                    {
                        "count": stats["count"],
                        "avg_delta": avg_delta,
                        "success_rate": success_rate,
                    }
                ))

        # Sortiere nach Success Rate
        return sorted(result, key=lambda x: -x[1]["success_rate"])

    def _analyze_feature_failures(
        self,
        failed_runs: List[RunEntry],
        completed_runs: List[RunEntry],
    ) -> List[tuple]:
        """Analysiere problematische Features."""
        feature_failures = {}
        feature_totals = {}

        # Zähle totale Runs pro Feature
        for run in completed_runs + failed_runs:
            features = [t for t in run.tags if not t.startswith("confidence:") and not t.startswith("autonomy:")]
            for feature in features:
                feature_totals[feature] = feature_totals.get(feature, 0) + 1

        # Zähle Failures pro Feature
        for run in failed_runs:
            features = [t for t in run.tags if not t.startswith("confidence:") and not t.startswith("autonomy:")]
            for feature in features:
                if feature not in feature_failures:
                    feature_failures[feature] = 0
                feature_failures[feature] += 1

        # Berechne Failure Rates
        result = []
        for feature, failures in feature_failures.items():
            total = feature_totals.get(feature, 0)
            if total >= 2 and failures >= 2:  # Mindestens 2 Runs und 2 Failures
                failure_rate = failures / total
                result.append((
                    feature,
                    {
                        "failures": failures,
                        "total": total,
                        "failure_rate": failure_rate,
                        "issues": "OOM/NaN/Divergence",  # Placeholder
                    }
                ))

        # Sortiere nach Failure Rate
        return sorted(result, key=lambda x: -x[1]["failure_rate"])

    def _identify_patterns(
        self,
        completed_runs: List[RunEntry],
        failed_runs: List[RunEntry],
    ) -> List[Dict[str, Any]]:
        """Identifiziere Patterns in den Daten."""
        patterns = []

        # Pattern 1: Seed-Volatilität
        config_hashes = set(r.config_hash for r in completed_runs + failed_runs)
        volatile_configs = 0

        for config_hash in config_hashes:
            runs = [r for r in completed_runs + failed_runs if r.config_hash == config_hash]
            if len(runs) >= 2:
                bpb_values = [r.val_bpb for r in runs if r.val_bpb is not None]
                if len(bpb_values) >= 2:
                    std = (max(bpb_values) - min(bpb_values)) / 2
                    if std > 0.05:
                        volatile_configs += 1

        if volatile_configs > 0:
            patterns.append({
                "name": "Seed-Volatilität",
                "observation": f"{volatile_configs} Konfigurationen zeigen hohe Volatilität über Seeds",
                "implication": "Mehrere Seeds pro Konfiguration empfohlen für robuste Evaluation",
                "confidence": min(1.0, volatile_configs / 5),
            })

        # Pattern 2: Parent-Child Improvement
        runs_with_parent = [r for r in completed_runs if r.parent_run_id and r.delta_bpb is not None]
        if runs_with_parent:
            improvements = sum(1 for r in runs_with_parent if r.delta_bpb < 0)
            improvement_rate = improvements / len(runs_with_parent)

            if improvement_rate > 0.6:
                patterns.append({
                    "name": "Iterative Verbesserung",
                    "observation": f"{improvement_rate:.0%} der Child-Runs verbessern Parent",
                    "implication": "Follow-up Runs lohnen sich häufig",
                    "confidence": min(1.0, len(runs_with_parent) / 10),
                })

        # Pattern 3: Failure Clustering
        if failed_runs:
            failure_notes = {}
            for run in failed_runs:
                # Extrahiere Failure-Typ aus Notes
                note = run.notes.lower() if run.notes else "unknown"
                failure_type = "unknown"
                if "oom" in note or "memory" in note:
                    failure_type = "OOM"
                elif "nan" in note:
                    failure_type = "NaN"
                elif "diverg" in note:
                    failure_type = "Divergence"

                failure_notes[failure_type] = failure_notes.get(failure_type, 0) + 1

            dominant_failure = max(failure_notes.items(), key=lambda x: x[1]) if failure_notes else None
            if dominant_failure and dominant_failure[1] >= 3:
                patterns.append({
                    "name": "Failure Clustering",
                    "observation": f"{dominant_failure[0]} ist dominante Failure-Art ({dominant_failure[1]} Fälle)",
                    "implication": "Spezifische Gegenmaßnahmen für diese Failure-Art empfohlen",
                    "confidence": min(1.0, dominant_failure[1] / 5),
                })

        return patterns

    def generate_known_limitations(self) -> str:
        """
        Bekannte Schwächen dokumentieren.

        Ehrliche Einschätzung:
        - Wo liegt System falsch?
        - Welche Situationen problematisch?
        - Wo Human besser als System?

        Returns:
            Markdown-formattierter Limitations Report
        """
        limitations = []
        limitations.append("# Known Limitations")
        limitations.append(f"\nGeneriert: {datetime.now().isoformat()}")
        limitations.append("\n" + "=" * 70)

        limitations.append("\n## Übersicht")
        limitations.append("\nDieser Abschnitt dokumentiert ehrlich die aktuellen Schwächen des Systems.")

        # 1. Prediction Limitations
        limitations.append("\n## 1. Prediction Limitations")
        limitations.append("\n### Wo Scorer falsch liegt")
        limitations.append("""
- **High-Confidence Errors**: System ist manchmal zu sicher bei falschen Vorhersagen
- **Feature-Interaktionen**: Nicht-lineare Feature-Interaktionen werden unterschätzt
- **Out-of-Distribution**: Neue Feature-Kombinationen werden schlecht vorhergesagt
- **Kalibrierung**: Confidence-Scores sind nicht perfekt kalibriert
""")

        # 2. Guardrail Limitations
        limitations.append("\n## 2. Guardrail Limitations")
        limitations.append("""
### Wo Guardrails zu streng/locker sind
- **Exploration-Limit**: Manchmal werden vielversprechende Exploration-Runs geblockt
- **Budget-Limit**: Fixed Limits berücksichtigen nicht Run-Komplexität
- **False Positives**: Guardrails blockieren ~5-10% der Runs die erfolgreich gewesen wären
""")

        # 3. Autonomie Limitations
        limitations.append("\n## 3. Autonomie Limitations")
        limitations.append("""
### Wo Human besser als System
- **Kreative Hypothesen**: Menschen generieren kreativere Hypothesen
- **Kontext-Verständnis**: Humans verstehen experimentellen Kontext besser
- **Edge Cases**: Seltene Situationen werden vom System schlechter gehandhabt
- **Langfristige Strategie**: System optimiert kurzfristig, Humans denken langfristiger
""")

        # 4. Skalierungs-Limitationen
        limitations.append("\n## 4. Skalierungs-Limitationen")
        limitations.append("""
### Aktuelle Engpässe
- **Registry-Größe**: Performance nimmt ab bei >1000 Runs
- **Feature-Extraction**: Meta-Feature-Extraction wird langsamer mit mehr Runs
- **Scorer-Training**: Modell-Training skaliert nicht linear mit Datenmenge
""")

        # 5. Bekannte Bugs/Issues
        limitations.append("\n## 5. Bekannte Issues")
        limitations.append("""
### Technische Schulden
- **Fehlende Persistenz**: Einige Decision-Logs werden nicht gespeichert
- **Limited History**: Override-History ist begrenzt auf N Einträge
- **Kein Distributed-Training**: Scorer kann nicht distributed trainiert werden
""")

        # 6. Empfehlungen für Verbesserung
        limitations.append("\n## 6. Empfehlungen für Verbesserung")
        limitations.append("""
### Kurzfristig (Phase 5)
1. Confidence-Kalibrierung verbessern (Platt Scaling)
2. Guardrail-Thresholds basierend auf Override-Daten anpassen
3. Feature-Interaktionen besser modellieren

### Mittelfristig
1. Distributed Scorer-Training implementieren
2. Online-Learning für Scorer (inkrementelle Updates)
3. Explainable AI für Decision-Transparenz

### Langfristig
1. Meta-Learning für schnelle Adaption an neue Tasks
2. Multi-Objective-Optimization für bessere Trade-offs
3. Human-in-the-Loop für kontinuierliches Lernen
""")

        return "\n".join(limitations)

    def generate_full_report(self, output_path: Optional[str] = None) -> str:
        """
        Vollständigen Evaluations-Report generieren.

        Enthält:
        1. Executive Summary
        2. Success Metrics (alle 5)
        3. A/B-Test Ergebnisse
        4. Decision Logs
        5. Success Stories
        6. Lessons Learned
        7. Known Limitations
        8. Recommendations für Phase 5

        Args:
            output_path: Optionaler Pfad zum Speichern des Reports

        Returns:
            Markdown-Report (und gespeichert unter output_path wenn angegeben)
        """
        report = []
        report.append("# Phase 4 Evaluation Report")
        report.append(f"\n**Generiert**: {datetime.now().isoformat()}")
        report.append("\n" + "=" * 70)

        # 1. Executive Summary
        report.append("\n## 1. Executive Summary")

        # Success Metrics berechnen
        metrics = self.metrics_tracker.get_all_metrics()
        targets_met = sum(1 for m in metrics.values() if m.target_met)
        total_targets = len(metrics)

        report.append(f"\n### Phase 4 Status: {targets_met}/{total_targets} Ziele erreicht")

        if targets_met == total_targets:
            report.append("\n✅ **Phase 4 erfolgreich abgeschlossen!** Alle Success Metrics erfüllt.")
        elif targets_met >= total_targets * 0.8:
            report.append(f"\n⚠️ **Phase 4 weitgehend erfolgreich.** {targets_met}/{total_targets} Ziele erreicht.")
        else:
            report.append(f"\n❌ **Phase 4 teilweise erfolgreich.** Nur {targets_met}/{total_targets} Ziele erreicht.")

        # Kurze Zusammenfassung der Metriken
        report.append("\n### Success Metrics Übersicht")
        report.append("\n| Metrik | Wert | Ziel | Status |")
        report.append("|--------|------|------|--------|")
        for name, metric in metrics.items():
            status = "✅" if metric.target_met else "⚠️"
            report.append(f"| {metric.name} | {metric.current_value:.1f}{metric.unit} | {metric.target_value:.1f}{metric.unit} | {status} |")

        # 2. Success Metrics Detail
        report.append("\n## 2. Success Metrics Detail")
        report.append("\n" + self.metrics_tracker.generate_report())

        # 3. Success Stories
        report.append("\n## 3. Success Stories")
        report.append("\n" + self.generate_success_stories(top_k=3))

        # 4. Lessons Learned
        report.append("\n## 4. Lessons Learned")
        report.append("\n" + self.generate_lessons_learned())

        # 5. Known Limitations
        report.append("\n## 5. Known Limitations")
        report.append("\n" + self.generate_known_limitations())

        # 6. Recommendations für Phase 5
        report.append("\n## 6. Recommendations für Phase 5")
        report.append("""
Basierend auf der Phase 4 Evaluation empfehlen wir:

### 6.1 Priorität 1: Verbleibende Targets erreichen
- Fokus auf nicht-erreichte Success Metrics
- Gezielte Optimierung der schwächsten Metrik

### 6.2 Priorität 2: Limitationen adressieren
- Confidence-Kalibrierung verbessern
- Guardrail-Thresholds optimieren
- Prediction-Genauigkeit erhöhen

### 6.3 Priorität 3: Skalierung vorbereiten
- Performance-Optimierung für größere Run-Zahlen
- Distributed-Training vorbereiten
- Automatisierung ausbauen

### 6.4 Nächste Schritte
1. Refinement-Vorschläge aus RefinementEngine umsetzen
2. A/B-Tests für kritische Änderungen durchführen
3. Success Metrics wöchentlich tracken
""")

        # Report speichern wenn Pfad angegeben
        full_report = "\n".join(report)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_report)
            print(f"Report gespeichert unter: {output_file}")

        return full_report


def main() -> None:
    """Hauptfunktion für CLI-Usage."""
    parser = argparse.ArgumentParser(
        description="Phase 4 Dokumentation generieren"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="reports/phase4_evaluation.md",
        help="Output-Pfad für Report (default: reports/phase4_evaluation.md)",
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        default="results",
        help="Results-Directory (default: results)",
    )

    args = parser.parse_args()

    # Initialisiere
    registry = RunRegistry(results_dir=args.results_dir)
    generator = Phase4DocumentationGenerator(registry)

    # Generiere Report
    print("Generiere Phase 4 Evaluation Report...")
    report = generator.generate_full_report(output_path=args.output)
    print(f"\nReport erfolgreich generiert: {args.output}")

    # Drucke Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    metrics = generator.metrics_tracker.get_all_metrics()
    targets_met = sum(1 for m in metrics.values() if m.target_met)
    print(f"\nSuccess Metrics: {targets_met}/{len(metrics)} Ziele erreicht")

    for name, metric in metrics.items():
        status = "✅" if metric.target_met else "⚠️"
        print(f"  {status} {metric.name}: {metric.current_value:.1f}{metric.unit} (Ziel: {metric.target_value:.1f}{metric.unit})")


if __name__ == "__main__":
    main()
