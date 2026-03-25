#!/usr/bin/env python3
"""
Refinement Engine für NeuroWeave Phase 4.

Basierend auf Performance Feedback-Schleife.

Analyse-Punkte:
1. Guardrail Violations: Zu strenge/lockere Grenzen?
2. Prediction Errors: Wo lag Scorer falsch?
3. Human Overrides: Was hat Human überschrieben?
4. Quarantine Patterns: Welche Features zu oft blockiert?
5. Drift Alerts: Welche Trends erkennen?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple
from collections import defaultdict

import numpy as np

from core.registry import RunRegistry, RunEntry
from research.surrogate_scorer import SurrogateScorer
from orchestrator.guardrails import GuardrailManager, Guardrail, GuardrailType
from research.override_learner import OverrideLearner, OverrideEvent


@dataclass
class RefinementSuggestion:
    """Vorschlag für Refinement."""

    component: Literal["guardrails", "scorer", "hypothesis_generator", "quarantine", "thresholds", "other"]
    current_behavior: str
    suggested_change: str
    expected_improvement: str
    confidence: float  # 0.0 - 1.0
    evidence: List[str] = field(default_factory=list)  # Run-IDs, Statistiken
    priority: int = 3  # 1 = highest, 5 = lowest

    def __post_init__(self) -> None:
        """Validiere und setze Priority basierend auf Confidence."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence muss zwischen 0.0 und 1.0 sein, ist: {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary."""
        return {
            "component": self.component,
            "current_behavior": self.current_behavior,
            "suggested_change": self.suggested_change,
            "expected_improvement": self.expected_improvement,
            "confidence": round(self.confidence, 2),
            "evidence": list(self.evidence),
            "priority": self.priority,
        }


class RefinementEngine:
    """
    Engine für iteratives Refinement von Phase 4 Komponenten.

    Analyse-Punkte:
    1. Guardrail Violations: Zu strenge/lockere Grenzen?
    2. Prediction Errors: Wo lag Scorer falsch?
    3. Human Overrides: Was hat Human überschrieben?
    4. Quarantine Patterns: Welche Features zu oft blockiert?
    5. Drift Alerts: Welche Trends erkennen?

    Example:
        engine = RefinementEngine(scorer, guardrail_manager, override_learner, registry)

        # Analysieren
        guardrail_suggestions = engine.analyze_guardrail_performance()
        prediction_suggestions = engine.analyze_prediction_errors()
        override_suggestions = engine.analyze_human_overrides()

        # Report generieren
        report = engine.generate_refinement_report()
        print(report)

        # Refinement anwenden
        for suggestion in guardrail_suggestions[:3]:
            engine.apply_refinement(suggestion)
    """

    def __init__(
        self,
        scorer: SurrogateScorer,
        guardrail_manager: GuardrailManager,
        override_learner: OverrideLearner,
        registry: RunRegistry,
    ) -> None:
        """
        Initialisiere Refinement Engine.

        Args:
            scorer: SurrogateScorer für Prediction-Analyse
            guardrail_manager: GuardrailManager für Guardrail-Analyse
            override_learner: OverrideLearner für Human-Override-Analyse
            registry: RunRegistry für Datenzugriff
        """
        self.scorer = scorer
        self.guardrail_manager = guardrail_manager
        self.override_learner = override_learner
        self.registry = registry

        # Thresholds für Analyse
        self.config = {
            "high_override_rate_threshold": 0.30,  # 30% Override-Rate als "hoch"
            "prediction_error_threshold": 0.50,  # 50% Fehler als "signifikant"
            "min_samples_for_suggestion": 5,  # Mindestens 5 Samples für Vorschlag
        }

    def _get_completed_runs(self) -> List[RunEntry]:
        """Hole alle abgeschlossenen Runs."""
        return self.registry.list_runs(status="completed")

    def _get_failed_runs(self) -> List[RunEntry]:
        """Hole alle fehlgeschlagenen Runs."""
        all_runs = self.registry.list_runs()
        return [r for r in all_runs if r.status in ("failed", "killed")]

    def analyze_guardrail_performance(self) -> List[RefinementSuggestion]:
        """
        Guardrail-Performance analysieren.

        Fragen:
        - Wie viele Runs wurden blockiert?
        - Waren Blockierungen gerechtfertigt?
        - Human hat Blockierungen wie oft überschrieben?

        Returns:
            Liste von RefinementSuggestions
        """
        suggestions: List[RefinementSuggestion] = []
        overrides = self.override_learner.overrides

        # 1. Analyse: Guardrail-Violations die vom Human überschrieben wurden
        guardrail_overrides = defaultdict(list)
        for override in overrides:
            if override.guardrail_violations:
                for violation in override.guardrail_violations:
                    guardrail_overrides[violation].append(override)

        # 2. Prüfe jede Guardrail auf häufige Overrides
        for guardrail_type, guardrail in self._get_guardrails_by_type():
            override_count = len(guardrail_overrides.get(guardrail_type.value, []))
            total_checks = self._estimate_guardrail_checks(guardrail_type)

            if total_checks == 0:
                continue

            override_rate = override_count / total_checks

            # Hohe Override-Rate → Guardrail zu streng
            if override_rate > self.config["high_override_rate_threshold"]:
                confidence = min(1.0, override_rate / 0.5)  # Confidence skaliert mit Rate

                suggestions.append(RefinementSuggestion(
                    component="guardrails",
                    current_behavior=f"Guardrail '{guardrail.name}' hat {override_count} Overrides bei {total_checks} Prüfungen ({override_rate:.1%})",
                    suggested_change=f"Threshold von {guardrail.threshold:.2f} auf {guardrail.threshold * 0.9:.2f} reduzieren (10% lockerer)",
                    expected_improvement=f"{int((override_rate - self.config['high_override_rate_threshold']) * 100)}% weniger falsche Blockierungen",
                    confidence=confidence,
                    evidence=[f"Override-Rate: {override_rate:.2%}", f"Betroffene Overrides: {override_count}"],
                    priority=1 if confidence > 0.7 else 2,
                ))

        # 3. Analyse: Guardrails die nie verletzt werden (potentiell unnötig)
        for guardrail_type, guardrail in self._get_guardrails_by_type():
            violation_count = sum(
                1 for o in overrides
                if guardrail_type.value in o.guardrail_violations
            )

            # Wenn Guardrail nie verletzt wurde, könnte sie zu streng sein
            if violation_count == 0 and guardrail.is_hard_limit:
                suggestions.append(RefinementSuggestion(
                    component="guardrails",
                    current_behavior=f"Guardrail '{guardrail.name}' wurde nie verletzt (0 Violations)",
                    suggested_change=f"Guardrail auf 'soft limit' umstellen oder entfernen",
                    expected_improvement="Weniger Overhead, schnellere Entscheidungen",
                    confidence=0.5,
                    evidence=["0 Violations in Historie"],
                    priority=4,
                ))

        return sorted(suggestions, key=lambda s: s.priority)

    def _get_guardrails_by_type(self) -> List[Tuple[GuardrailType, Guardrail]]:
        """Hole alle Guardrails mit ihrem Typ."""
        # Hole Guardrails vom Manager (wenn verfügbar)
        try:
            guardrails = self.guardrail_manager.get_all_guardrails()
            result = []
            for g in guardrails:
                result.append((g.guardrail_type, g))
            return result
        except Exception:
            # Fallback: Leere Liste
            return []

    def _estimate_guardrail_checks(self, guardrail_type: GuardrailType) -> int:
        """Schätze Anzahl der Guardrail-Prüfungen."""
        # Einfache Schätzung: Anzahl aller Runs
        return len(self.registry.list_runs())

    def analyze_prediction_errors(self) -> List[RefinementSuggestion]:
        """
        Vorhersage-Fehler analysieren.

        Fragen:
        - Wo lag Scorer signifikant falsch?
        - Welche Features wurden falsch gewichtet?
        - Confidence war zu hoch/niedrig?

        Returns:
            Liste von RefinementSuggestions
        """
        suggestions: List[RefinementSuggestion] = []
        completed_runs = self._get_completed_runs()

        if len(completed_runs) < self.config["min_samples_for_suggestion"]:
            return suggestions

        # 1. Sammle Vorhersagen und tatsächliche Werte
        prediction_errors: List[Tuple[str, float, float, Optional[float]]] = []  # (run_id, predicted, actual, confidence)

        for run in completed_runs:
            if run.delta_bpb is None:
                continue

            # Hole Confidence aus Tags wenn verfügbar
            confidence = None
            for tag in run.tags:
                if tag.startswith("confidence:"):
                    try:
                        confidence = float(tag.split(":")[1])
                    except (ValueError, IndexError):
                        pass

            # Scorer Vorhersage (wenn Scorers verfügbar)
            try:
                # Hinweis: In echter Implementierung würde man Features extrahieren
                # und Scorers vorhersage lassen
                # Hier: Placeholder-Logik
                predicted = 0.0  # Placeholder
                actual = run.delta_bpb
                prediction_errors.append((run.run_id, predicted, actual, confidence))
            except Exception:
                continue

        if not prediction_errors:
            return suggestions

        # 2. Analysiere Fehler-Verteilung
        errors = [actual - pred for _, pred, actual, _ in prediction_errors]
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mae = np.mean(np.abs(errors))  # Mean Absolute Error

        # 3. Systematische Überschätzung/Unterschätzung
        if abs(mean_error) > self.config["prediction_error_threshold"] * std_error:
            direction = "überschätzt" if mean_error < 0 else "unterschätzt"
            confidence = min(1.0, abs(mean_error) / (2 * std_error)) if std_error > 0 else 0.5

            suggestions.append(RefinementSuggestion(
                component="scorer",
                current_behavior=f"Scorer {direction} systematisch ΔBPB (Mean Error: {mean_error:.4f})",
                suggested_change="Bias-Korrektur im Scorer-Modell oder Feature-Gewichtung anpassen",
                expected_improvement=f"Reduktion des systematischen Fehlers um {abs(mean_error):.4f}",
                confidence=confidence,
                evidence=[
                    f"Mean Error: {mean_error:.4f}",
                    f"Std Error: {std_error:.4f}",
                    f"MAE: {mae:.4f}",
                ],
                priority=2 if confidence > 0.6 else 3,
            ))

        # 4. High-Confidence Errors (Scorer war sicher aber falsch)
        high_conf_errors = [
            (run_id, pred, actual, conf)
            for run_id, pred, actual, conf in prediction_errors
            if conf is not None and conf >= 0.7 and abs(actual - pred) > 0.05
        ]

        if len(high_conf_errors) >= 3:
            error_rate = len(high_conf_errors) / len([x for x in prediction_errors if x[3] is not None])

            suggestions.append(RefinementSuggestion(
                component="scorer",
                current_behavior=f"{len(high_conf_errors)} High-Confidence Errors (>70% Confidence, >0.05 Fehler)",
                suggested_change="Confidence-Kalibrierung: Platt Scaling oder Isotonic Regression",
                expected_improvement=f"Bessere Confidence-Zuverlässigkeit (aktuell {1-error_rate:.1%} Accuracy bei high confidence)",
                confidence=min(1.0, error_rate),
                evidence=[f"High-Confidence Error-Rate: {error_rate:.2%}"],
                priority=2,
            ))

        return sorted(suggestions, key=lambda s: s.priority)

    def analyze_human_overrides(self) -> List[RefinementSuggestion]:
        """
        Human-Overrides analysieren (via OverrideLearner).

        Fragen:
        - Welche Actions wurden häufig überschrieben?
        - Pattern erkennbar (z.B. Human blockt immer Feature X)?
        - Threshold-Anpassungen nötig?

        Returns:
            Liste von RefinementSuggestions
        """
        suggestions: List[RefinementSuggestion] = []
        overrides = self.override_learner.overrides

        if len(overrides) < self.config["min_samples_for_suggestion"]:
            return suggestions

        # 1. Analyse: Override-Rate nach Action-Typ
        action_overrides = defaultdict(list)
        for override in overrides:
            action_type = override.action_type or "unknown"
            action_overrides[action_type].append(override)

        total_actions = len(self.registry.list_runs())

        for action_type, action_overrides_list in action_overrides.items():
            override_rate = len(action_overrides_list) / total_actions if total_actions > 0 else 0

            if override_rate > self.config["high_override_rate_threshold"]:
                # Pattern-Analyse
                justifications = [o.justification for o in action_overrides_list if o.justification]
                common_justification = self._find_common_pattern(justifications)

                confidence = min(1.0, override_rate / 0.4)

                suggestions.append(RefinementSuggestion(
                    component="thresholds",
                    current_behavior=f"Action '{action_type}' wird in {override_rate:.1%} der Fällen überschrieben",
                    suggested_change=f"Threshold für '{action_type}' anpassen oder Decision-Logik überprüfen",
                    expected_improvement=f"{int(override_rate * 100)}% weniger Human-Interventionen",
                    confidence=confidence,
                    evidence=[
                        f"Override-Rate: {override_rate:.2%}",
                        f"Anzahl Overrides: {len(action_overrides_list)}",
                    ] + ([f"Häufigste Begründung: {common_justification}"] if common_justification else []),
                    priority=1 if confidence > 0.7 else 2,
                ))

        # 2. Analyse: Human blockt bestimmte Features konsistent
        feature_blocks = defaultdict(int)
        for override in overrides:
            if override.human_decision == "block":
                # Extrahiere Features aus Context
                context = override.context or {}
                features = context.get("features_active", [])
                for feature in features:
                    feature_blocks[feature] += 1

        # Features die konsistent geblockt werden
        for feature, block_count in feature_blocks.items():
            if block_count >= 5:
                suggestions.append(RefinementSuggestion(
                    component="hypothesis_generator",
                    current_behavior=f"Feature '{feature}' wurde {block_count}x vom Human geblockt",
                    suggested_change=f"Feature '{feature}' in Quarantäne verschieben oder Gewicht reduzieren",
                    expected_improvement="Bessere Alignment mit Human-Präferenzen",
                    confidence=min(1.0, block_count / 10),
                    evidence=[f"Block-Count: {block_count}"],
                    priority=2,
                ))

        # 3. Confidence-Kalibrierung
        confidence_data = [
            (o.confidence_before, o.human_decision != o.original_decision)
            for o in overrides
            if o.confidence_before is not None
        ]

        if len(confidence_data) >= 10:
            confidences = [c for c, _ in confidence_data]
            wrong_decisions = [1 if wrong else 0 for _, wrong in confidence_data]

            correlation = np.corrcoef(confidences, wrong_decisions)[0, 1] if len(confidences) > 2 else 0

            if not np.isnan(correlation) and abs(correlation) > 0.3:
                suggestions.append(RefinementSuggestion(
                    component="scorer",
                    current_behavior=f"Confidence korreliert mit falschen Entscheidungen (r={correlation:.2f})",
                    suggested_change="Confidence-Score neu kalibrieren",
                    expected_improvement="Zuverlässigere Confidence-Aussagen",
                    confidence=min(1.0, abs(correlation)),
                    evidence=[f"Korrelation: {correlation:.3f}"],
                    priority=2,
                ))

        return sorted(suggestions, key=lambda s: s.priority)

    def _find_common_pattern(self, texts: List[str]) -> Optional[str]:
        """Finde gemeinsames Pattern in Texten (einfache Heuristik)."""
        if not texts:
            return None

        # Einfache Wort-Häufigkeit
        word_counts = defaultdict(int)
        for text in texts:
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Nur längere Wörter
                    word_counts[word] += 1

        if not word_counts:
            return None

        # Häufigstes Wort
        most_common = max(word_counts.items(), key=lambda x: x[1])
        if most_common[1] >= len(texts) * 0.3:  # Mindestens 30% der Texte
            return most_common[0]

        return None

    def generate_refinement_report(self) -> str:
        """
        Refinement-Report generieren.

        Returns:
            Markdown-formattierter Report mit:
            - Top 5 Refinement Suggestions
            - Priority Ranking
            - Expected Impact
        """
        # Alle Analysen durchführen
        guardrail_suggestions = self.analyze_guardrail_performance()
        prediction_suggestions = self.analyze_prediction_errors()
        override_suggestions = self.analyze_human_overrides()

        # Alle Suggestions kombinieren und nach Priority sortieren
        all_suggestions = guardrail_suggestions + prediction_suggestions + override_suggestions
        all_suggestions = sorted(all_suggestions, key=lambda s: (s.priority, -s.confidence))

        report = []
        report.append("# Phase 4 Refinement Report")
        report.append(f"\nGeneriert: {datetime.now().isoformat()}")
        report.append("\n" + "=" * 70)

        # Zusammenfassung
        report.append(f"\n## Zusammenfassung")
        report.append(f"\n**{len(all_suggestions)} Refinement-Vorschläge identifiziert**")
        report.append(f"- Priority 1 (Hoch): {sum(1 for s in all_suggestions if s.priority == 1)}")
        report.append(f"- Priority 2 (Mittel): {sum(1 for s in all_suggestions if s.priority == 2)}")
        report.append(f"- Priority 3-5 (Niedrig): {sum(1 for s in all_suggestions if s.priority >= 3)}")

        # Top 5 Suggestions
        report.append("\n## Top 5 Refinement-Vorschläge")

        for i, suggestion in enumerate(all_suggestions[:5], 1):
            priority_icon = "🔴" if suggestion.priority == 1 else "🟡" if suggestion.priority == 2 else "🟢"
            report.append(f"\n### {i}. {suggestion.component.capitalize()} {priority_icon}")
            report.append(f"\n**Current Behavior:** {suggestion.current_behavior}")
            report.append(f"\n**Suggested Change:** {suggestion.suggested_change}")
            report.append(f"\n**Expected Improvement:** {suggestion.expected_improvement}")
            report.append(f"\n**Confidence:** {suggestion.confidence:.0%}")
            if suggestion.evidence:
                report.append(f"\n**Evidence:**")
                for ev in suggestion.evidence[:3]:
                    report.append(f"  - {ev}")

        # Nach Komponente gruppiert
        report.append("\n## Vorschläge nach Komponente")

        by_component = defaultdict(list)
        for s in all_suggestions:
            by_component[s.component].append(s)

        for component, component_suggestions in sorted(by_component.items()):
            report.append(f"\n### {component.capitalize()} ({len(component_suggestions)} Vorschläge)")
            for s in component_suggestions:
                report.append(f"- [P{s.priority}] {s.suggested_change[:80]}...")

        # Implementierungsempfehlungen
        report.append("\n## Implementierungsempfehlungen")

        high_priority = [s for s in all_suggestions if s.priority == 1]
        if high_priority:
            report.append("\n**Sofort umsetzen (Priority 1):**")
            for s in high_priority:
                report.append(f"1. {s.suggested_change}")
        else:
            report.append("\n✅ Keine kritischen Refinements identifiziert.")

        return "\n".join(report)

    def apply_refinement(self, suggestion: RefinementSuggestion) -> bool:
        """
        Refinement anwenden.

        Args:
            suggestion: Anzuwendende RefinementSuggestion

        Returns:
            True wenn erfolgreich angewendet, False sonst
        """
        # Hinweis: Echte Implementierung würde hier die tatsächlichen
        # Komponenten anpassen. Für jetzt: Placeholder-Logik.

        if suggestion.component == "guardrails":
            # Würde Guardrail-Thresholds anpassen
            return self._apply_guardrail_refinement(suggestion)
        elif suggestion.component == "scorer":
            # Würde Scorer-Parameter anpassen
            return self._apply_scorer_refinement(suggestion)
        elif suggestion.component == "thresholds":
            # Würde Decision-Thresholds anpassen
            return self._apply_threshold_refinement(suggestion)
        else:
            # Andere Komponenten: Logging nur
            return True

    def _apply_guardrail_refinement(self, suggestion: RefinementSuggestion) -> bool:
        """Guardrail-Refinement anwenden."""
        # Placeholder: In echter Implementierung würde man die Guardrail-Thresholds
        # im GuardrailManager anpassen
        return True

    def _apply_scorer_refinement(self, suggestion: RefinementSuggestion) -> bool:
        """Scorer-Refinement anwenden."""
        # Placeholder: In echter Implementierung würde man Scorer-Parameter
        # anpassen oder Modell neu trainieren
        return True

    def _apply_threshold_refinement(self, suggestion: RefinementSuggestion) -> bool:
        """Threshold-Refinement anwenden."""
        # Placeholder: In echter Implementierung würde man Decision-Thresholds
        # im AutonomyOrchestrator anpassen
        return True
