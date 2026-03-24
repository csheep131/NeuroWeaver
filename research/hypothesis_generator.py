#!/usr/bin/env python3
"""
Hypothesis Generator für Phase 4A.

Generiert Run-Vorschläge basierend auf Contextual Bandits und Pattern Mining.

Strategien:
1. Exploitation: Bewährte Features kombinieren
2. Exploration: Neue Feature-Kombinationen testen
3. Pattern-based: Erfolgreiche Muster erkennen und anwenden
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from core.meta_features import RunMetaFeatures
from research.surrogate_scorer import SurrogateScorer


@dataclass
class RunHypothesis:
    """Ein Run-Vorschlag mit Begründung."""
    
    features_proposed: List[str]
    predicted_delta_bpb: float
    confidence: float
    hypothesis_type: Literal["exploitation", "exploration", "pattern_based"]
    reasoning: str
    similar_successful_runs: List[str]
    risk_level: Literal["low", "medium", "high"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary."""
        return {
            "features_proposed": self.features_proposed,
            "predicted_delta_bpb": self.predicted_delta_bpb,
            "confidence": self.confidence,
            "hypothesis_type": self.hypothesis_type,
            "reasoning": self.reasoning,
            "similar_successful_runs": self.similar_successful_runs,
            "risk_level": self.risk_level,
        }


class HypothesisGenerator:
    """
    Generiert Run-Vorschläge basierend auf historischen Erfolgen.
    
    Strategien:
    1. Exploitation: Bewährte Features kombinieren
    2. Exploration: Neue Feature-Kombinationen testen
    3. Pattern-based: Erfolgreiche Muster erkennen und anwenden
    
    Attributes:
        scorer: Surrogate Scorer für Vorhersagen
        meta_features: Historische Meta-Features aller Runs
        feature_success_rates: Erfolgsrate pro Feature
        feature_co_occurrence: Feature-Kookkurrenz Statistiken
    """
    
    # Alle verfügbaren Features für Exploration
    ALL_FEATURES: List[str] = [
        "gqa", "film", "leaky_relu", "swiglu", "rope",
        "mixed_quant", "int6_quant", "int5_quant", "gptq_lite",
        "flash_attn", "parallel_attn", "no_bias", "shared_embeddings",
    ]
    
    def __init__(
        self,
        scorer: SurrogateScorer,
        meta_features: List[RunMetaFeatures]
    ) -> None:
        """
        Initialisiere Hypothesis Generator.
        
        Args:
            scorer: Trainierter Surrogate Scorer
            meta_features: Historische Meta-Features
        """
        self.scorer = scorer
        self.meta_features = meta_features
        
        # Statistiken berechnen
        self.feature_success_rates: Dict[str, float] = {}
        self.feature_co_occurrence: Dict[Tuple[str, str], int] = {}
        self.feature_counts: Dict[str, int] = {}
        self.successful_runs: List[RunMetaFeatures] = []
        
        self._compute_statistics()
    
    def _compute_statistics(self) -> None:
        """Berechne Feature-Statistiken aus historischen Daten."""
        feature_success_sum: Dict[str, float] = {}
        feature_count: Dict[str, int] = {}
        co_occurrence: Dict[Tuple[str, str], int] = {}
        
        for feat in self.meta_features:
            is_successful = feat.delta_bpb_vs_parent is not None and feat.delta_bpb_vs_parent < 0
            
            if is_successful:
                self.successful_runs.append(feat)
            
            # Feature-Counts
            for feature in feat.features_active:
                feature_count[feature] = feature_count.get(feature, 0) + 1
                
                if is_successful:
                    feature_success_sum[feature] = feature_success_sum.get(feature, 0) + 1
                
                # Co-occurrence zählen
                active_sorted = sorted(feat.features_active)
                for i, f1 in enumerate(active_sorted):
                    for f2 in active_sorted[i + 1:]:
                        key = (f1, f2)
                        co_occurrence[key] = co_occurrence.get(key, 0) + 1
        
        # Erfolgsraten berechnen
        for feature in feature_count:
            success_count = feature_success_sum.get(feature, 0)
            self.feature_success_rates[feature] = (
                success_count / feature_count[feature]
                if feature_count[feature] > 0 else 0.0
            )
        
        self.feature_counts = feature_count
        self.feature_co_occurrence = co_occurrence
    
    def _create_hypothesis_features(
        self,
        features: List[str],
        predicted_bpb: float,
        confidence: float,
        hypothesis_type: Literal["exploitation", "exploration", "pattern_based"],
        reasoning: str,
        similar_runs: Optional[List[str]] = None,
    ) -> RunHypothesis:
        """
        Erstelle RunHypothesis mit automatischer Risk-Level-Bestimmung.
        
        Args:
            features: Vorgeschlagene Features
            predicted_bpb: Vorhergesagte ΔBPB
            confidence: Confidence-Score
            hypothesis_type: Typ der Hypothese
            reasoning: Begründung
            similar_runs: Ähnliche erfolgreiche Runs
            
        Returns:
            RunHypothesis mit berechnetem Risk-Level
        """
        # Risk-Level bestimmen
        risk_level = self._assess_risk(features, predicted_bpb, confidence)
        
        return RunHypothesis(
            features_proposed=features,
            predicted_delta_bpb=predicted_bpb,
            confidence=confidence,
            hypothesis_type=hypothesis_type,
            reasoning=reasoning,
            similar_successful_runs=similar_runs or [],
            risk_level=risk_level,
        )
    
    def _assess_risk(
        self,
        features: List[str],
        predicted_bpb: float,
        confidence: float
    ) -> Literal["low", "medium", "high"]:
        """
        Bewerte Risiko einer Feature-Kombination.
        
        Args:
            features: Features der Hypothese
            predicted_bpb: Vorhergesagte ΔBPB
            confidence: Confidence-Score
            
        Returns:
            Risk-Level
        """
        # Faktoren für Risiko
        n_features = len(features)
        avg_success_rate = np.mean([
            self.feature_success_rates.get(f, 0.5) for f in features
        ]) if features else 0.5
        
        # Neue Features erhöhen Risiko
        known_features = sum(1 for f in features if f in self.feature_counts)
        novelty_ratio = 1.0 - (known_features / max(len(features), 1))
        
        # Risiko-Score berechnen
        risk_score = 0.0
        
        # Viele Features = höheres Risiko
        if n_features >= 4:
            risk_score += 0.3
        elif n_features >= 3:
            risk_score += 0.15
        
        # Niedrige Erfolgsrate = höheres Risiko
        if avg_success_rate < 0.3:
            risk_score += 0.3
        elif avg_success_rate < 0.5:
            risk_score += 0.15
        
        # Niedrige Confidence = höheres Risiko
        if confidence < 0.5:
            risk_score += 0.25
        elif confidence < 0.7:
            risk_score += 0.1
        
        # Hohe Neuheit = höheres Risiko
        risk_score += novelty_ratio * 0.2
        
        # Klassifizierung
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        else:
            return "high"
    
    def _find_similar_successful_runs(
        self,
        features: List[str],
        limit: int = 3
    ) -> List[str]:
        """
        Finde ähnliche erfolgreiche Runs.
        
        Args:
            features: Features für Ähnlichkeitssuche
            limit: Maximale Anzahl zurückgegebener Runs
            
        Returns:
            Liste von Run-IDs
        """
        if not self.successful_runs:
            return []
        
        feature_set = set(features)
        similarities: List[Tuple[str, float]] = []
        
        for run in self.successful_runs:
            run_features = set(run.features_active)
            
            # Jaccard-Ähnlichkeit
            intersection = len(feature_set & run_features)
            union = len(feature_set | run_features)
            similarity = intersection / union if union > 0 else 0.0
            
            similarities.append((run.run_id, similarity))
        
        # Top-Ähnliche sortieren
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [run_id for run_id, _ in similarities[:limit] if _ > 0.0]
    
    def generate_exploitation_hypotheses(self, top_k: int = 5) -> List[RunHypothesis]:
        """
        Generiere Vorschläge basierend auf erfolgreichen Features.
        
        Nutze Features mit hoher Erfolgsrate.
        
        Args:
            top_k: Anzahl der Hypothesen
            
        Returns:
            Liste von RunHypothesis
        """
        hypotheses: List[RunHypothesis] = []
        
        # Features nach Erfolgsrate sortieren
        sorted_features = sorted(
            self.feature_success_rates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Top-Features identifizieren
        top_features = [f for f, rate in sorted_features[:6] if rate > 0.4]
        
        if not top_features:
            return hypotheses
        
        # Kombinationen aus Top-Features generieren
        # 1. Einzelne Top-Features
        for feature in top_features[:3]:
            # Mock-Features für Vorhersage erstellen
            mock_features = self._create_mock_features([feature])
            if mock_features is None:
                continue
            
            predicted_bpb, predicted_eff, confidence = self.scorer.predict(mock_features)
            
            reasoning = (
                f"Feature '{feature}' hat {self.feature_success_rates.get(feature, 0):.1%} "
                f"Erfolgsrate in {self.feature_counts.get(feature, 0)} Runs."
            )
            
            similar_runs = self._find_similar_successful_runs([feature])
            
            hypothesis = self._create_hypothesis_features(
                features=[feature],
                predicted_bpb=predicted_bpb,
                confidence=confidence,
                hypothesis_type="exploitation",
                reasoning=reasoning,
                similar_runs=similar_runs,
            )
            hypotheses.append(hypothesis)
        
        # 2. Kombinationen aus 2 Top-Features
        for i, f1 in enumerate(top_features[:4]):
            for f2 in top_features[i + 1:5]:
                # Prüfe ob Kombination schon erfolgreich war
                co_occurrence_key = (f1, f2) if f1 < f2 else (f2, f1)
                co_occurrence_count = self.feature_co_occurrence.get(co_occurrence_key, 0)
                
                mock_features = self._create_mock_features([f1, f2])
                if mock_features is None:
                    continue
                
                predicted_bpb, predicted_eff, confidence = self.scorer.predict(mock_features)
                
                if co_occurrence_count > 0:
                    reasoning = (
                        f"Kombination '{f1} + {f2}' war bereits {co_occurrence_count}x "
                        f"erfolgreich. Beide Features haben hohe Erfolgsraten."
                    )
                else:
                    reasoning = (
                        f"Kombination '{f1} + {f2}': Beide Features haben hohe "
                        f"Erfolgsraten ({self.feature_success_rates.get(f1, 0):.1%}, "
                        f"{self.feature_success_rates.get(f2, 0):.1%})."
                    )
                
                similar_runs = self._find_similar_successful_runs([f1, f2])
                
                hypothesis = self._create_hypothesis_features(
                    features=[f1, f2],
                    predicted_bpb=predicted_bpb,
                    confidence=confidence,
                    hypothesis_type="exploitation",
                    reasoning=reasoning,
                    similar_runs=similar_runs,
                )
                hypotheses.append(hypothesis)
        
        # Nach Confidence sortieren und Top-K zurückgeben
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:top_k]
    
    def generate_exploration_hypotheses(self, top_k: int = 5) -> List[RunHypothesis]:
        """
        Generiere Vorschläge für unerforschte Feature-Kombinationen.
        
        Nutze Contextual Bandits für Exploration vs Exploitation.
        
        Args:
            top_k: Anzahl der Hypothesen
            
        Returns:
            Liste von RunHypothesis
        """
        hypotheses: List[RunHypothesis] = []
        
        # Unerforschte Features identifizieren
        explored_features = set(self.feature_counts.keys())
        unexplored_features = [
            f for f in self.ALL_FEATURES if f not in explored_features
        ]
        
        # Selten erforschte Features (weniger als 3 Runs)
        rare_features = [
            f for f, count in self.feature_counts.items() if count < 3
        ]
        
        # 1. Komplett unerforschte Features
        for feature in unexplored_features[:2]:
            mock_features = self._create_mock_features([feature])
            if mock_features is None:
                continue
            
            predicted_bpb, predicted_eff, confidence = self.scorer.predict(mock_features)
            
            # Confidence für unerforschte Features reduzieren
            confidence *= 0.6
            
            reasoning = (
                f"Feature '{feature}' wurde noch nicht getestet. "
                f"Potenzielle Entdeckung mit moderatem Risiko."
            )
            
            hypothesis = self._create_hypothesis_features(
                features=[feature],
                predicted_bpb=predicted_bpb,
                confidence=confidence,
                hypothesis_type="exploration",
                reasoning=reasoning,
                similar_runs=[],
            )
            hypotheses.append(hypothesis)
        
        # 2. Seltene Features mit neuen Kombinationen
        for feature in rare_features[:3]:
            # Kombiniere mit etabliertem Feature
            established_features = [
                f for f, rate in self.feature_success_rates.items()
                if rate > 0.5 and f != feature
            ]
            
            if not established_features:
                continue
            
            partner = established_features[0]
            mock_features = self._create_mock_features([feature, partner])
            if mock_features is None:
                continue
            
            predicted_bpb, predicted_eff, confidence = self.scorer.predict(mock_features)
            
            # Confidence anpassen
            confidence *= 0.75
            
            reasoning = (
                f"Feature '{feature}' wurde nur {self.feature_counts.get(feature, 0)}x "
                f"getestet. Kombination mit bewährtem '{partner}' könnte "
                f"neue Erkenntnisse liefern."
            )
            
            similar_runs = self._find_similar_successful_runs([partner])
            
            hypothesis = self._create_hypothesis_features(
                features=[feature, partner],
                predicted_bpb=predicted_bpb,
                confidence=confidence,
                hypothesis_type="exploration",
                reasoning=reasoning,
                similar_runs=similar_runs,
            )
            hypotheses.append(hypothesis)
        
        # 3. Neue 3-Feature-Kombinationen (höheres Risiko)
        if len(rare_features) >= 2 and established_features:
            combo_features = rare_features[:2] + [established_features[0]]
            mock_features = self._create_mock_features(combo_features)
            
            if mock_features is not None:
                predicted_bpb, predicted_eff, confidence = self.scorer.predict(mock_features)
                confidence *= 0.5  # Niedrige Confidence für komplexe Exploration
                
                reasoning = (
                    f"Explorative 3-Feature-Kombination: {', '.join(combo_features)}. "
                    f"Höheres Risiko, aber potenziell hohe Belohnung."
                )
                
                hypothesis = self._create_hypothesis_features(
                    features=combo_features,
                    predicted_bpb=predicted_bpb,
                    confidence=confidence,
                    hypothesis_type="exploration",
                    reasoning=reasoning,
                    similar_runs=[],
                )
                hypotheses.append(hypothesis)
        
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:top_k]
    
    def generate_pattern_based_hypotheses(self, top_k: int = 5) -> List[RunHypothesis]:
        """
        Generiere Vorschläge basierend auf erfolgreichen Mustern.
        
        Erkenne: "Feature X + Y war erfolgreich, also probiere X + Y + Z"
        
        Args:
            top_k: Anzahl der Hypothesen
            
        Returns:
            Liste von RunHypothesis
        """
        hypotheses: List[RunHypothesis] = []
        
        # Erfolgreiche 2-Feature-Kombinationen finden
        successful_combos: List[Tuple[List[str], float]] = []
        
        for run in self.successful_runs:
            if len(run.features_active) == 2:
                avg_delta = run.delta_bpb_vs_parent or 0.0
                successful_combos.append((run.features_active.copy(), avg_delta))
        
        # Nach Erfolg sortieren
        successful_combos.sort(key=lambda x: x[1])
        
        # Für jede erfolgreiche Kombination: Drittes Feature hinzufügen
        for combo, delta in successful_combos[:5]:
            # Passende dritte Features finden
            for third_feature in self.ALL_FEATURES:
                if third_feature in combo:
                    continue
                
                # Prüfe ob dritte Feature sinnvoll ist
                co_occurrence_sum = sum(
                    self.feature_co_occurrence.get((min(f, third_feature), max(f, third_feature)), 0)
                    for f in combo
                )
                
                # Nur wenn mindestens eine Co-occurrence existiert
                if co_occurrence_sum == 0 and third_feature not in self.feature_success_rates:
                    continue
                
                extended_combo = combo + [third_feature]
                mock_features = self._create_mock_features(extended_combo)
                
                if mock_features is None:
                    continue
                
                predicted_bpb, predicted_eff, confidence = self.scorer.predict(mock_features)
                
                reasoning = (
                    f"Muster-Erkennung: {', '.join(combo)} war erfolgreich "
                    f"(ΔBPB: {delta:.4f}). Erweiterung um '{third_feature}' "
                    f"basierend auf Co-occurrence Mustern."
                )
                
                similar_runs = self._find_similar_successful_runs(combo)
                
                hypothesis = self._create_hypothesis_features(
                    features=extended_combo,
                    predicted_bpb=predicted_bpb,
                    confidence=confidence,
                    hypothesis_type="pattern_based",
                    reasoning=reasoning,
                    similar_runs=similar_runs,
                )
                hypotheses.append(hypothesis)
                
                # Nur beste Erweiterung pro Combo
                break
        
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:top_k]
    
    def _create_mock_features(self, features: List[str]) -> Optional[RunMetaFeatures]:
        """
        Erstelle Mock-RunMetaFeatures für Vorhersage.
        
        Args:
            features: Aktive Features
            
        Returns:
            RunMetaFeatures oder None bei Fehler
        """
        try:
            # Durchschnittswerte aus historischen Daten verwenden
            avg_lineage = np.mean([f.lineage_depth for f in self.meta_features]) if self.meta_features else 1
            avg_siblings = np.mean([f.siblings_count for f in self.meta_features]) if self.meta_features else 2
            
            return RunMetaFeatures(
                run_id="hypothesis_mock",
                features_active=features,
                lineage_depth=int(avg_lineage),
                siblings_count=int(avg_siblings),
                budget_class="medium",
                sequence_length="local",
                quantization_type="none",
                step_time_ms=10.0,
                memory_usage_mb=1024.0,
                training_stability=0.8,
                delta_bpb_vs_parent=None,
                efficiency_gain_percent=None,
                model_size_change_percent=None,
                quant_gap=0.0,
            )
        except Exception:
            return None
    
    def generate_all(self) -> List[RunHypothesis]:
        """
        Alle Hypothesen generieren und nach Confidence sortieren.
        
        Returns:
            Kombinierte und sortierte Liste aller Hypothesen
        """
        all_hypotheses: List[RunHypothesis] = []
        
        # Alle Strategien ausführen
        exploitation = self.generate_exploitation_hypotheses(top_k=5)
        exploration = self.generate_exploration_hypotheses(top_k=5)
        pattern_based = self.generate_pattern_based_hypotheses(top_k=5)
        
        all_hypotheses.extend(exploitation)
        all_hypotheses.extend(exploration)
        all_hypotheses.extend(pattern_based)
        
        # Nach Confidence sortieren
        all_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return all_hypotheses
    
    def get_feature_success_rates(self) -> Dict[str, float]:
        """
        Erfolgsrate pro Feature berechnen.
        
        Returns:
            Dictionary {feature: success_rate}
        """
        return self.feature_success_rates.copy()
    
    def get_diminishing_returns_features(self) -> List[str]:
        """
        Features mit sinkendem Grenznutzen identifizieren.
        
        Returns:
            Liste von Features mit diminishing returns
        """
        diminishing: List[str] = []
        
        # Features mit vielen Runs aber niedriger Erfolgsrate
        for feature, count in self.feature_counts.items():
            if count >= 5:  # Mindestens 5 Runs
                success_rate = self.feature_success_rates.get(feature, 0.0)
                if success_rate < 0.3:  # Weniger als 30% Erfolgsrate
                    diminishing.append(feature)
        
        return diminishing
    
    def get_feature_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Feature-Korrelationsmatrix berechnen.
        
        Returns:
            Nested Dictionary {feature1: {feature2: correlation}}
        """
        if not self.meta_features:
            return {}
        
        # Feature-Präsenz-Matrix erstellen
        feature_names = sorted(set(
            f for feat in self.meta_features for f in feat.features_active
        ))
        
        if len(feature_names) < 2:
            return {}
        
        # Matrix initialisieren
        correlation_matrix: Dict[str, Dict[str, float]] = {
            f1: {f2: 0.0 for f2 in feature_names}
            for f1 in feature_names
        }
        
        # Co-occurrence zu Korrelation umwandeln
        total_runs = len(self.meta_features)
        
        for f1 in feature_names:
            for f2 in feature_names:
                if f1 == f2:
                    correlation_matrix[f1][f2] = 1.0
                    continue
                
                key = (min(f1, f2), max(f1, f2))
                co_occurrence = self.feature_co_occurrence.get(key, 0)
                
                # Einfache Korrelation: Co-occurrence / Gesamt-Runs
                correlation_matrix[f1][f2] = co_occurrence / total_runs if total_runs > 0 else 0.0
        
        return correlation_matrix
