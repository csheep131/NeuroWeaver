#!/usr/bin/env python3
"""
Adaptive Kill Thresholds für Phase 4A.

Kontext-sensitive Grenzen für Run-Abbruch.

Prinzip:
- Strenger bei knappem Budget
- Explorativer bei viel Budget
- Anpassung basierend auf Erfolgsrate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from core.meta_features import RunMetaFeatures


@dataclass
class KillThresholds:
    """
    Adaptive Kill-Thresholds für einen Kontext.
    
    Attributes:
        context: Kontext-Bezeichnung ("low_budget", "medium_budget", "high_budget")
        max_delta_bpb: Maximal erlaubte Verschlechterung (BPB-Anstieg)
        min_efficiency_gain: Minimal erforderlicher Effizienz-Gewinn (%)
        max_quant_gap: Maximaler Quantisierungs-Verlust (BPB-Anstieg nach Quant)
        max_step_time_increase: Maximaler Step-Time Anstieg (%)
        max_memory_increase: Maximaler Speicher-Anstieg (%)
        min_training_stability: Minimale Trainings-Stabilität (0-1)
    """
    
    context: str
    max_delta_bpb: float = 0.05  # 5% BPB-Anstieg erlaubt
    min_efficiency_gain: float = -5.0  # -5% = 5% Verlust erlaubt
    max_quant_gap: float = 0.02  # 2% BPB-Anstieg nach Quant
    max_step_time_increase: float = 20.0  # 20% langsamer erlaubt
    max_memory_increase: float = 30.0  # 30% mehr Speicher erlaubt
    min_training_stability: float = 0.6  # Mindestens 60% Stabilität
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary."""
        return {
            "context": self.context,
            "max_delta_bpb": self.max_delta_bpb,
            "min_efficiency_gain": self.min_efficiency_gain,
            "max_quant_gap": self.max_quant_gap,
            "max_step_time_increase": self.max_step_time_increase,
            "max_memory_increase": self.max_memory_increase,
            "min_training_stability": self.min_training_stability,
        }
    
    def is_stricter_than(self, other: "KillThresholds") -> bool:
        """
        Prüfe ob diese Thresholds strenger sind als andere.
        
        Args:
            other: Andere KillThresholds
            
        Returns:
            True wenn diese Thresholds strenger sind
        """
        return (
            self.max_delta_bpb < other.max_delta_bpb and
            self.min_efficiency_gain > other.min_efficiency_gain and
            self.max_quant_gap < other.max_quant_gap
        )


class AdaptiveKillThresholdManager:
    """
    Verwaltet kontext-sensitive Kill-Thresholds.
    
    Prinzip:
    - Strenger bei knappem Budget
    - Explorativer bei viel Budget
    - Anpassung basierend auf Erfolgsrate
    
    Attributes:
        thresholds: Dictionary von Budget-Klasse zu Thresholds
        feature_thresholds: Feature-spezifische Thresholds
        recent_decisions: Historie der Kill-Entscheidungen
    """
    
    # Feature-spezifische Threshold-Anpassungen
    FEATURE_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
        "mixed_quant": {"max_quant_gap": 0.05},  # Mehr Toleranz für Mixed Quant
        "gptq_lite": {"max_quant_gap": 0.04},
        "int6_quant": {"max_quant_gap": 0.03},
        "flash_attn": {"max_step_time_increase": 10.0},  # Weniger Toleranz für Speed
        "gqa": {"max_memory_increase": 20.0},
    }
    
    def __init__(self) -> None:
        """Initialisiere Threshold Manager mit Default-Werten."""
        self.thresholds: Dict[str, KillThresholds] = {
            "low_budget": KillThresholds(
                context="low_budget",
                max_delta_bpb=0.02,  # Streng: nur 2% Verschlechterung
                min_efficiency_gain=-2.0,  # Nur 2% Verlust erlaubt
                max_quant_gap=0.01,  # Sehr streng bei Quant
                max_step_time_increase=10.0,  # Max 10% langsamer
                max_memory_increase=15.0,  # Max 15% mehr Speicher
                min_training_stability=0.7,  # Hohe Stabilität erforderlich
            ),
            "medium_budget": KillThresholds(
                context="medium_budget",
                max_delta_bpb=0.05,  # 5% Verschlechterung
                min_efficiency_gain=-5.0,  # 5% Verlust erlaubt
                max_quant_gap=0.02,
                max_step_time_increase=20.0,
                max_memory_increase=30.0,
                min_training_stability=0.6,
            ),
            "high_budget": KillThresholds(
                context="high_budget",
                max_delta_bpb=0.10,  # 10% Verschlechterung erlaubt
                min_efficiency_gain=-10.0,  # 10% Verlust erlaubt
                max_quant_gap=0.03,  # Mehr Exploration bei Quant
                max_step_time_increase=40.0,  # Mehr Toleranz für Speed
                max_memory_increase=50.0,  # Mehr Speicher erlaubt
                min_training_stability=0.5,  # Niedrigere Stabilitäts-Anforderung
            ),
        }
        
        # Feature-spezifische Thresholds
        self.feature_thresholds: Dict[str, KillThresholds] = {}
        
        # Historie der Entscheidungen
        self.recent_decisions: List[Tuple[str, bool, str]] = []  # (run_id, killed, reason)
        self.success_rate_history: List[float] = []
        
        # Adaptions-Parameter
        self._adaptation_rate = 0.1  # Wie stark Thresholds angepasst werden
        self._min_success_rate = 0.3  # Unter 30%: Thresholds lockern
        self._max_success_rate = 0.8  # Über 80%: Thresholds strengen
    
    def get_thresholds(self, budget_class: str) -> KillThresholds:
        """
        Thresholds für Budget-Klasse zurückgeben.
        
        Args:
            budget_class: "low", "medium", "high" oder "low_budget", "medium_budget", "high_budget"
            
        Returns:
            KillThresholds für die Budget-Klasse
            
        Raises:
            ValueError: Bei ungültiger Budget-Klasse
        """
        # Normalisiere Budget-Klasse (unterstütze beide Formate)
        normalized = budget_class
        if budget_class in ("low", "medium", "high"):
            normalized = f"{budget_class}_budget"
        
        if normalized not in self.thresholds:
            raise ValueError(
                f"Ungültige Budget-Klasse: {budget_class}. "
                f"Erlaubt: 'low', 'medium', 'high' oder 'low_budget', 'medium_budget', 'high_budget'"
            )
        
        return self.thresholds[normalized]
    
    def should_kill_run(
        self,
        run_features: RunMetaFeatures
    ) -> Tuple[bool, str]:
        """
        Entscheiden ob Run gekillt werden soll.
        
        Args:
            run_features: Meta-Features des laufenden/existierenden Runs
            
        Returns:
            (should_kill, reason)
        """
        # Thresholds für Budget-Klasse holen
        budget_class = run_features.budget_class
        thresholds = self.get_thresholds(budget_class)
        
        # Feature-spezifische Anpassungen
        thresholds = self._apply_feature_adjustments(
            thresholds, run_features.features_active
        )
        
        # Prüfkriterien auswerten
        violations: List[str] = []
        
        # 1. ΔBPB prüfen (wenn verfügbar)
        if run_features.delta_bpb_vs_parent is not None:
            if run_features.delta_bpb_vs_parent > thresholds.max_delta_bpb:
                violations.append(
                    f"ΔBPB {run_features.delta_bpb_vs_parent:.4f} > "
                    f"Limit {thresholds.max_delta_bpb:.4f}"
                )
        
        # 2. Efficiency Gain prüfen (wenn verfügbar)
        if run_features.efficiency_gain_percent is not None:
            if run_features.efficiency_gain_percent < thresholds.min_efficiency_gain:
                violations.append(
                    f"Effizienz {run_features.efficiency_gain_percent:.1f}% < "
                    f"Limit {thresholds.min_efficiency_gain:.1f}%"
                )
        
        # 3. Quant Gap prüfen (wenn quantisiert)
        if run_features.quant_gap is not None:
            if run_features.quant_gap > thresholds.max_quant_gap:
                violations.append(
                    f"Quant-Gap {run_features.quant_gap:.4f} > "
                    f"Limit {thresholds.max_quant_gap:.4f}"
                )
        
        # 4. Step Time prüfen (wenn verfügbar)
        # Hinweis: Hier müsste der Vergleich mit Parent-Run erfolgen
        # Für jetzt: nur wenn step_time_ms extrem hoch
        if run_features.step_time_ms is not None:
            # Heuristik: > 100ms pro Step ist verdächtig
            if run_features.step_time_ms > 100:
                violations.append(
                    f"Step-Time {run_features.step_time_ms:.1f}ms extrem hoch"
                )
        
        # 5. Memory Usage prüfen
        if run_features.memory_usage_mb is not None:
            # Heuristik: > 8GB ist problematisch
            if run_features.memory_usage_mb > 8192:
                violations.append(
                    f"Memory {run_features.memory_usage_mb:.0f}MB > 8GB Limit"
                )
        
        # 6. Training Stability prüfen
        if run_features.training_stability is not None:
            if run_features.training_stability < thresholds.min_training_stability:
                violations.append(
                    f"Stabilität {run_features.training_stability:.2f} < "
                    f"Limit {thresholds.min_training_stability:.2f}"
                )
        
        # Entscheidung
        if violations:
            reason = "Kill-Kriterien verletzt: " + "; ".join(violations)
            self._record_decision(run_features.run_id, True, reason)
            return True, reason
        
        reason = "Alle Kill-Kriterien erfüllt"
        self._record_decision(run_features.run_id, False, reason)
        return False, reason
    
    def _apply_feature_adjustments(
        self,
        base_thresholds: KillThresholds,
        active_features: List[str]
    ) -> KillThresholds:
        """
        Wende feature-spezifische Threshold-Anpassungen an.
        
        Args:
            base_thresholds: Basis-Thresholds
            active_features: Aktive Features im Run
            
        Returns:
            Angepasste KillThresholds
        """
        # Kopie der Basis-Thresholds erstellen (Immutabilität)
        adjusted = KillThresholds(
            context=base_thresholds.context,
            max_delta_bpb=base_thresholds.max_delta_bpb,
            min_efficiency_gain=base_thresholds.min_efficiency_gain,
            max_quant_gap=base_thresholds.max_quant_gap,
            max_step_time_increase=base_thresholds.max_step_time_increase,
            max_memory_increase=base_thresholds.max_memory_increase,
            min_training_stability=base_thresholds.min_training_stability,
        )
        
        # Feature-spezifische Anpassungen
        for feature in active_features:
            if feature in self.FEATURE_ADJUSTMENTS:
                adjustments = self.FEATURE_ADJUSTMENTS[feature]
                
                if "max_quant_gap" in adjustments:
                    # Bei Quant-Features: mehr Toleranz
                    adjusted.max_quant_gap = max(
                        adjusted.max_quant_gap,
                        adjustments["max_quant_gap"]
                    )
                
                if "max_step_time_increase" in adjustments:
                    adjusted.max_step_time_increase = max(
                        adjusted.max_step_time_increase,
                        adjustments["max_step_time_increase"]
                    )
                
                if "max_memory_increase" in adjustments:
                    adjusted.max_memory_increase = max(
                        adjusted.max_memory_increase,
                        adjustments["max_memory_increase"]
                    )
        
        return adjusted
    
    def _record_decision(
        self,
        run_id: str,
        killed: bool,
        reason: str
    ) -> None:
        """
        Zeichne Kill-Entscheidung auf.
        
        Args:
            run_id: ID des Runs
            killed: Ob der Run gekillt wurde
            reason: Begründung
        """
        self.recent_decisions.append((run_id, killed, reason))
        
        # Nur letzte 100 Entscheidungen behalten
        if len(self.recent_decisions) > 100:
            self.recent_decisions = self.recent_decisions[-100:]
    
    def adapt_thresholds(self, recent_success_rate: float) -> None:
        """
        Thresholds basierend auf Erfolgsrate anpassen.
        
        Prinzip:
        - Bei hoher Erfolgsrate (> 80%): Thresholds strenger
        - Bei niedriger Erfolgsrate (< 30%): Thresholds lockerer
        - Bei mittlerer Erfolgsrate (30-80%): Keine Änderung
        
        Args:
            recent_success_rate: Erfolgsrate der letzten Runs (0-1)
        """
        self.success_rate_history.append(recent_success_rate)
        
        # Nur letzte 20 Werte behalten
        if len(self.success_rate_history) > 20:
            self.success_rate_history = self.success_rate_history[-20:]
        
        # Durchschnittliche Erfolgsrate berechnen
        avg_success_rate = sum(self.success_rate_history) / len(self.success_rate_history)
        
        # Anpassungsfaktor berechnen
        if avg_success_rate > self._max_success_rate:
            # Zu erfolgreich → Thresholds strengen
            adjustment_factor = 1.0 - self._adaptation_rate
        elif avg_success_rate < self._min_success_rate:
            # Zu viele Failures → Thresholds lockern
            adjustment_factor = 1.0 + self._adaptation_rate
        else:
            # Im guten Bereich → keine Änderung
            return
        
        # Alle Thresholds anpassen
        for budget_class, thresholds in self.thresholds.items():
            self.thresholds[budget_class] = KillThresholds(
                context=thresholds.context,
                max_delta_bpb=thresholds.max_delta_bpb * adjustment_factor,
                min_efficiency_gain=thresholds.min_efficiency_gain * adjustment_factor,
                max_quant_gap=thresholds.max_quant_gap * adjustment_factor,
                max_step_time_increase=thresholds.max_step_time_increase * adjustment_factor,
                max_memory_increase=thresholds.max_memory_increase * adjustment_factor,
                min_training_stability=thresholds.min_training_stability,  # Nicht anpassen
            )
    
    def get_feature_specific_thresholds(self, feature: str) -> KillThresholds:
        """
        Feature-spezifische Thresholds zurückgeben.
        
        Manche Features (z.B. Quantization) benötigen spezielle Grenzen.
        
        Args:
            feature: Feature-Name
            
        Returns:
            Feature-spezifische KillThresholds oder Default
        """
        if feature in self.feature_thresholds:
            return self.feature_thresholds[feature]
        
        # Default für Feature-Typen
        if "quant" in feature.lower():
            # Quant-Features: mehr Toleranz für Quant-Gap
            return KillThresholds(
                context=f"feature_{feature}",
                max_delta_bpb=0.05,
                min_efficiency_gain=-5.0,
                max_quant_gap=0.05,  # Mehr Toleranz
                max_step_time_increase=20.0,
                max_memory_increase=30.0,
                min_training_stability=0.6,
            )
        
        if "flash" in feature.lower() or "speed" in feature.lower():
            # Speed-Features: strenge Step-Time Limits
            return KillThresholds(
                context=f"feature_{feature}",
                max_delta_bpb=0.05,
                min_efficiency_gain=-5.0,
                max_quant_gap=0.02,
                max_step_time_increase=10.0,  # Streng bei Speed
                max_memory_increase=30.0,
                min_training_stability=0.6,
            )
        
        # Default zurückgeben
        return self.thresholds["medium_budget"]
    
    def set_feature_threshold(
        self,
        feature: str,
        thresholds: KillThresholds
    ) -> None:
        """
        Setze feature-spezifische Thresholds.
        
        Args:
            feature: Feature-Name
            thresholds: KillThresholds für dieses Feature
        """
        self.feature_thresholds[feature] = thresholds
    
    def get_kill_statistics(self) -> Dict[str, Any]:
        """
        Statistik über Kill-Entscheidungen.
        
        Returns:
            Dictionary mit Statistiken
        """
        if not self.recent_decisions:
            return {
                "total_decisions": 0,
                "kills": 0,
                "kill_rate": 0.0,
            }
        
        kills = sum(1 for _, killed, _ in self.recent_decisions if killed)
        total = len(self.recent_decisions)
        
        # Gründe analysieren
        reason_counts: Dict[str, int] = {}
        for _, _, reason in self.recent_decisions:
            # Ersten Teil der Reason extrahieren
            reason_key = reason.split(":")[0] if ":" in reason else reason[:50]
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
        
        return {
            "total_decisions": total,
            "kills": kills,
            "survived": total - kills,
            "kill_rate": kills / total if total > 0 else 0.0,
            "recent_success_rate": (
                sum(self.success_rate_history) / len(self.success_rate_history)
                if self.success_rate_history else 0.0
            ),
            "top_kill_reasons": sorted(
                reason_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "current_thresholds": {
                name: thresh.to_dict()
                for name, thresh in self.thresholds.items()
            },
        }
    
    def reset(self) -> None:
        """Setze alle Thresholds auf Default-Werte zurück."""
        self.__init__()  # type: ignore
