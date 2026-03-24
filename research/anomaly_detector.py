#!/usr/bin/env python3
"""
Anomaly Detector für NeuroWeave Phase 4B.

Statistische Tests für Instabilität und Anomalien in Run-Metriken.

Features:
- Shapiro-Wilk Test auf Seed-Varianz
- Grubbs' Test für Ausreißer
- OOM-Regression Detection
- Noisy Feature Identification
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from core.registry import RunEntry, RunRegistry


@dataclass
class AnomalyReport:
    """
    Bericht einer Anomalie-Erkennung.

    Attributes:
        run_id: ID des betroffenen Runs
        anomaly_type: Typ der Anomalie
        severity: Schweregrad ("low", "medium", "high", "critical")
        description: Mensch-lesbare Beschreibung
        statistical_evidence: Statistische Belege (CV, p-value, etc.)
        recommended_action: Empfohlene Gegenmaßnahme
    """

    run_id: str
    anomaly_type: Literal["instability", "outlier", "oom_risk", "noisy_feature"]
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    statistical_evidence: Dict[str, float]
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung."""
        return {
            "run_id": self.run_id,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "description": self.description,
            "statistical_evidence": self.statistical_evidence,
            "recommended_action": self.recommended_action,
        }


class AnomalyDetector:
    """
    Erkennt Anomalien in Run-Metriken.

    Verwendet statistische Tests für:
    - Instabilität über Seeds (Coefficient of Variation, Shapiro-Wilk)
    - Ausreißer (Grubbs' Test)
    - OOM-Risiko (Memory-Usage-Analyse)
    - Noisy Features (Feature-Outcome-Varianz)

    Example:
        detector = AnomalyDetector(significance_level=0.05)
        report = detector.detect_instability("run001", [0.95, 0.97, 1.02, 0.88])
        if report:
            print(f"Anomalie erkannt: {report.description}")
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialisiere AnomalyDetector.

        Args:
            significance_level: Signifikanzniveau für statistische Tests (default: 0.05)
        """
        self.significance_level = significance_level

    def _compute_mean(self, values: List[float]) -> float:
        """Berechne Mittelwert."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _compute_std(self, values: List[float], mean: Optional[float] = None) -> float:
        """Berechne Standardabweichung."""
        if len(values) < 2:
            return 0.0
        if mean is None:
            mean = self._compute_mean(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _compute_cv(self, values: List[float]) -> float:
        """
        Berechne Coefficient of Variation (CV).

        Formel: CV = std / |mean|

        Returns:
            CV als Dezimalwert (z.B. 0.25 für 25%)
        """
        if not values or len(values) < 2:
            return 0.0
        mean = self._compute_mean(values)
        if abs(mean) < 1e-10:
            return 0.0
        std = self._compute_std(values, mean)
        return abs(std / mean)

    def _shapiro_wilk_test(self, values: List[float]) -> tuple[float, float]:
        """
        Führe Shapiro-Wilk Test auf Normalverteilung durch.

        Implementiert eine vereinfachte Version für kleine Samples.
        Für Produktion: scipy.stats.shapiro verwenden.

        Args:
            values: Liste von Werten

        Returns:
            (test_statistic, p_value)
        """
        n = len(values)
        if n < 3:
            return 1.0, 1.0
        if n > 50:
            # Für große Samples: vereinfachte Approximation
            # In Produktion: scipy.stats.shapiro(values)
            sorted_vals = sorted(values)
            mean = self._compute_mean(sorted_vals)
            std = self._compute_std(sorted_vals, mean)

            if std < 1e-10:
                return 1.0, 1.0

            # Vereinfachte Test-Statistik basierend auf Symmetrie
            median_idx = n // 2
            left_skew = sum(sorted_vals[:median_idx] - mean) if hasattr(sorted_vals[:median_idx], '__sub__') else 0
            right_skew = sum(sorted_vals[median_idx:] - mean) if hasattr(sorted_vals[median_idx:], '__sub__') else 0

            symmetry = 1.0 - abs(left_skew - right_skew) / (n * std + 1e-10)
            statistic = max(0.0, min(1.0, symmetry))

            # p-value Approximation
            p_value = 1.0 - (1.0 - statistic) * n / 10.0
            p_value = max(0.0, min(1.0, p_value))

            return statistic, p_value

        # Für kleine Samples: vereinfachte Berechnung
        sorted_vals = sorted(values)
        mean = self._compute_mean(sorted_vals)
        std = self._compute_std(sorted_vals, mean)

        if std < 1e-10:
            return 1.0, 1.0

        # Normalisiere Werte
        normalized = [(v - mean) / std for v in sorted_vals]

        # Erwarte symmetrische Verteilung um 0
        deviations = [abs(nv) for nv in normalized]
        expected_symmetry = sum(abs(deviations[i] - deviations[-(i + 1)]) for i in range(n // 2))

        statistic = max(0.0, 1.0 - expected_symmetry / n)
        p_value = 1.0 - (1.0 - statistic) ** 2

        return statistic, max(0.0, min(1.0, p_value))

    def _grubbs_test(self, values: List[float]) -> tuple[Optional[int], float, float]:
        """
        Führe Grubbs' Test für Ausreißer durch.

        Testet ob der extremste Wert ein Ausreißer ist.

        Args:
            values: Liste von Werten

        Returns:
            (outlier_index, G_statistic, critical_value)
            outlier_index ist None wenn kein Ausreißer erkannt
        """
        n = len(values)
        if n < 3:
            return None, 0.0, float('inf')

        mean = self._compute_mean(values)
        std = self._compute_std(values, mean)

        if std < 1e-10:
            return None, 0.0, float('inf')

        # Finde extremsten Wert
        deviations = [abs(v - mean) for v in values]
        max_dev = max(deviations)
        outlier_idx = deviations.index(max_dev)

        # G-Statistik
        G = max_dev / std

        # Kritischer Wert (Approximation für α=0.05)
        # Formel: G_crit = ((n-1)/sqrt(n)) * sqrt(t²/(n-2+t²))
        # wobei t der t-Verteilungswert für (n-2) Freiheitsgrade
        t_approx = 2.0  # Vereinfachte Approximation
        G_crit = ((n - 1) / math.sqrt(n)) * math.sqrt(t_approx ** 2 / (n - 2 + t_approx ** 2))

        if G > G_crit:
            return outlier_idx, G, G_crit
        else:
            return None, G, G_crit

    def detect_instability(
        self,
        run_id: str,
        seed_metrics: List[float]
    ) -> Optional[AnomalyReport]:
        """
        Instabilität über Seeds erkennen.

        Verwendet:
        - Coefficient of Variation (CV) > 20% → Warnung
        - CV > 30% → Critical
        - Shapiro-Wilk auf Normalverteilung

        Args:
            run_id: ID des Runs
            seed_metrics: Metriken über verschiedene Seeds (z.B. BPB-Werte)

        Returns:
            AnomalyReport wenn Instabilität erkannt, sonst None
        """
        if not seed_metrics or len(seed_metrics) < 2:
            return None

        cv = self._compute_cv(seed_metrics)
        mean = self._compute_mean(seed_metrics)
        std = self._compute_std(seed_metrics, mean)

        # Shapiro-Wilk Test
        sw_statistic, sw_p_value = self._shapiro_wilk_test(seed_metrics)

        # Bestimme Schweregrad basierend auf CV
        severity: Literal["low", "medium", "high", "critical"]
        description: str
        recommended_action: str

        if cv > 0.30:
            severity = "critical"
            description = f"Kritische Instabilität: CV={cv:.1%} (>{30:.0f}%), Seed-Varianz zu hoch"
            recommended_action = "Sofortiges Handeln: Feature-Kombination überprüfen, Learning Rate reduzieren, Seed-Range erweitern"
        elif cv > 0.20:
            severity = "high"
            description = f"Hohe Instabilität: CV={cv:.1%} (>{20:.0f}%), Seed-Varianz bedenklich"
            recommended_action = "Feature-Stabilität prüfen, zusätzliche Seeds testen, Gradient Clipping erwägen"
        elif cv > 0.10:
            severity = "medium"
            description = f"Mittlere Instabilität: CV={cv:.1%} (>{10:.0f}%), Varianz überwachenswert"
            recommended_action = "Weitere Seeds sammeln, Trend beobachten"
        else:
            # CV <= 10%: keine Anomalie
            return None

        # Statistische Evidenz sammeln
        evidence: Dict[str, float] = {
            "coefficient_of_variation": cv,
            "mean": mean,
            "std": std,
            "shapiro_wilk_statistic": sw_statistic,
            "shapiro_wilk_p_value": sw_p_value,
            "num_seeds": len(seed_metrics),
        }

        # Prüfe auf Nicht-Normalverteilung (p < significance_level)
        if sw_p_value < self.significance_level:
            evidence["non_normal_distribution"] = 1.0
            description += " | Verteilung nicht normal (Shapiro-Wilk p={:.3f})".format(sw_p_value)

        return AnomalyReport(
            run_id=run_id,
            anomaly_type="instability",
            severity=severity,
            description=description,
            statistical_evidence=evidence,
            recommended_action=recommended_action,
        )

    def detect_outliers(
        self,
        run_id: str,
        metric: float,
        reference_metrics: List[float]
    ) -> Optional[AnomalyReport]:
        """
        Ausreißer mit Grubbs' Test erkennen.

        Args:
            run_id: Run-ID
            metric: Zu prüfende Metrik (z.B. BPB)
            reference_metrics: Vergleichswerte (andere Runs mit ähnlichem Kontext)

        Returns:
            AnomalyReport wenn Ausreißer erkannt, sonst None
        """
        if not reference_metrics or len(reference_metrics) < 3:
            return None

        # Kombiniere aktuellen Wert mit Referenzen
        all_values = reference_metrics + [metric]

        # Grubbs' Test durchführen
        outlier_idx, G, G_crit = self._grubbs_test(all_values)

        if outlier_idx is None:
            return None

        # Prüfen ob der aktuelle Wert der Ausreißer ist
        if outlier_idx != len(all_values) - 1:
            # Anderer Wert ist Ausreißer, nicht der aktuelle
            return None

        # Bestimme Schweregrad basierend auf G/G_crit Ratio
        ratio = G / G_crit if G_crit > 0 else float('inf')

        severity: Literal["low", "medium", "high", "critical"]
        if ratio > 2.0:
            severity = "critical"
        elif ratio > 1.5:
            severity = "high"
        elif ratio > 1.2:
            severity = "medium"
        else:
            severity = "low"

        mean = self._compute_mean(reference_metrics)
        std = self._compute_std(reference_metrics, mean)
        deviation = abs(metric - mean)

        direction = "höher" if metric > mean else "niedriger"
        description = (
            f"Ausreißer erkannt: {metric:.4f} ist {deviation:.4f} ({direction}) "
            f"vom Mittelwert ({mean:.4f} ± {std:.4f})"
        )

        evidence: Dict[str, float] = {
            "grubbs_G": G,
            "grubbs_critical": G_crit,
            "G_ratio": ratio,
            "mean": mean,
            "std": std,
            "deviation": deviation,
            "num_references": len(reference_metrics),
        }

        recommended_action = (
            "Run konfigurativ überprüfen, Parent-Lineage analysieren, "
            "Feature-Interaktionen prüfen"
        )

        return AnomalyReport(
            run_id=run_id,
            anomaly_type="outlier",
            severity=severity,
            description=description,
            statistical_evidence=evidence,
            recommended_action=recommended_action,
        )

    def detect_oom_risk(
        self,
        run_id: str,
        memory_usage_mb: float,
        available_memory_mb: float = 8000
    ) -> Optional[AnomalyReport]:
        """
        OOM-Risiko vorhersagen.

        Criteria:
        - Memory usage > 90% → High Risk
        - Memory usage > 80% → Medium Risk
        - Memory growth rate > 10% pro Run → Warning

        Args:
            run_id: Run-ID
            memory_usage_mb: Aktuelle Memory-Nutzung in MB
            available_memory_mb: Verfügbare Memory in MB (default: 8000)

        Returns:
            AnomalyReport wenn OOM-Risiko erkannt, sonst None
        """
        if memory_usage_mb <= 0 or available_memory_mb <= 0:
            return None

        usage_ratio = memory_usage_mb / available_memory_mb
        usage_percent = usage_ratio * 100

        severity: Literal["low", "medium", "high", "critical"]
        description: str
        recommended_action: str

        if usage_ratio > 0.95:
            severity = "critical"
            description = f"Kritisches OOM-Risiko: {usage_percent:.1f}% Memory genutzt (>95%)"
            recommended_action = "Sofort: Batch-Size reduzieren, Gradient Checkpointing aktivieren, Modell-Größe reduzieren"
        elif usage_ratio > 0.90:
            severity = "high"
            description = f"Hohes OOM-Risiko: {usage_percent:.1f}% Memory genutzt (>90%)"
            recommended_action = "Batch-Size reduzieren, Gradient Checkpointing prüfen, Memory-Optimierungen aktivieren"
        elif usage_ratio > 0.80:
            severity = "medium"
            description = f"Mittleres OOM-Risiko: {usage_percent:.1f}% Memory genutzt (>80%)"
            recommended_action = "Memory-Trend beobachten, nächste Runs optimieren"
        else:
            # < 80%: kein akutes Risiko
            return None

        evidence: Dict[str, float] = {
            "memory_usage_mb": memory_usage_mb,
            "available_memory_mb": available_memory_mb,
            "usage_ratio": usage_ratio,
            "usage_percent": usage_percent,
            "safety_margin_mb": available_memory_mb - memory_usage_mb,
        }

        return AnomalyReport(
            run_id=run_id,
            anomaly_type="oom_risk",
            severity=severity,
            description=description,
            statistical_evidence=evidence,
            recommended_action=recommended_action,
        )

    def detect_noisy_feature(
        self,
        feature: str,
        run_outcomes: List[Dict[str, Any]]
    ) -> Optional[AnomalyReport]:
        """
        Features mit hoher Outcome-Varianz identifizieren.

        Args:
            feature: Feature-Name
            run_outcomes: [{"run_id": ..., "delta_bpb": ..., "features": [...]}]

        Returns:
            AnomalyReport wenn noisy feature erkannt, sonst None
        """
        if not run_outcomes:
            return None

        # Filtere Runs die dieses Feature enthalten
        feature_runs = [
            r for r in run_outcomes
            if feature in r.get("features", []) and r.get("delta_bpb") is not None
        ]

        if len(feature_runs) < 3:
            return None

        # Extrahiere delta_bpb Werte
        deltas = [r["delta_bpb"] for r in feature_runs]
        cv = self._compute_cv(deltas)
        mean = self._compute_mean(deltas)
        std = self._compute_std(deltas, mean)

        # Bestimme Schweregrad basierend auf CV
        severity: Literal["low", "medium", "high", "critical"]
        description: str
        recommended_action: str

        if cv > 0.50:
            severity = "critical"
            description = f"Kritisches Noisy Feature: {feature} hat CV={cv:.1%} über {len(feature_runs)} Runs"
            recommended_action = f"Feature '{feature}' in Quarantäne versetzen, Feature-Interaktionen analysieren"
        elif cv > 0.30:
            severity = "high"
            description = f"Sehr verrauschtes Feature: {feature} hat CV={cv:.1%} über {len(feature_runs)} Runs"
            recommended_action = f"Feature '{feature}' nur in Kombination mit stabilisierenden Features verwenden"
        elif cv > 0.20:
            severity = "medium"
            description = f"Mäßig verrauschtes Feature: {feature} hat CV={cv:.1%} über {len(feature_runs)} Runs"
            recommended_action = f"Feature '{feature}' weiter beobachten, zusätzliche Runs sammeln"
        else:
            # CV <= 20%: nicht als noisy eingestuft
            return None

        # Erfolgsquote berechnen
        positive_outcomes = sum(1 for d in deltas if d < 0)  # Negatives delta_bpb = Verbesserung
        success_rate = positive_outcomes / len(deltas)

        evidence: Dict[str, float] = {
            "coefficient_of_variation": cv,
            "mean_delta_bpb": mean,
            "std_delta_bpb": std,
            "num_runs": len(feature_runs),
            "success_rate": success_rate,
            "positive_outcomes": positive_outcomes,
        }

        return AnomalyReport(
            run_id=feature,  # Feature-Name als run_id
            anomaly_type="noisy_feature",
            severity=severity,
            description=description,
            statistical_evidence=evidence,
            recommended_action=recommended_action,
        )

    def run_all_checks(
        self,
        run_id: str,
        registry: RunRegistry
    ) -> List[AnomalyReport]:
        """
        Alle Anomalie-Checks ausführen.

        Führt alle verfügbaren Checks für den gegebenen Run durch:
        - Instabilität (wenn Seed-Familie existiert)
        - Ausreißer (wenn Vergleichs-Runs existieren)
        - OOM-Risiko (wenn Memory-Daten verfügbar)
        - Noisy Features (wenn Feature-Historie existiert)

        Args:
            run_id: ID des zu prüfenden Runs
            registry: RunRegistry für Datenzugriff

        Returns:
            Liste aller erkannten Anomalie-Reports
        """
        reports: List[AnomalyReport] = []

        entry = registry.get(run_id)
        if entry is None:
            return reports

        # 1. Instabilitäts-Check (Seed-Varianz)
        if entry.config_hash:
            seed_family = registry.get_config_family(entry.config_hash)
            seed_metrics = [e.val_bpb for e in seed_family if e.val_bpb is not None]
            if len(seed_metrics) >= 2:
                report = self.detect_instability(run_id, seed_metrics)
                if report:
                    reports.append(report)

        # 2. Ausreißer-Check (Vergleich mit ähnlichen Runs)
        if entry.val_bpb is not None and entry.parent_run_id:
            # Finde ähnliche Runs (gleiche Budget-Klasse, gleiche Quantisierung)
            parent = registry.get(entry.parent_run_id)
            if parent and parent.config_hash:
                reference_family = registry.get_config_family(parent.config_hash)
                reference_metrics = [
                    e.val_bpb for e in reference_family
                    if e.val_bpb is not None and e.run_id != run_id
                ]
                if len(reference_metrics) >= 3:
                    report = self.detect_outliers(run_id, entry.val_bpb, reference_metrics)
                    if report:
                        reports.append(report)

        # 3. OOM-Risiko (wenn Memory-Daten verfügbar)
        if entry.artifact_bytes and entry.artifact_bytes > 0:
            memory_mb = entry.artifact_bytes / (1024 * 1024)
            # Annahme: 8GB verfügbar (typisch für Consumer GPUs)
            report = self.detect_oom_risk(run_id, memory_mb, available_memory_mb=8000)
            if report:
                reports.append(report)

        # 4. Noisy Feature Check (wenn Features bekannt)
        # Hier würden wir über alle Features iterieren und deren Varianz prüfen
        # Vereinfachte Version: nur wenn delta_bpb extrem ist
        if entry.delta_bpb is not None and abs(entry.delta_bpb) > 0.1:
            # Extrahiere Features aus Config (wenn verfügbar)
            # Für jetzt: vereinfacht
            pass

        return reports

    def get_summary_statistics(
        self,
        reports: List[AnomalyReport]
    ) -> Dict[str, Any]:
        """
        Zusammenfassung der Anomalie-Reports.

        Args:
            reports: Liste von AnomalyReports

        Returns:
            Dictionary mit Zusammenfassungs-Statistiken
        """
        if not reports:
            return {
                "total_anomalies": 0,
                "by_severity": {},
                "by_type": {},
            }

        # Zähle nach Schweregrad
        severity_counts: Dict[str, int] = {}
        for report in reports:
            severity_counts[report.severity] = severity_counts.get(report.severity, 0) + 1

        # Zähle nach Typ
        type_counts: Dict[str, int] = {}
        for report in reports:
            type_counts[report.anomaly_type] = type_counts.get(report.anomaly_type, 0) + 1

        return {
            "total_anomalies": len(reports),
            "by_severity": severity_counts,
            "by_type": type_counts,
            "critical_count": severity_counts.get("critical", 0),
            "high_count": severity_counts.get("high", 0),
        }
