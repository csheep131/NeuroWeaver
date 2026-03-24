#!/usr/bin/env python3
"""
Drift Monitor für NeuroWeave Phase 4B.

Frühwarnung bei Performance-Drift und Umwelt-Änderungen.

Methoden:
- CUSUM (Cumulative Sum) für sequentielle Drift-Erkennung
- ADWIN (Adaptive Windowing) für Concept Drift
- Umwelt-Änderungen (Dataset, Hardware, Dependencies)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from core.registry import RunEntry, RunRegistry


@dataclass
class DriftReport:
    """
    Bericht über Performance-Drift.

    Attributes:
        drift_type: Typ des Drifts
        severity: Schweregrad
        affected_features: Betroffene Features
        drift_magnitude: % Änderung
        statistical_significance: p-value
        detected_at: Zeitpunkt der Erkennung
        recommended_action: Empfohlene Maßnahme
    """

    drift_type: Literal["performance_drift", "environment_drift", "concept_drift"]
    severity: Literal["low", "medium", "high"]
    affected_features: List[str]
    drift_magnitude: float
    statistical_significance: float
    detected_at: str
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung."""
        return {
            "drift_type": self.drift_type,
            "severity": self.severity,
            "affected_features": self.affected_features,
            "drift_magnitude": self.drift_magnitude,
            "statistical_significance": self.statistical_significance,
            "detected_at": self.detected_at,
            "recommended_action": self.recommended_action,
        }


class DriftMonitor:
    """
    Erkennt Drift in Performance und Umwelt.

    Verwendet statistische Methoden für Frühwarnung:
    - CUSUM für kleine, persistente Änderungen
    - ADWIN für Concept Drift
    - Umwelt-Monitoring für externe Änderungen

    Example:
        monitor = DriftMonitor(window_size=20, threshold=0.05)
        
        # Performance Drift prüfen
        report = monitor.detect_performance_drift("leaky_relu", run_history)
        if report:
            print(f"Drift erkannt: {report.drift_magnitude:.2%}")
        
        # Environment Drift prüfen
        env_report = monitor.detect_environment_drift(registry)
    """

    def __init__(self, window_size: int = 20, threshold: float = 0.05):
        """
        Initialisiere DriftMonitor.

        Args:
            window_size: Fenstergröße für Drift-Erkennung
            threshold: Schwelle für Drift-Erkennung
        """
        self.window_size = window_size
        self.threshold = threshold

        self._cusum_history: Dict[str, List[float]] = {}
        self._cusum_positive: Dict[str, float] = {}
        self._cusum_negative: Dict[str, float] = {}
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        self._environment_snapshots: List[Dict[str, Any]] = []
        self._active_alerts: List[DriftReport] = []

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
        return variance ** 0.5

    def _cusum_test(
        self,
        values: List[float],
        target: float = 0.0,
        slack: float = 0.5
    ) -> tuple[float, float, bool]:
        """
        CUSUM (Cumulative Sum) Test für sequentielle Drift-Erkennung.

        Erkennt kleine, persistente Änderungen im Mittelwert.

        Args:
            values: Zeitreihe von Werten
            target: Ziel-Mittelwert (baseline)
            slack: Toleranz-Bereich

        Returns:
            (cusum_pos, cusum_neg, drift_detected)
        """
        if not values:
            return 0.0, 0.0, False

        cusum_pos = 0.0
        cusum_neg = 0.0
        drift_detected = False

        for value in values:
            deviation = value - target

            # Positive CUSUM (für Anstieg)
            cusum_pos = max(0, cusum_pos + deviation - slack)

            # Negative CUSUM (für Abfall)
            cusum_neg = max(0, cusum_neg - deviation - slack)

            # Prüfen auf Drift
            if cusum_pos > self.threshold or cusum_neg > self.threshold:
                drift_detected = True

        return cusum_pos, cusum_neg, drift_detected

    def _adwin_test(self, values: List[float]) -> tuple[Optional[int], float]:
        """
        ADWIN (Adaptive Windowing) für Concept Drift.

        Findet den Punkt wo sich die Verteilung ändert.

        Args:
            values: Zeitreihe von Werten

        Returns:
            (change_point, confidence)
            change_point ist None wenn kein Drift erkannt
        """
        n = len(values)
        if n < 10:
            return None, 0.0

        # Vereinfachte ADWIN-Implementierung
        # In Produktion: vollständigen ADWIN-Algorithmus verwenden

        best_change_point = None
        best_confidence = 0.0

        # Teste mögliche Change Points
        for split in range(n // 3, 2 * n // 3):
            left = values[:split]
            right = values[split:]

            left_mean = self._compute_mean(left)
            right_mean = self._compute_mean(right)
            left_std = self._compute_std(left)
            right_std = self._compute_std(right)

            # Effektgröße berechnen (Cohen's d)
            pooled_std = ((left_std ** 2 + right_std ** 2) / 2) ** 0.5
            if pooled_std > 0:
                effect_size = abs(right_mean - left_mean) / pooled_std
            else:
                effect_size = 0.0

            # Statistische Signifikanz (vereinfacht)
            n1, n2 = len(left), len(right)
            se = pooled_std * ((1 / n1 + 1 / n2) ** 0.5) if pooled_std > 0 else 1.0
            t_stat = abs(right_mean - left_mean) / se if se > 0 else 0.0

            confidence = min(1.0, t_stat / 3.0)  # Vereinfachte p-value Approximation

            if confidence > best_confidence and confidence > 0.5:
                best_confidence = confidence
                best_change_point = split

        if best_confidence > 0.5:
            return best_change_point, best_confidence
        else:
            return None, 0.0

    def detect_performance_drift(
        self,
        feature: str,
        run_history: List[Dict[str, Any]]
    ) -> Optional[DriftReport]:
        """
        Drift in Feature-Performance erkennen.

        Verwendet CUSUM für sequentielle Drift-Erkennung.

        Args:
            feature: Feature-Name
            run_history: [{"run_id": ..., "delta_bpb": ..., "timestamp": ...}]

        Returns:
            DriftReport wenn Drift erkannt, sonst None
        """
        if not run_history or len(run_history) < 5:
            return None

        # Extrahiere delta_bpb Werte (chronologisch)
        values = []
        for run in sorted(run_history, key=lambda r: r.get("timestamp", "")):
            if feature in run.get("features", []) and run.get("delta_bpb") is not None:
                values.append(run["delta_bpb"])

        if len(values) < 5:
            return None

        # Baseline berechnen (erste Hälfte der Daten)
        baseline_end = len(values) // 2
        baseline = values[:baseline_end]
        baseline_mean = self._compute_mean(baseline)
        baseline_std = self._compute_std(baseline)

        # CUSUM Test auf gesamter Zeitreihe
        cusum_pos, cusum_neg, drift_detected = self._cusum_test(
            values, target=baseline_mean, slack=0.1
        )

        if not drift_detected:
            return None

        # Drift-Magnitude berechnen
        recent = values[-baseline_end:] if len(values) >= 2 * baseline_end else values[baseline_end:]
        recent_mean = self._compute_mean(recent)

        if abs(baseline_mean) > 1e-10:
            drift_magnitude = (recent_mean - baseline_mean) / abs(baseline_mean)
        else:
            drift_magnitude = 0.0

        # Statistische Signifikanz (t-Test vereinfacht)
        n1, n2 = len(baseline), len(recent)
        if n1 > 1 and n2 > 1:
            se = ((baseline_std ** 2 / n1) + (recent_std ** 2 / n2)) ** 0.5 if (n1 * n2) > 0 else 1.0
            t_stat = abs(recent_mean - baseline_mean) / se if se > 0 else 0.0
            p_value = 2 * (1 - min(0.999, t_stat / 5.0))  # Vereinfachte Approximation
        else:
            p_value = 1.0

        # Schweregrad bestimmen
        abs_drift = abs(drift_magnitude)
        if abs_drift > 0.30 or p_value < 0.01:
            severity: Literal["low", "medium", "high"] = "high"
        elif abs_drift > 0.15 or p_value < 0.05:
            severity = "medium"
        else:
            severity = "low"

        # Richtung bestimmen
        direction = "verschlechtert" if drift_magnitude > 0 else "verbessert"

        description = (
            f"Performance-Drift erkannt: {feature} {direction} sich "
            f"(Δ={drift_magnitude:.1%}, p={p_value:.3f})"
        )

        # Empfohlene Aktion
        if drift_magnitude > 0:
            recommended_action = (
                f"Feature '{feature}'' überprüfen, Learning Rate reduzieren, "
                "Feature-Interaktionen analysieren"
            )
        else:
            recommended_action = (
                f"Feature '{feature}' zeigt positive Entwicklung, "
                "weitere Beobachtung empfohlen"
            )

        detected_at = datetime.now().isoformat()

        report = DriftReport(
            drift_type="performance_drift",
            severity=severity,
            affected_features=[feature],
            drift_magnitude=drift_magnitude,
            statistical_significance=p_value,
            detected_at=detected_at,
            recommended_action=recommended_action,
        )

        self._active_alerts.append(report)
        return report

    def detect_environment_drift(self, registry: RunRegistry) -> Optional[DriftReport]:
        """
        Umwelt-Änderungen erkennen.

        Indicators:
        - Dataset-Updates (checksum change)
        - Hardware-Änderungen (GPU type, memory)
        - Dependency-Updates (torch version, etc.)

        Args:
            registry: RunRegistry

        Returns:
            DriftReport wenn Umwelt-Änderung erkannt, sonst None
        """
        current_env = self._capture_environment_snapshot(registry)

        if not self._environment_snapshots:
            self._environment_snapshots.append(current_env)
            return None

        # Vergleiche mit letztem Snapshot
        last_snapshot = self._environment_snapshots[-1]

        changes = []
        severity_score = 0.0

        # Dataset-Änderungen
        if current_env.get("dataset_checksum") != last_snapshot.get("dataset_checksum"):
            changes.append("Dataset hat sich geändert")
            severity_score += 0.4

        # Hardware-Änderungen
        if current_env.get("gpu_type") != last_snapshot.get("gpu_type"):
            changes.append(f"GPU-Typ geändert: {last_snapshot.get('gpu_type')} → {current_env.get('gpu_type')}")
            severity_score += 0.3

        if current_env.get("available_memory") != last_snapshot.get("available_memory"):
            changes.append(f"Memory geändert: {last_snapshot.get('available_memory')} → {current_env.get('available_memory')} MB")
            severity_score += 0.2

        # Dependency-Änderungen
        current_deps = current_env.get("dependencies", {})
        last_deps = last_snapshot.get("dependencies", {})

        for dep, version in current_deps.items():
            if dep in last_deps and last_deps[dep] != version:
                changes.append(f"{dep} aktualisiert: {last_deps[dep]} → {version}")
                severity_score += 0.3

        if not changes:
            self._environment_snapshots.append(current_env)
            return None

        # Snapshot speichern
        self._environment_snapshots.append(current_env)

        # Begrenze Snapshot-Historie
        if len(self._environment_snapshots) > 10:
            self._environment_snapshots = self._environment_snapshots[-10:]

        # Schweregrad bestimmen
        if severity_score >= 0.6:
            severity: Literal["low", "medium", "high"] = "high"
        elif severity_score >= 0.3:
            severity = "medium"
        else:
            severity = "low"

        detected_at = datetime.now().isoformat()

        report = DriftReport(
            drift_type="environment_drift",
            severity=severity,
            affected_features=[],  # Umwelt-Drift betrifft alle
            drift_magnitude=severity_score,
            statistical_significance=1.0 - severity_score,
            detected_at=detected_at,
            recommended_action="Umwelt-Änderungen dokumentieren, Baseline-Performance neu kalibrieren",
        )

        self._active_alerts.append(report)
        return report

    def _capture_environment_snapshot(
        self,
        registry: RunRegistry
    ) -> Dict[str, Any]:
        """
        Erfasse aktuellen Umwelt-Status.

        Args:
            registry: RunRegistry

        Returns:
            Snapshot-Dictionary
        """
        snapshot: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
        }

        # Dataset-Checksumme (wenn verfügbar)
        dataset_path = Path("data/train.txt")
        if dataset_path.exists():
            with open(dataset_path, "rb") as f:
                content = f.read()
                snapshot["dataset_checksum"] = hashlib.md5(content).hexdigest()[:16]
        else:
            snapshot["dataset_checksum"] = "unknown"

        # Hardware-Informationen (vereinfacht)
        snapshot["gpu_type"] = "unknown"  # In Produktion: nvidia-smi oder torch.cuda
        snapshot["available_memory"] = 8000  # Default Annahme

        # Dependencies (wenn requirements.txt verfügbar)
        requirements_path = Path("requirements.txt")
        dependencies = {}
        if requirements_path.exists():
            with open(requirements_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "==" in line:
                            pkg, version = line.split("==", 1)
                            dependencies[pkg.strip()] = version.strip()
                        elif ">=" in line:
                            pkg = line.split(">=")[0].strip()
                            dependencies[pkg] = ">=..."
        snapshot["dependencies"] = dependencies

        return snapshot

    def detect_concept_drift(
        self,
        feature: str,
        run_history: Optional[List[Dict[str, Any]]] = None,
        window_size: Optional[int] = None
    ) -> Optional[DriftReport]:
        """
        Concept Drift: Erfolgreiche Patterns werden weniger erfolgreich.

        Vergleiche Feature-Performance in letztem Fenster vs. Historie.

        Args:
            feature: Feature-Name
            run_history: Run-Historie (wird aus Registry geladen wenn None)
            window_size: Fenstergröße (default: self.window_size)

        Returns:
            DriftReport wenn Concept Drift erkannt, sonst None
        """
        if window_size is None:
            window_size = self.window_size

        if run_history is None:
            # Run-Historie aus Registry laden (müsste übergeben werden)
            return None

        if len(run_history) < window_size * 2:
            return None

        # Extrahiere Erfolgsraten über Zeit
        success_rates = []
        for i in range(0, len(run_history), window_size // 2):
            window = run_history[i : i + window_size]
            if not window:
                continue

            # Erfolgsrate berechnen
            successes = sum(1 for r in window if r.get("delta_bpb", 0) < 0)
            rate = successes / len(window) if window else 0.0
            success_rates.append(rate)

        if len(success_rates) < 3:
            return None

        # ADWIN Test auf Success Rates
        change_point, confidence = self._adwin_test(success_rates)

        if change_point is None or confidence < 0.5:
            return None

        # Drift-Magnitude berechnen
        before = success_rates[:change_point]
        after = success_rates[change_point:]

        before_rate = self._compute_mean(before)
        after_rate = self._compute_mean(after)

        drift_magnitude = after_rate - before_rate

        # Schweregrad
        abs_drift = abs(drift_magnitude)
        if abs_drift > 0.20 or confidence > 0.8:
            severity: Literal["low", "medium", "high"] = "high"
        elif abs_drift > 0.10 or confidence > 0.6:
            severity = "medium"
        else:
            severity = "low"

        direction = "sinkt" if drift_magnitude < 0 else "steigt"
        detected_at = datetime.now().isoformat()

        report = DriftReport(
            drift_type="concept_drift",
            severity=severity,
            affected_features=[feature],
            drift_magnitude=drift_magnitude,
            statistical_significance=confidence,
            detected_at=detected_at,
            recommended_action=(
                f"Feature '{feature}' Erfolgsquote {direction} sich. "
                "Feature-Interaktionen überprüfen, Kontext-Analyse durchführen"
            ),
        )

        self._active_alerts.append(report)
        return report

    def get_drift_alerts(
        self,
        active_only: bool = True,
        limit: int = 10
    ) -> List[DriftReport]:
        """
        Alle aktiven Drift-Alerts.

        Args:
            active_only: Nur aktive Alerts
            limit: Maximale Anzahl zurückgegebener Alerts

        Returns:
            Liste von DriftReports
        """
        alerts = self._active_alerts.copy()

        # Nach Zeitpunkt sortieren (neueste zuerst)
        alerts.sort(key=lambda a: a.detected_at, reverse=True)

        return alerts[:limit]

    def clear_alerts(self) -> None:
        """Alle Alerts löschen."""
        self._active_alerts.clear()

    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Zusammenfassung aller Drift-Aktivitäten.

        Returns:
            Dictionary mit Zusammenfassung
        """
        if not self._active_alerts:
            return {
                "total_alerts": 0,
                "by_type": {},
                "by_severity": {},
                "active_alerts": [],
            }

        # Zähle nach Typ
        type_counts: Dict[str, int] = {}
        for alert in self._active_alerts:
            type_counts[alert.drift_type] = type_counts.get(alert.drift_type, 0) + 1

        # Zähle nach Schweregrad
        severity_counts: Dict[str, int] = {}
        for alert in self._active_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        return {
            "total_alerts": len(self._active_alerts),
            "by_type": type_counts,
            "by_severity": severity_counts,
            "high_severity_count": severity_counts.get("high", 0),
            "active_alerts": [a.to_dict() for a in self._active_alerts[-5:]],  # Letzte 5
        }

    def export_drift_report(self, output_path: str) -> str:
        """
        Exportiere Drift-Report.

        Args:
            output_path: Pfad zur Ausgabedatei

        Returns:
            Pfad zur exportierten Datei
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_drift_summary(),
            "all_alerts": [a.to_dict() for a in self._active_alerts],
            "environment_snapshots": len(self._environment_snapshots),
            "window_size": self.window_size,
            "threshold": self.threshold,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        return str(output_file)

    def reset_baseline(self, feature: Optional[str] = None) -> None:
        """
        Setze Baseline für Drift-Erkennung zurück.

        Args:
            feature: Spezifisches Feature (alle wenn None)
        """
        if feature:
            self._cusum_history.pop(feature, None)
            self._cusum_positive.pop(feature, None)
            self._cusum_negative.pop(feature, None)
            self._baseline_stats.pop(feature, None)
        else:
            self._cusum_history.clear()
            self._cusum_positive.clear()
            self._cusum_negative.clear()
            self._baseline_stats.clear()
