#!/usr/bin/env python3
"""
Failure Classifier für NeuroWeave Phase 4B.

ML-basierte Fehlerkategorisierung für Run-Fehler.

Features:
- Entscheidungsbaum / Random Forest Klassifikator
- Root-Cause-Analyse
- Similar-Failure-Detection
- Empfohlene Fixes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from core.registry import RunEntry, RunRegistry


@dataclass
class FailureDiagnosis:
    """
    Diagnose eines Fehlers.

    Attributes:
        run_id: ID des fehlgeschlagenen Runs
        failure_category: Kategorie des Fehlers
        confidence: Konfidenz der Klassifikation (0-1)
        root_cause: Beschreibung der Hauptursache
        contributing_factors: Liste von beitragenden Faktoren
        similar_failures: Run-IDs mit ähnlichem Fehler
        recommended_fix: Empfohlene Lösung
    """

    run_id: str
    failure_category: Literal[
        "oom",
        "nan_gradients",
        "training_divergence",
        "quant_explosion",
        "performance_regression",
    ]
    confidence: float
    root_cause: str
    contributing_factors: List[str]
    similar_failures: List[str]
    recommended_fix: str

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung."""
        return {
            "run_id": self.run_id,
            "failure_category": self.failure_category,
            "confidence": self.confidence,
            "root_cause": self.root_cause,
            "contributing_factors": self.contributing_factors,
            "similar_failures": self.similar_failures,
            "recommended_fix": self.recommended_fix,
        }


class FailureClassifier:
    """
    Klassifiziert Run-Fehler mit ML.

    Features für Klassifikation:
    - Model-Parameter (depth, width, mlp_ratio)
    - Activations (type, leakiness)
    - Quantization (type, bit-width)
    - Training (lr, batch_size, gradient_norm)
    - Error-Signature (loss curve shape, gradient stats)

    Example:
        classifier = FailureClassifier()
        diagnosis = classifier.classify("run017", registry)
        if diagnosis:
            print(f"Fehler: {diagnosis.failure_category}")
            print(f"Ursache: {diagnosis.root_cause}")
            print(f"Fix: {diagnosis.recommended_fix}")
    """

    def __init__(self):
        """Initialisiere FailureClassifier."""
        self.model: Optional[Any] = None  # Entscheidungsbaum / Random Forest
        self.is_trained: bool = False

        self.category_definitions: Dict[str, str] = {
            "oom": "Out of Memory - VRAM exceeded",
            "nan_gradients": "NaN/Inf in Gradients",
            "training_divergence": "Loss Explosion nach N Steps",
            "quant_explosion": "Katastrophaler BPB-Verlust nach Quantisierung",
            "performance_regression": "Step-Time > 50% schlechter ohne BPB-Gewinn",
        }

        # Root-Cause-Informationen pro Kategorie
        self._root_cause_info: Dict[str, Dict[str, Any]] = {
            "oom": {
                "common_causes": [
                    "depth > 12",
                    "batch_size > 32",
                    "no gradient checkpointing",
                    "activation type requires extra memory",
                ],
                "prevention": [
                    "reduce depth",
                    "enable gradient checkpointing",
                    "reduce batch_size",
                    "use memory-efficient activations",
                ],
            },
            "nan_gradients": {
                "common_causes": [
                    "learning rate too high",
                    "unstable activation function",
                    "missing gradient clipping",
                    "numerical instability in attention",
                ],
                "prevention": [
                    "reduce learning rate",
                    "add gradient clipping",
                    "use stable activations (gelu, leaky_relu)",
                    "add epsilon to attention scores",
                ],
            },
            "training_divergence": {
                "common_causes": [
                    "learning rate too high",
                    "warmup steps insufficient",
                    "batch size too small",
                    "unstable feature combination",
                ],
                "prevention": [
                    "reduce learning rate",
                    "increase warmup steps",
                    "increase batch size",
                    "remove unstable features",
                ],
            },
            "quant_explosion": {
                "common_causes": [
                    "quantization bit-width too low",
                    "sensitive layers not excluded",
                    "missing calibration data",
                    "aggressive quantization strategy",
                ],
                "prevention": [
                    "use higher bit-width (int6 instead of int5)",
                    "exclude sensitive layers",
                    "use representative calibration data",
                    "apply mixed quantization",
                ],
            },
            "performance_regression": {
                "common_causes": [
                    "inefficient feature implementation",
                    "suboptimal kernel fusion",
                    "increased memory bandwidth usage",
                    "complex attention mechanism",
                ],
                "prevention": [
                    "profile and optimize hot paths",
                    "use fused operations",
                    "reduce memory footprint",
                    "simplify attention mechanism",
                ],
            },
        }

        # Historische Fehler für Similarity-Search
        self._historical_failures: List[Dict[str, Any]] = []

    def _extract_features(self, run_id: str, registry: RunRegistry) -> Dict[str, Any]:
        """
        Extrahiere Features für Klassifikation.

        Args:
            run_id: ID des Runs
            registry: RunRegistry

        Returns:
            Dictionary mit Features
        """
        entry = registry.get(run_id)
        if entry is None:
            return {}

        features: Dict[str, Any] = {
            "run_id": run_id,
            "status": entry.status,
            "val_bpb": entry.val_bpb,
            "ms_per_step": entry.ms_per_step,
            "artifact_bytes": entry.artifact_bytes,
            "delta_bpb": entry.delta_bpb,
            "delta_ms": entry.delta_ms,
            "notes": entry.notes,
        }

        # Lineage-Informationen
        if entry.parent_run_id:
            parent = registry.get(entry.parent_run_id)
            if parent:
                features["parent_val_bpb"] = parent.val_bpb
                features["parent_ms_per_step"] = parent.ms_per_step

        # Seed-Varianz
        if entry.config_hash:
            seed_stats = registry.get_seed_statistics(entry.config_hash)
            features["seed_std"] = seed_stats.get("bpb", {}).get("std", 0.0)
            features["num_seeds"] = seed_stats.get("num_seeds", 0)

        return features

    def _detect_error_signature(self, entry: RunEntry) -> str:
        """
        Erkenne Error-Signatur aus Run-Einträgen.

        Args:
            entry: RunEntry

        Returns:
            Error-Signatur-String
        """
        notes = entry.notes.lower() if entry.notes else ""

        # OOM-Signaturen
        if any(s in notes for s in ["oom", "out of memory", "cuda out of memory", "memory"]):
            return "oom"

        # NaN-Signaturen
        if any(s in notes for s in ["nan", "inf", "infinity", "gradient exploded"]):
            return "nan_gradients"

        # Divergenz-Signaturen
        if any(s in notes for s in ["divergence", "loss exploded", "training unstable"]):
            return "training_divergence"

        # Quantisierungs-Signaturen
        if any(s in notes for s in ["quant", "degradation", "bpb increased"]):
            return "quant_explosion"

        # Performance-Signaturen
        if entry.delta_ms is not None and entry.delta_ms > 0.5:
            return "performance_regression"

        return "unknown"

    def _heuristic_classify(
        self,
        run_id: str,
        registry: RunRegistry
    ) -> Optional[Tuple[str, float]]:
        """
        Heuristische Klassifikation basierend auf Error-Signaturen.

        Args:
            run_id: Run-ID
            registry: RunRegistry

        Returns:
            (category, confidence) oder None
        """
        entry = registry.get(run_id)
        if entry is None:
            return None

        # Status-Check
        if entry.status not in ("failed", "killed"):
            return None

        # Error-Signatur analysieren
        signature = self._detect_error_signature(entry)

        if signature != "unknown":
            # Direkte Signatur-Erkennung
            if signature == "oom":
                return ("oom", 0.9)
            elif signature == "nan_gradients":
                return ("nan_gradients", 0.85)
            elif signature == "training_divergence":
                return ("training_divergence", 0.8)
            elif signature == "quant_explosion":
                return ("quant_explosion", 0.85)
            elif signature == "performance_regression":
                return ("performance_regression", 0.75)

        # Heuristische Analyse basierend auf Metriken
        if entry.delta_bpb is not None:
            # Große BPB-Verschlechterung
            if entry.delta_bpb > 0.5:
                return ("quant_explosion", 0.7)
            elif entry.delta_bpb > 0.2:
                return ("training_divergence", 0.6)

        if entry.delta_ms is not None and entry.delta_ms > 0.5:
            # Step-Time Verschlechterung > 50%
            if entry.delta_bpb is None or entry.delta_bpb >= 0:
                return ("performance_regression", 0.7)

        # Artifact-Größe als OOM-Indikator
        if entry.artifact_bytes and entry.artifact_bytes > 16_000_000:
            return ("oom", 0.5)

        return None

    def train(
        self,
        historical_failures: List[Dict[str, Any]],
        labels: List[str]
    ) -> Dict[str, float]:
        """
        Klassifikator trainieren.

        Args:
            historical_failures: [{"run_id": ..., "features": {...}, "error_signature": {...}}]
            labels: ["oom", "nan_gradients", ...]

        Returns:
            Dictionary mit Trainings-Metriken
        """
        if not historical_failures or not labels:
            return {"accuracy": 0.0, "num_samples": 0}

        if len(historical_failures) != len(labels):
            raise ValueError(
                f"historical_failures ({len(historical_failures)}) und "
                f"labels ({len(labels)}) müssen gleiche Länge haben"
            )

        # Speichere historische Fehler für Similarity-Search
        self._historical_failures = historical_failures.copy()

        # Für jetzt: heuristische Klassifikation verwenden
        # ML-Modell könnte später hinzugefügt werden (sklearn)
        self.is_trained = True

        # Einfache Accuracy-Schätzung durch Selbst-Validierung
        correct = 0
        for failure, true_label in zip(historical_failures, labels):
            # Simuliere Klassifikation
            predicted = self._classify_from_features(failure)
            if predicted == true_label:
                correct += 1

        accuracy = correct / len(labels) if labels else 0.0

        return {
            "accuracy": accuracy,
            "num_samples": len(labels),
            "categories": list(set(labels)),
        }

    def _classify_from_features(
        self,
        features: Dict[str, Any]
    ) -> Optional[str]:
        """
        Klassifikation basierend auf Features.

        Args:
            features: Feature-Dictionary

        Returns:
            Fehlerkategorie oder None
        """
        notes = features.get("notes", "").lower()
        delta_bpb = features.get("delta_bpb")
        delta_ms = features.get("delta_ms")

        # OOM
        if any(s in notes for s in ["oom", "out of memory", "cuda out of memory"]):
            return "oom"

        # NaN Gradients
        if any(s in notes for s in ["nan", "inf", "gradient exploded"]):
            return "nan_gradients"

        # Training Divergence
        if delta_bpb is not None and delta_bpb > 0.5:
            return "quant_explosion"
        elif delta_bpb is not None and delta_bpb > 0.2:
            return "training_divergence"

        # Performance Regression
        if delta_ms is not None and delta_ms > 0.5:
            return "performance_regression"

        return None

    def classify(
        self,
        run_id: str,
        registry: RunRegistry
    ) -> Optional[FailureDiagnosis]:
        """
        Fehler klassifizieren.

        Args:
            run_id: ID des Runs
            registry: RunRegistry

        Returns:
            FailureDiagnosis oder None wenn kein Fehler
        """
        entry = registry.get(run_id)
        if entry is None:
            return None

        # Prüfen ob Fehler vorliegt
        if entry.status not in ("failed", "killed"):
            # Auch completed Runs können Probleme haben (z.B. performance_regression)
            if entry.status == "completed":
                if entry.delta_ms is None or entry.delta_ms <= 0.5:
                    return None
            else:
                return None

        # Heuristische Klassifikation
        classification = self._heuristic_classify(run_id, registry)

        if classification is None:
            # Kein klarer Fehler erkennbar
            return None

        category, confidence = classification

        # Root-Cause-Analyse
        root_cause_info = self.get_root_cause_analysis(category)
        root_cause = root_cause_info.get("common_causes", ["Unbekannt"])[0]

        # Contributing Factors extrahieren
        contributing_factors = self._extract_contributing_factors(run_id, registry, category)

        # Similar Failures finden
        similar_failures = self.find_similar_failures(run_id, top_k=5)

        # Empfohlenen Fix bestimmen
        recommended_fix = root_cause_info.get("prevention", ["Manuell analysieren"])[0]

        return FailureDiagnosis(
            run_id=run_id,
            failure_category=category,
            confidence=confidence,
            root_cause=root_cause,
            contributing_factors=contributing_factors,
            similar_failures=similar_failures,
            recommended_fix=recommended_fix,
        )

    def _extract_contributing_factors(
        self,
        run_id: str,
        registry: RunRegistry,
        category: str
    ) -> List[str]:
        """
        Extrahiere beitragende Faktoren für einen Fehler.

        Args:
            run_id: Run-ID
            registry: RunRegistry
            category: Fehlerkategorie

        Returns:
            Liste von beitragenden Faktoren
        """
        entry = registry.get(run_id)
        if entry is None:
            return []

        factors: List[str] = []

        # Lineage-Analyse
        if entry.parent_run_id:
            parent = registry.get(entry.parent_run_id)
            if parent:
                # Prüfe ob Parent auch Probleme hatte
                if parent.status in ("failed", "killed"):
                    factors.append("Parent-Run ebenfalls fehlgeschlagen")
                elif parent.delta_bpb is not None and parent.delta_bpb > 0.1:
                    factors.append("Parent-Run bereits verschlechtert")

        # Seed-Varianz
        if entry.config_hash:
            seed_stats = registry.get_seed_statistics(entry.config_hash)
            bpb_std = seed_stats.get("bpb", {}).get("std", 0.0)
            if bpb_std > 0.05:
                factors.append(f"Hohe Seed-Varianz (std={bpb_std:.3f})")

        # Kategorie-spezifische Faktoren
        if category == "oom":
            if entry.artifact_bytes and entry.artifact_bytes > 12_000_000:
                factors.append("Große Artifact-Größe (>12MB)")
        elif category == "nan_gradients":
            if entry.delta_bpb is not None and entry.delta_bpb > 1.0:
                factors.append("Extreme BPB-Verschlechterung")
        elif category == "performance_regression":
            if entry.delta_ms is not None and entry.delta_ms > 1.0:
                factors.append("Step-Time >100% schlechter")

        return factors

    def get_root_cause_analysis(self, failure_category: str) -> Dict[str, Any]:
        """
        Root-Cause-Information für Fehlerkategorie.

        Args:
            failure_category: Fehlerkategorie

        Returns:
            Dictionary mit common_causes und prevention
        """
        return self._root_cause_info.get(failure_category, {
            "common_causes": ["Unbekannte Ursache"],
            "prevention": ["Manuelle Analyse erforderlich"],
        })

    def find_similar_failures(
        self,
        run_id: str,
        top_k: int = 5
    ) -> List[str]:
        """
        Ähnliche historische Fehler finden.

        Args:
            run_id: Run-ID
            top_k: Anzahl der Ergebnisse

        Returns:
            Liste von Run-IDs ähnlicher Fehler
        """
        if not self._historical_failures:
            return []

        # Einfache Similarity-Suche basierend auf Kategorie
        # Könnte durch Feature-basierte Similarity erweitert werden
        current_features = self._extract_features_from_historical(run_id)
        if not current_features:
            return []

        current_category = current_features.get("category", "unknown")

        # Finde gleiche Kategorie
        similar = [
            f for f in self._historical_failures
            if f.get("category") == current_category
        ]

        # Nach Similarity sortieren (hier: einfach nach run_id)
        similar_ids = [f.get("run_id", "") for f in similar[:top_k]]

        return similar_ids

    def _extract_features_from_historical(
        self,
        run_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extrahiere Features aus historischen Fehlern.

        Args:
            run_id: Run-ID

        Returns:
            Feature-Dictionary oder None
        """
        for failure in self._historical_failures:
            if failure.get("run_id") == run_id:
                return failure
        return None

    def get_failure_statistics(
        self,
        registry: RunRegistry
    ) -> Dict[str, Any]:
        """
        Fehler-Statistiken berechnen.

        Args:
            registry: RunRegistry

        Returns:
            Dictionary mit Statistiken
        """
        all_runs = registry.list_runs()

        # Zähle nach Status
        status_counts: Dict[str, int] = {}
        for run in all_runs:
            status_counts[run.status] = status_counts.get(run.status, 0) + 1

        # Zähle Fehlerkategorien
        category_counts: Dict[str, int] = {}
        for run in all_runs:
            if run.status in ("failed", "killed"):
                signature = self._detect_error_signature(run)
                if signature != "unknown":
                    category_counts[signature] = category_counts.get(signature, 0) + 1

        total = len(all_runs)
        failed = status_counts.get("failed", 0) + status_counts.get("killed", 0)
        failure_rate = failed / total if total > 0 else 0.0

        return {
            "total_runs": total,
            "failed_runs": failed,
            "failure_rate": failure_rate,
            "by_status": status_counts,
            "by_category": category_counts,
            "most_common_failure": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
        }

    def export_training_data(
        self,
        registry: RunRegistry,
        output_path: str
    ) -> int:
        """
        Exportiere Trainingsdaten aus Registry.

        Args:
            registry: RunRegistry
            output_path: Pfad zur Ausgabedatei

        Returns:
            Anzahl exportierter Samples
        """
        all_runs = registry.list_runs()

        samples = []
        for run in all_runs:
            if run.status in ("failed", "killed", "completed"):
                features = self._extract_features(run.run_id, registry)
                category = self._detect_error_signature(run)

                if category != "unknown" or run.status == "completed":
                    samples.append({
                        "run_id": run.run_id,
                        "features": features,
                        "category": category if run.status in ("failed", "killed") else "success",
                    })

        # Speichern
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(samples, f, indent=2)

        return len(samples)
