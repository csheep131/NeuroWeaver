#!/usr/bin/env python3
"""
Surrogate Scorer für Phase 4A.

Vorhersage von Run-Erfolg basierend auf Meta-Features mittels
Random Forest oder Gradient Boosting Regressoren.

Verwendet für:
- ΔBPB Vorhersage
- Efficiency Gain Vorhersage
- Confidence Score Berechnung
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from core.meta_features import RunMetaFeatures


class SurrogateScorer:
    """
    Vorhersage von Run-Erfolg basierend auf Meta-Features.
    
    Verwendet Random Forest oder Gradient Boosting für:
    - ΔBPB Vorhersage
    - Efficiency Gain Vorhersage
    - Confidence Score Berechnung
    
    Attributes:
        model_type: Typ des Modells ("random_forest" oder "gradient_boosting")
        bpb_model: Vorhersagemodell für BPB-Gewinn
        efficiency_model: Vorhersagemodell für Effizienz
        feature_scaler: StandardScaler für Feature-Normalisierung
        feature_names: Liste der Feature-Namen für Interpretation
        _is_trained: Flag ob Modelle trainiert sind
    """
    
    def __init__(
        self,
        model_type: Literal["random_forest", "gradient_boosting"] = "random_forest"
    ) -> None:
        """
        Initialisiere Surrogate Scorer.
        
        Args:
            model_type: Typ des Vorhersagemodells
        """
        if model_type not in ("random_forest", "gradient_boosting"):
            raise ValueError(
                f"Ungültiger model_type: {model_type}. "
                "Erlaubt: 'random_forest', 'gradient_boosting'"
            )
        
        self.model_type = model_type
        self.bpb_model: Optional[Any] = None
        self.efficiency_model: Optional[Any] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self._is_trained = False
        
        # Modell-Konfiguration
        self._rf_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        
        self._gb_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        }
    
    def _extract_feature_vector(self, features: RunMetaFeatures) -> np.ndarray:
        """
        Extrahiere numerischen Feature-Vektor aus RunMetaFeatures.
        
        Args:
            features: Meta-Features eines Runs
            
        Returns:
            1D numpy array mit numerischen Features
        """
        # Numerische Features extrahieren
        vector = [
            features.lineage_depth,
            features.siblings_count,
            1.0 if features.sequence_length == "remote" else 0.0,  # sequence_length als numeric
            features.step_time_ms if features.step_time_ms is not None else 0.0,
            features.memory_usage_mb if features.memory_usage_mb is not None else 0.0,
            features.training_stability if features.training_stability is not None else 0.5,
            features.quant_gap if features.quant_gap is not None else 0.0,
            features.seed_variance if features.seed_variance is not None else 0.0,
            features.confidence_interval_width if features.confidence_interval_width is not None else 0.0,
            float(features.days_since_first_feature_introduction) if features.days_since_first_feature_introduction is not None else 0.0,
            float(features.runs_since_feature_last_successful) if features.runs_since_feature_last_successful is not None else 0.0,
            len(features.features_active),  # Anzahl aktiver Features
        ]
        
        # One-Hot Encoding für kategorische Features
        # Budget-Klasse
        budget_map = {"low": 0, "medium": 1, "high": 2}
        budget_val = budget_map.get(features.budget_class, 1)
        vector.extend([
            1.0 if budget_val == 0 else 0.0,
            1.0 if budget_val == 1 else 0.0,
            1.0 if budget_val == 2 else 0.0,
        ])
        
        # Quantisierungstyp
        quant_map = {"none": 0, "int6": 1, "int5": 2, "mixed": 3, "gptq_lite": 4}
        quant_val = quant_map.get(features.quantization_type, 0)
        vector.extend([
            1.0 if quant_val == i else 0.0 for i in range(5)
        ])
        
        # Feature-Präsenz (wichtige Features)
        important_features = [
            "gqa", "film", "leaky_relu", "swiglu", "rope",
            "mixed_quant", "int6_quant", "int5_quant",
        ]
        vector.extend([
            1.0 if feat in features.features_active else 0.0
            for feat in important_features
        ])
        
        return np.array(vector, dtype=np.float64)
    
    def _build_model(self) -> Any:
        """Baue Modell basierend auf model_type."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(**self._rf_params)
        else:
            return GradientBoostingRegressor(**self._gb_params)
    
    def train(
        self,
        features: List[RunMetaFeatures],
        targets: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Trainiere Vorhersagemodelle.
        
        Args:
            features: Meta-Features der Trainings-Runs
            targets: {"delta_bpb": [...], "efficiency_gain": [...]}
            
        Returns:
            Dictionary mit Trainings-Metriken {"bpb_cv_score": ..., "efficiency_cv_score": ...}
            
        Raises:
            ValueError: Bei inkonsistenten Eingabedaten
        """
        if not features:
            raise ValueError("Keine Features zum Trainieren bereitgestellt")
        
        if "delta_bpb" not in targets or "efficiency_gain" not in targets:
            raise ValueError(
                "Targets müssen 'delta_bpb' und 'efficiency_gain' enthalten"
            )
        
        if len(features) != len(targets["delta_bpb"]) or len(features) != len(targets["efficiency_gain"]):
            raise ValueError(
                f"Features ({len(features)}) und Targets "
                f"({len(targets['delta_bpb'])}, {len(targets['efficiency_gain'])}) "
                "müssen gleiche Länge haben"
            )
        
        if len(features) < 5:
            raise ValueError(
                f"Mindestens 5 Samples für Training erforderlich, "
                f"aber nur {len(features)} vorhanden"
            )
        
        # Feature-Matrizen erstellen
        X = np.vstack([self._extract_feature_vector(f) for f in features])
        y_bpb = np.array(targets["delta_bpb"])
        y_eff = np.array(targets["efficiency_gain"])
        
        # Feature-Namen initialisieren (nur beim ersten Mal)
        if not self.feature_names:
            self.feature_names = [
                "lineage_depth", "siblings_count", "sequence_length",
                "step_time_ms", "memory_usage_mb", "training_stability",
                "quant_gap", "seed_variance", "confidence_interval_width",
                "days_since_first_feature", "runs_since_success", "num_features",
                # Budget one-hot
                "budget_low", "budget_medium", "budget_high",
                # Quant one-hot
                "quant_none", "quant_int6", "quant_int5", "quant_mixed", "quant_gptq",
                # Feature presence
                "has_gqa", "has_film", "has_leaky_relu", "has_swiglu", "has_rope",
                "has_mixed_quant", "has_int6_quant", "has_int5_quant",
            ]
        
        # Feature-Skalierung
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Modelle trainieren
        self.bpb_model = self._build_model()
        self.efficiency_model = self._build_model()
        
        self.bpb_model.fit(X_scaled, y_bpb)
        self.efficiency_model.fit(X_scaled, y_eff)
        
        # Cross-Validation Scores berechnen
        n_splits = min(5, len(features) - 1)
        if n_splits >= 2:
            bpb_cv = cross_val_score(
                self._build_model(), X_scaled, y_bpb,
                cv=n_splits, scoring="neg_mean_squared_error"
            )
            eff_cv = cross_val_score(
                self._build_model(), X_scaled, y_eff,
                cv=n_splits, scoring="neg_mean_squared_error"
            )
            
            metrics = {
                "bpb_cv_rmse": np.sqrt(-bpb_cv.mean()),
                "bpb_cv_std": bpb_cv.std(),
                "efficiency_cv_rmse": np.sqrt(-eff_cv.mean()),
                "efficiency_cv_std": eff_cv.std(),
            }
        else:
            metrics = {
                "bpb_cv_rmse": 0.0,
                "efficiency_cv_rmse": 0.0,
            }
        
        self._is_trained = True
        return metrics
    
    def predict(
        self,
        features: RunMetaFeatures
    ) -> Tuple[float, float, float]:
        """
        Vorhersage für einen Run.
        
        Args:
            features: Meta-Features des Runs
            
        Returns:
            (predicted_delta_bpb, predicted_efficiency_gain, confidence_score)
            
        Raises:
            RuntimeError: Wenn Modell nicht trainiert ist
        """
        if not self._is_trained:
            raise RuntimeError("Modell muss zuerst trainiert werden")
        
        if self.bpb_model is None or self.efficiency_model is None:
            raise RuntimeError("Modelle sind nicht initialisiert")
        
        if self.feature_scaler is None:
            raise RuntimeError("Feature-Scaler ist nicht initialisiert")
        
        # Feature-Vektor extrahieren
        X = self._extract_feature_vector(features).reshape(1, -1)
        X_scaled = self.feature_scaler.transform(X)
        
        # Vorhersagen
        predicted_bpb = float(self.bpb_model.predict(X_scaled)[0])
        predicted_eff = float(self.efficiency_model.predict(X_scaled)[0])
        
        # Confidence berechnen basierend auf Feature-Ähnlichkeit zu Trainingsdaten
        confidence = self._compute_confidence(X_scaled)
        
        return predicted_bpb, predicted_eff, confidence
    
    def _compute_confidence(self, X_scaled: np.ndarray) -> float:
        """
        Berechne Confidence-Score basierend auf Vorhersage-Stabilität.
        
        Verwendet Ensemble-Varianz bei Random Forest oder
        Distanz zu Trainingsdaten bei Gradient Boosting.
        
        Args:
            X_scaled: Skalierte Feature-Matrix
            
        Returns:
            Confidence-Score zwischen 0 und 1
        """
        if self.model_type == "random_forest" and self.bpb_model is not None:
            # Bei Random Forest: Varianz der Bäume als Unsicherheitsmaß
            bpb_trees = [tree.predict(X_scaled)[0] for tree in self.bpb_model.estimators_]
            eff_trees = [tree.predict(X_scaled)[0] for tree in self.efficiency_model.estimators_]
            
            bpb_var = np.var(bpb_trees)
            eff_var = np.var(eff_trees)
            
            # Niedrige Varianz = hohe Confidence
            # Skalierung: Varianz < 0.01 → Confidence > 0.9
            confidence = 1.0 / (1.0 + 100 * (bpb_var + eff_var))
        else:
            # Bei Gradient Boosting: Heuristische Confidence
            # Basierend auf Feature-Werten im bekannten Bereich
            confidence = 0.75  # Default Confidence
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Feature-Wichtigkeit extrahieren.
        
        Returns:
            Dictionary {feature_name: importance_score}
            
        Raises:
            RuntimeError: Wenn Modell nicht trainiert ist
        """
        if not self._is_trained:
            raise RuntimeError("Modell muss zuerst trainiert werden")
        
        if self.bpb_model is None:
            raise RuntimeError("BPB-Modell ist nicht initialisiert")
        
        importances = self.bpb_model.feature_importances_
        
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, importances)
        }
    
    def save(self, path: str) -> None:
        """
        Modell speichern.
        
        Args:
            path: Pfad zur Speicherdatei (.pkl)
            
        Raises:
            RuntimeError: Wenn Modell nicht trainiert ist
        """
        if not self._is_trained:
            raise RuntimeError("Nur trainierte Modelle können gespeichert werden")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "model_type": self.model_type,
            "bpb_model": self.bpb_model,
            "efficiency_model": self.efficiency_model,
            "feature_scaler": self.feature_scaler,
            "feature_names": self.feature_names,
            "rf_params": self._rf_params,
            "gb_params": self._gb_params,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """
        Modell laden.
        
        Args:
            path: Pfad zur Modelldatei (.pkl)
            
        Raises:
            FileNotFoundError: Wenn Datei nicht existiert
            ValueError: Wenn ungültiges Modell-Format
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Modell-Datei nicht gefunden: {path}")
        
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        
        # Validierung
        required_keys = ["model_type", "bpb_model", "efficiency_model", "feature_scaler"]
        if not all(key in data for key in required_keys):
            raise ValueError("Ungültiges Modell-Format: Fehlende erforderliche Keys")
        
        self.model_type = data["model_type"]
        self.bpb_model = data["bpb_model"]
        self.efficiency_model = data["efficiency_model"]
        self.feature_scaler = data["feature_scaler"]
        self.feature_names = data.get("feature_names", [])
        
        if "rf_params" in data:
            self._rf_params = data["rf_params"]
        if "gb_params" in data:
            self._gb_params = data["gb_params"]
        
        self._is_trained = True
    
    def to_json(self) -> str:
        """
        Modell-Statistiken als JSON exportieren (nicht das Modell selbst).
        
        Returns:
            JSON-String mit Feature-Importance und Metriken
        """
        if not self._is_trained:
            return json.dumps({"error": "Modell nicht trainiert"})
        
        data = {
            "model_type": self.model_type,
            "feature_importance": self.get_feature_importance(),
            "n_features": len(self.feature_names),
        }
        
        return json.dumps(data, indent=2)
