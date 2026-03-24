"""
Meta-Feature Extraktion für NeuroWeave Runs.

Dieses Modul extrahiert strukturierte Meta-Daten aus bestehenden Runs
für die autonome Selbstverbesserung (Phase 4).

Features:
- Feature-Vektor (aktivierte Features)
- Lineage & History (Parent, Depth, Siblings)
- Context (Budget-Klasse, Sequence Length, Quantisierung)
- Performance Characteristics (Step-Time, Memory, Stabilität)
- Outcomes (ΔBPB, Effizienz-Gewinn, Modell-Größe)
- Statistical Properties (Seed-Varianz, Konfidenz-Intervalle)
- Temporal Features (Zeit seit Feature-Einführung)
- Interaction Features (Co-occurrence, Erfolgsquote)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from .config import Config, load_config
from .registry import RunEntry, RunRegistry


@dataclass
class RunMetaFeatures:
    """
    Meta-Features für einen einzelnen Run.
    
    Attributes:
        run_id: Eindeutige ID des Runs
        features_active: Liste der aktivierten Features (z.B. ["gqa", "film", "leaky_relu"])
        parent_run_id: ID des Parent-Runs (None für Root-Runs)
        lineage_depth: Anzahl der Generationen von Parent-Runs
        siblings_count: Anzahl der Runs mit gleichem Parent
        budget_class: Budget-Klasse basierend auf Modell-Parametern ("low", "medium", "high")
        sequence_length: Kontext-Länge Kategorie ("local" für short, "remote" für long)
        quantization_type: Quantisierungs-Typ ("none", "int6", "int5", "mixed", "gptq_lite")
        step_time_ms: Zeit pro Step in Millisekunden (normalisiert pro Parameter)
        memory_usage_mb: Speichernutzung in MB
        training_stability: Stabilität der Loss-Kurve (0-1, höher = stabiler)
        delta_bpb_vs_parent: BPB-Veränderung gegenüber Parent (negativ = Verbesserung)
        efficiency_gain_percent: Effizienz-Gewinn in Prozent gegenüber Parent
        model_size_change_percent: Modell-Größen-Änderung in Prozent gegenüber Parent
        quant_gap: Post-Quantisierungs-Degradation (BPB-Anstieg nach Quantisierung)
        seed_variance: Varianz über mehrere Seeds (wenn verfügbar)
        confidence_interval_width: Breite des 95% Konfidenz-Intervalls für BPB
        days_since_first_feature_introduction: Tage seit Einführung des ersten Features
        runs_since_feature_last_successful: Runs seit letztem erfolgreichen Feature-Einsatz
        co_occurrence_with: Count von Feature-Kookkurrenzen {feature_name: count}
        previous_success_rate_in_similar_context: Erfolgsquote in ähnlichem Kontext (0-1)
    """
    
    run_id: str
    features_active: List[str] = field(default_factory=list)
    parent_run_id: Optional[str] = None
    lineage_depth: int = 0
    siblings_count: int = 0
    budget_class: Literal["low", "medium", "high"] = "medium"
    sequence_length: Literal["local", "remote"] = "local"
    quantization_type: Literal["none", "int6", "int5", "mixed", "gptq_lite"] = "none"
    step_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    training_stability: float = 1.0
    delta_bpb_vs_parent: float = 0.0
    efficiency_gain_percent: float = 0.0
    model_size_change_percent: float = 0.0
    quant_gap: float = 0.0
    seed_variance: float = 0.0
    confidence_interval_width: float = 0.0
    days_since_first_feature_introduction: int = 0
    runs_since_feature_last_successful: int = 0
    co_occurrence_with: Dict[str, int] = field(default_factory=dict)
    previous_success_rate_in_similar_context: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung."""
        return {
            "run_id": self.run_id,
            "features_active": self.features_active,
            "parent_run_id": self.parent_run_id,
            "lineage_depth": self.lineage_depth,
            "siblings_count": self.siblings_count,
            "budget_class": self.budget_class,
            "sequence_length": self.sequence_length,
            "quantization_type": self.quantization_type,
            "step_time_ms": self.step_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "training_stability": self.training_stability,
            "delta_bpb_vs_parent": self.delta_bpb_vs_parent,
            "efficiency_gain_percent": self.efficiency_gain_percent,
            "model_size_change_percent": self.model_size_change_percent,
            "quant_gap": self.quant_gap,
            "seed_variance": self.seed_variance,
            "confidence_interval_width": self.confidence_interval_width,
            "days_since_first_feature_introduction": self.days_since_first_feature_introduction,
            "runs_since_feature_last_successful": self.runs_since_feature_last_successful,
            "co_occurrence_with": self.co_occurrence_with,
            "previous_success_rate_in_similar_context": self.previous_success_rate_in_similar_context,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunMetaFeatures:
        """Erstelle RunMetaFeatures aus Dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            features_active=data.get("features_active", []),
            parent_run_id=data.get("parent_run_id"),
            lineage_depth=data.get("lineage_depth", 0),
            siblings_count=data.get("siblings_count", 0),
            budget_class=data.get("budget_class", "medium"),
            sequence_length=data.get("sequence_length", "local"),
            quantization_type=data.get("quantization_type", "none"),
            step_time_ms=data.get("step_time_ms", 0.0),
            memory_usage_mb=data.get("memory_usage_mb", 0.0),
            training_stability=data.get("training_stability", 1.0),
            delta_bpb_vs_parent=data.get("delta_bpb_vs_parent", 0.0),
            efficiency_gain_percent=data.get("efficiency_gain_percent", 0.0),
            model_size_change_percent=data.get("model_size_change_percent", 0.0),
            quant_gap=data.get("quant_gap", 0.0),
            seed_variance=data.get("seed_variance", 0.0),
            confidence_interval_width=data.get("confidence_interval_width", 0.0),
            days_since_first_feature_introduction=data.get("days_since_first_feature_introduction", 0),
            runs_since_feature_last_successful=data.get("runs_since_feature_last_successful", 0),
            co_occurrence_with=data.get("co_occurrence_with", {}),
            previous_success_rate_in_similar_context=data.get("previous_success_rate_in_similar_context", 0.0),
        )


class MetaFeatureExtractor:
    """
    Extrahiert Meta-Features aus bestehenden Runs.
    
    Verwendung:
        extractor = MetaFeatureExtractor()
        features = extractor.extract("run001", registry)
        
        # Batch-Extraktion
        features_list = extractor.extract_batch(["run001", "run002"], registry)
        
        # Co-occurrence Statistiken
        co_occ = extractor.compute_co_occurrence(features_list)
    """
    
    def __init__(self, configs_dir: str | Path = "configs"):
        """
        Initialisiere den MetaFeatureExtractor.
        
        Args:
            configs_dir: Pfad zum Konfigurationsverzeichnis
        """
        self.configs_dir = Path(configs_dir)
    
    def _load_run_config(self, run_id: str) -> Optional[Config]:
        """Lade die Konfiguration für einen Run."""
        config_path = self.configs_dir / "runs" / f"{run_id}.yaml"
        if config_path.exists():
            return load_config(config_path)
        return None
    
    def _extract_features_from_config(self, config: Config) -> List[str]:
        """
        Extrahiere aktivierte Features aus der Konfiguration.
        
        Features werden aus verschiedenen Bereichen extrahiert:
        - model.attention.type (gqa, mha, mqa)
        - model.recurrence.enabled
        - model.xsa.enabled
        - model.film.enabled
        - model.ttt.enabled
        - model.gated_mlp.enabled
        - features.hasher, features.leaky_relu, etc.
        - quant.enabled und quant.type
        """
        features = []
        
        # Attention Features
        attention = config.model.get("attention", {})
        attn_type = attention.get("type", "mha")
        if attn_type == "gqa":
            features.append("gqa")
        elif attn_type == "mqa":
            features.append("mqa")
        
        if attention.get("rope", False):
            features.append("rope")
        if attention.get("partial_rope", False):
            features.append("partial_rope")
        if attention.get("kv_sharing", False):
            features.append("kv_sharing")
        
        # Recurrence Features
        recurrence = config.model.get("recurrence", {})
        if recurrence.get("enabled", False):
            features.append("recurrence")
            if recurrence.get("tied", False):
                features.append("tied_recurrence")
            if recurrence.get("loop_embeddings", False):
                features.append("loop_embeddings")
        
        # Feature Gates (model level)
        if config.model.get("xsa", {}).get("enabled", False):
            features.append("xsa")
        if config.model.get("film", {}).get("enabled", False):
            features.append("film")
        if config.model.get("ttt", {}).get("enabled", False):
            features.append("ttt")
        if config.model.get("gated_mlp", {}).get("enabled", False):
            features.append("gated_mlp")
        
        # Activation Features
        activation = config.model.get("activation", "gelu")
        if activation == "leaky_relu":
            features.append("leaky_relu")
        elif activation == "star_relu":
            features.append("star_relu")
        elif activation == "gated":
            features.append("gated_activation")
        
        # Top-level Feature Gates
        if config.features.get("hasher", False):
            features.append("hasher")
        if config.features.get("leaky_relu", False):
            features.append("leaky_relu")
        if config.features.get("gated_mlp", False):
            features.append("gated_mlp")
        if config.features.get("mixed_quant", False):
            features.append("mixed_quant")
        
        # Tokenizer Features
        tokenizer = config.tokenizer
        tokenizer_type = tokenizer.get("type", "byte")
        if tokenizer_type == "bigram_hash":
            features.append("bigram_hash")
        elif tokenizer_type == "trigram_hash":
            features.append("trigram_hash")
        if tokenizer.get("byte_fallback", False):
            features.append("byte_fallback")
        
        # Quantization Features
        quant = config.quant
        if quant.get("enabled", False):
            quant_type = quant.get("type", "int6")
            if quant_type == "int5_int6_mixed":
                features.append("mixed_quant")
            elif quant_type == "gptq_lite":
                features.append("gptq_lite")
            else:
                features.append(f"quant_{quant_type}")
        
        return sorted(features)
    
    def _compute_budget_class(self, config: Config) -> Literal["low", "medium", "high"]:
        """
        Berechne Budget-Klasse basierend auf Modell-Parametern.
        
        Heuristik:
        - low: d_model < 256 oder num_layers < 4
        - medium: 256 <= d_model <= 512 und 4 <= num_layers <= 12
        - high: d_model > 512 oder num_layers > 12
        """
        model = config.model
        d_model = model.get("d_model", 512)
        num_layers = model.get("num_layers", 6)
        
        # Parameter-basierte Schätzung (ungefähre Parameterzahl in Millionen)
        # Transformer: ~12 * d_model^2 * num_layers (sehr grob)
        approx_params = 12 * (d_model ** 2) * num_layers / 1e6
        
        if approx_params < 50 or d_model < 256 or num_layers < 4:
            return "low"
        elif approx_params > 200 or d_model > 768 or num_layers > 16:
            return "high"
        else:
            return "medium"
    
    def _compute_sequence_length(self, config: Config) -> Literal["local", "remote"]:
        """
        Bestimme Sequenzlängen-Kategorie.
        
        - local: max_seq_len <= 1024 (kurzer Kontext, lokal ausführbar)
        - remote: max_seq_len > 1024 (langer Kontext, benötigt mehr Ressourcen)
        """
        max_seq_len = config.model.get("max_seq_len", 1024)
        
        # Auch local_proxy Konfiguration berücksichtigen
        local_proxy = getattr(config, 'local_proxy', {})
        if isinstance(local_proxy, dict):
            if local_proxy.get("enabled", False):
                seq_len = local_proxy.get("seq_len", max_seq_len)
                if seq_len <= 512:
                    return "local"
        
        if max_seq_len <= 1024:
            return "local"
        else:
            return "remote"
    
    def _extract_quantization_type(self, config: Config) -> Literal["none", "int6", "int5", "mixed", "gptq_lite"]:
        """Extrahiere Quantisierungs-Typ aus der Konfiguration."""
        quant = config.quant
        
        if not quant.get("enabled", False):
            return "none"
        
        quant_type = quant.get("type", "int6")
        
        if quant_type == "int5_int6_mixed" or config.features.get("mixed_quant", False):
            return "mixed"
        elif quant_type == "gptq_lite" or quant.get("gptq_lite", False):
            return "gptq_lite"
        elif quant_type == "int5":
            return "int5"
        else:
            return "int6"
    
    def _compute_lineage_depth(self, run_entry: RunEntry, registry: RunRegistry) -> int:
        """Berechne die Lineage-Tiefe (Anzahl der Parent-Generationen)."""
        depth = 0
        current = run_entry
        
        while current.parent_run_id:
            parent = registry.get(current.parent_run_id)
            if parent:
                depth += 1
                current = parent
            else:
                break
        
        return depth
    
    def _compute_siblings_count(self, run_entry: RunEntry, registry: RunRegistry) -> int:
        """Berechne Anzahl der Siblings (Runs mit gleichem Parent)."""
        if not run_entry.parent_run_id:
            return 0
        
        siblings = registry.get_children(run_entry.parent_run_id)
        # Exkludiere den Run selbst
        return len(siblings) - 1
    
    def _compute_training_stability(self, run_entry: RunEntry) -> float:
        """
        Berechne Training-Stabilität basierend auf verfügbaren Metriken.
        
        Heuristik:
        - Completed Runs mit validen Metriken: 1.0
        - Failed/Killed Runs: 0.0
        - Running Runs: 0.5 (unbekannt)
        - Pending Runs: 0.0 (noch nicht gestartet)
        """
        status = run_entry.status
        
        if status == "completed":
            # Prüfe ob valide Metriken vorhanden sind
            if run_entry.val_bpb is not None and run_entry.ms_per_step is not None:
                return 1.0
            return 0.5
        elif status == "failed" or status == "killed":
            return 0.0
        elif status == "running":
            return 0.5
        else:  # pending
            return 0.0
    
    def _compute_efficiency_gain(self, run_entry: RunEntry, registry: RunRegistry) -> float:
        """
        Berechne Effizienz-Gewinn in Prozent gegenüber Parent.
        
        Formel: (parent_step_time / current_step_time - 1) * 100
        Positiv = schneller als Parent
        """
        if not run_entry.parent_run_id:
            return 0.0
        
        parent = registry.get(run_entry.parent_run_id)
        if not parent or parent.ms_per_step is None or run_entry.ms_per_step is None:
            return 0.0
        
        if run_entry.ms_per_step <= 0:
            return 0.0
        
        efficiency = (parent.ms_per_step / run_entry.ms_per_step - 1) * 100
        return round(efficiency, 2)
    
    def _compute_model_size_change(self, run_entry: RunEntry, registry: RunRegistry) -> float:
        """
        Berechne Modell-Größen-Änderung in Prozent gegenüber Parent.
        
        Formel: (current_size / parent_size - 1) * 100
        Positiv = größer als Parent
        """
        if not run_entry.parent_run_id:
            return 0.0
        
        parent = registry.get(run_entry.parent_run_id)
        if not parent or parent.artifact_bytes is None or run_entry.artifact_bytes is None:
            return 0.0
        
        if parent.artifact_bytes <= 0:
            return 0.0
        
        change = (run_entry.artifact_bytes / parent.artifact_bytes - 1) * 100
        return round(change, 2)
    
    def _compute_quant_gap(self, run_entry: RunEntry, registry: RunRegistry) -> float:
        """
        Berechne Post-Quantisierungs-Degradation.
        
        Formel: quantized_val_bpb - val_bpb
        Positiv = Performance-Verlust durch Quantisierung
        """
        if run_entry.quantized_val_bpb is None or run_entry.val_bpb is None:
            return 0.0
        
        gap = run_entry.quantized_val_bpb - run_entry.val_bpb
        return round(gap, 4)
    
    def _compute_seed_variance(self, run_entry: RunEntry, registry: RunRegistry) -> float:
        """
        Berechne Varianz über Seeds für die gleiche Konfiguration.
        
        Verwendet registry.get_seed_statistics() für Berechnung.
        """
        if not run_entry.config_hash:
            return 0.0
        
        stats = registry.get_seed_statistics(run_entry.config_hash)
        bpb_stats = stats.get("bpb", {})
        
        # Standardabweichung als Varianz-Maß
        std = bpb_stats.get("std", 0.0)
        return round(std, 4)
    
    def _compute_confidence_interval_width(self, run_entry: RunEntry, registry: RunRegistry) -> float:
        """
        Berechne Breite des 95% Konfidenz-Intervalls für BPB.
        
        Formel: 1.96 * std / sqrt(n) * 2 (beide Seiten)
        """
        if not run_entry.config_hash:
            return 0.0
        
        stats = registry.get_seed_statistics(run_entry.config_hash)
        bpb_stats = stats.get("bpb", {})
        
        std = bpb_stats.get("std", 0.0)
        n = bpb_stats.get("count", 1)
        
        if n <= 1 or std == 0:
            return 0.0
        
        # 95% CI Breite = 2 * 1.96 * std / sqrt(n)
        width = 2 * 1.96 * std / (n ** 0.5)
        return round(width, 4)
    
    def _compute_previous_success_rate(self, run_entry: RunEntry, registry: RunRegistry, 
                                       features: List[str]) -> float:
        """
        Berechne Erfolgsquote in ähnlichem Kontext.
        
        Ähnlicher Kontext = gleiche Budget-Klasse + gleiche Quantisierung
        
        Formel: successful_runs / total_runs
        """
        # Finde ähnliche Runs
        all_runs = registry.list_runs()
        
        similar_runs = []
        for run in all_runs:
            if run.run_id == run_entry.run_id:
                continue
            
            # Prüfe Ähnlichkeit
            run_config = self._load_run_config(run.run_id)
            if not run_config:
                continue
            
            run_budget = self._compute_budget_class(run_config)
            run_quant = self._extract_quantization_type(run_config)
            
            if run_budget == self._compute_budget_class(run_config) and \
               run_quant == self._extract_quantization_type(run_config):
                similar_runs.append(run)
        
        if not similar_runs:
            return 0.0
        
        # Zähle erfolgreiche Runs
        successful = sum(1 for r in similar_runs if r.status == "completed" and r.val_bpb is not None)
        
        return round(successful / len(similar_runs), 2)
    
    def extract(self, run_id: str, registry: RunRegistry) -> RunMetaFeatures:
        """
        Extrahiere Meta-Features für einen einzelnen Run.
        
        Args:
            run_id: ID des Runs
            registry: RunRegistry Instanz
            
        Returns:
            RunMetaFeatures Objekt mit allen extrahierten Features
            
        Raises:
            ValueError: Wenn der Run nicht gefunden wird
        """
        run_entry = registry.get(run_id)
        if run_entry is None:
            raise ValueError(f"Run '{run_id}' nicht im Registry gefunden")
        
        # Lade Konfiguration
        config = self._load_run_config(run_id)
        if config is None:
            # Fallback: Erstelle leere Meta-Features wenn Config fehlt
            return RunMetaFeatures(
                run_id=run_id,
                parent_run_id=run_entry.parent_run_id,
            )
        
        # Extrahiere Features
        features_active = self._extract_features_from_config(config)
        budget_class = self._compute_budget_class(config)
        sequence_length = self._compute_sequence_length(config)
        quantization_type = self._extract_quantization_type(config)
        
        # Berechne Lineage-Metriken
        lineage_depth = self._compute_lineage_depth(run_entry, registry)
        siblings_count = self._compute_siblings_count(run_entry, registry)
        
        # Berechne Performance-Metriken
        step_time_ms = run_entry.ms_per_step or 0.0
        memory_usage_mb = run_entry.artifact_bytes / (1024 * 1024) if run_entry.artifact_bytes else 0.0
        training_stability = self._compute_training_stability(run_entry)
        delta_bpb_vs_parent = run_entry.delta_bpb or 0.0
        efficiency_gain_percent = self._compute_efficiency_gain(run_entry, registry)
        model_size_change_percent = self._compute_model_size_change(run_entry, registry)
        quant_gap = self._compute_quant_gap(run_entry, registry)
        
        # Statistische Eigenschaften
        seed_variance = self._compute_seed_variance(run_entry, registry)
        confidence_interval_width = self._compute_confidence_interval_width(run_entry, registry)
        
        # Erfolgsquote (wird später mit Co-occurrence aktualisiert)
        previous_success_rate = self._compute_previous_success_rate(run_entry, registry, features_active)
        
        return RunMetaFeatures(
            run_id=run_id,
            features_active=features_active,
            parent_run_id=run_entry.parent_run_id,
            lineage_depth=lineage_depth,
            siblings_count=siblings_count,
            budget_class=budget_class,
            sequence_length=sequence_length,
            quantization_type=quantization_type,
            step_time_ms=step_time_ms,
            memory_usage_mb=memory_usage_mb,
            training_stability=training_stability,
            delta_bpb_vs_parent=delta_bpb_vs_parent,
            efficiency_gain_percent=efficiency_gain_percent,
            model_size_change_percent=model_size_change_percent,
            quant_gap=quant_gap,
            seed_variance=seed_variance,
            confidence_interval_width=confidence_interval_width,
            days_since_first_feature_introduction=0,  # TODO: Implementierung wenn historische Daten verfügbar
            runs_since_feature_last_successful=0,  # TODO: Implementierung wenn historische Daten verfügbar
            co_occurrence_with={},  # Wird separat berechnet
            previous_success_rate_in_similar_context=previous_success_rate,
        )
    
    def extract_batch(self, run_ids: List[str], registry: RunRegistry) -> List[RunMetaFeatures]:
        """
        Extrahiere Meta-Features für mehrere Runs.
        
        Args:
            run_ids: Liste von Run-IDs
            registry: RunRegistry Instanz
            
        Returns:
            Liste von RunMetaFeatures Objekten
        """
        features_list = []
        
        for run_id in run_ids:
            try:
                features = self.extract(run_id, registry)
                features_list.append(features)
            except ValueError as e:
                # Überspringe Runs die nicht gefunden wurden
                print(f"Warnung: {e}")
                continue
        
        return features_list
    
    def compute_co_occurrence(self, features: List[RunMetaFeatures]) -> Dict[Tuple[str, str], int]:
        """
        Berechne Feature-Co-occurrence Statistiken.
        
        Zählt wie oft Feature-Paare gemeinsam in Runs vorkommen.
        
        Args:
            features: Liste von RunMetaFeatures Objekten
            
        Returns:
            Dictionary mit (feature1, feature2) -> count
        """
        co_occurrence: Dict[Tuple[str, str], int] = {}
        
        for feat in features:
            active = sorted(feat.features_active)
            
            # Alle Paare zählen
            for i, f1 in enumerate(active):
                for f2 in active[i + 1:]:
                    key = (f1, f2)
                    co_occurrence[key] = co_occurrence.get(key, 0) + 1
        
        return co_occurrence
    
    def enrich_features_with_co_occurrence(self, features: List[RunMetaFeatures]) -> List[RunMetaFeatures]:
        """
        Reichere Meta-Features mit Co-occurrence Informationen an.
        
        Args:
            features: Liste von RunMetaFeatures Objekten
            
        Returns:
            Liste von angereicherten RunMetaFeatures Objekten
        """
        co_occurrence = self.compute_co_occurrence(features)
        
        enriched = []
        for feat in features:
            # Kopiere existierende Features
            enriched_feat = RunMetaFeatures(
                run_id=feat.run_id,
                features_active=feat.features_active.copy(),
                parent_run_id=feat.parent_run_id,
                lineage_depth=feat.lineage_depth,
                siblings_count=feat.siblings_count,
                budget_class=feat.budget_class,
                sequence_length=feat.sequence_length,
                quantization_type=feat.quantization_type,
                step_time_ms=feat.step_time_ms,
                memory_usage_mb=feat.memory_usage_mb,
                training_stability=feat.training_stability,
                delta_bpb_vs_parent=feat.delta_bpb_vs_parent,
                efficiency_gain_percent=feat.efficiency_gain_percent,
                model_size_change_percent=feat.model_size_change_percent,
                quant_gap=feat.quant_gap,
                seed_variance=feat.seed_variance,
                confidence_interval_width=feat.confidence_interval_width,
                days_since_first_feature_introduction=feat.days_since_first_feature_introduction,
                runs_since_feature_last_successful=feat.runs_since_feature_last_successful,
                co_occurrence_with={},
                previous_success_rate_in_similar_context=feat.previous_success_rate_in_similar_context,
            )
            
            # Fülle Co-occurrence für aktive Features
            for f1, f2 in co_occurrence.keys():
                if f1 in feat.features_active:
                    enriched_feat.co_occurrence_with[f2] = enriched_feat.co_occurrence_with.get(f2, 0) + co_occurrence[(f1, f2)]
                if f2 in feat.features_active:
                    enriched_feat.co_occurrence_with[f1] = enriched_feat.co_occurrence_with.get(f1, 0) + co_occurrence[(f1, f2)]
            
            enriched.append(enriched_feat)
        
        return enriched
