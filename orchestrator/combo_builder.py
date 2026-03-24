"""Dynamic Combo Builder für Phase 3.

Dieses Modul erstellt automatisch Kombinationen der besten Features
aus Phasen 1-2 basierend auf Gate-Status und Metriken.

Rules for Combinations:
1. Nur Features kombinieren, die stabil positiv waren (Gate=PASS)
2. Maximal 3 neue Freiheitsgrade pro Kombi-Run
3. Keine Kombination aus zwei "knapp positiven" Features (Gate=WATCH)
4. Mindestens ein "starkes" Feature pro Kombi (Gate=PASS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from core.config import Config, load_config
from core.registry import RunRegistry, RunEntry


class GateStatus(Enum):
    """Gate-Status für Features."""
    PASS = "pass"
    WATCH = "watch"
    FAIL = "fail"
    PENDING = "pending"


@dataclass
class FeatureCandidate:
    """Kandidat für Kombination."""

    feature_name: str
    run_id: str
    gate_status: GateStatus
    val_bpb: float | None = None
    delta_bpb: float | None = None
    ms_per_step: float | None = None
    delta_ms: float | None = None
    artifact_bytes: int = 0
    priority_score: float = 0.0

    def is_combinable(self) -> bool:
        """Check if feature can be combined."""
        return self.gate_status == GateStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "run_id": self.run_id,
            "gate_status": self.gate_status.value,
            "val_bpb": self.val_bpb,
            "delta_bpb": self.delta_bpb,
            "priority_score": self.priority_score,
        }


@dataclass
class ComboConfig:
    """Konfiguration für Combo-Run."""

    combo_id: str
    parent_run_id: str
    seed: int = 42

    # Modell-Konfiguration
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: int = 4

    # Tokenizer
    tokenizer_type: str = "byte"
    tokenizer_vocab_size: int = 256

    # Attention
    attention_type: str = "gqa"
    gqa_groups: int = 4

    # Aktivierung
    activation: str = "gelu"

    # Features
    recurrence_enabled: bool = False
    xsa_enabled: bool = False
    gated_mlp_enabled: bool = False

    # Quantisierung
    quant_enabled: bool = False
    quant_attention_dtype: str = "int6"
    quant_mlp_dtype: str = "int6"

    # Metadata
    selected_features: list[FeatureCandidate] = field(default_factory=list)
    max_new_features: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.combo_id,
            "parent_run_id": self.parent_run_id,
            "seed": self.seed,
            "model": {
                "d_model": self.d_model,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "activation": self.activation,
                "attention": {
                    "type": self.attention_type,
                    "gqa_groups": self.gqa_groups,
                    "kv_heads": self.num_heads // self.gqa_groups if self.gqa_groups > 0 else 4,
                    "kv_sharing": True,
                    "rope": True,
                },
                "recurrence": {
                    "enabled": self.recurrence_enabled,
                    "type": "tied",
                    "depth": 4,
                    "loop_embeddings": True,
                },
                "xsa": {"enabled": self.xsa_enabled},
                "gated_mlp": {"enabled": self.gated_mlp_enabled},
            },
            "tokenizer": {
                "type": self.tokenizer_type,
                "vocab_size": self.tokenizer_vocab_size,
                "byte_fallback": True,
            },
            "quant": {
                "enabled": self.quant_enabled,
                "type": "int5_int6_mixed" if self.quant_enabled else "int6",
                "attention_dtype": self.quant_attention_dtype,
                "mlp_dtype": self.quant_mlp_dtype,
                "embedding_dtype": "int6",
                "gptq_lite": False,
                "calibration_samples": 256,
            },
            "features": {
                "hasher": self.tokenizer_type != "byte",
                "leaky_relu": self.activation in ["leaky_relu", "leaky_relu_squared"],
                "gated_mlp": self.gated_mlp_enabled,
                "mixed_quant": self.quant_enabled,
            },
            "training": {
                "num_steps": 10000,
                "batch_size": 32,
                "learning_rate": 1e-3,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "warmup_steps": 1000,
                "weight_decay": 0.1,
                "ema_decay": None,
            },
            "phase3": {
                "is_combo": True,
                "combo_source": "dynamic",
                "max_new_features": self.max_new_features,
                "gate_freeze_required": True,
                "selected_features": [f.to_dict() for f in self.selected_features],
            },
        }

    def save(self, path: str | Path) -> None:
        """Save combo config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


class DynamicComboBuilder:
    """Builder für dynamische Feature-Kombinationen."""

    def __init__(self, registry: RunRegistry):
        self.registry = registry
        self.feature_candidates: dict[str, list[FeatureCandidate]] = {}

    def analyze_phase1_results(self) -> dict[str, FeatureCandidate]:
        """Analysiere Phase 1 Ergebnisse für Tokenizer-Auswahl."""
        candidates = {}

        # Tokenizer candidates
        tokenizer_runs = {
            "byte": "run001_control",
            "bigram_4k": "run002a_bigram_4k",
            "bigram_8k": "run002b_bigram_8k",
            "trigram_small": "run002c_trigram_small",
        }

        for tokenizer_type, run_id in tokenizer_runs.items():
            entry = self.registry.get(run_id)
            if entry and entry.status == "completed":
                gate_status = self._infer_gate_status(entry)
                candidates[f"tokenizer_{tokenizer_type}"] = FeatureCandidate(
                    feature_name=f"tokenizer_{tokenizer_type}",
                    run_id=run_id,
                    gate_status=gate_status,
                    val_bpb=entry.val_bpb if entry.val_bpb is not None else 1.5,  # Default
                    delta_bpb=entry.delta_bpb if entry.delta_bpb is not None else 0.0,
                    ms_per_step=entry.ms_per_step if entry.ms_per_step is not None else 10.0,
                    delta_ms=entry.delta_ms if entry.delta_ms is not None else 0.0,
                    artifact_bytes=entry.artifact_bytes if entry.artifact_bytes is not None else 10_000_000,
                    priority_score=self._compute_priority_score(entry),
                )

        return candidates

    def analyze_phase2_results(self) -> dict[str, FeatureCandidate]:
        """Analysiere Phase 2 Ergebnisse für Feature-Auswahl."""
        candidates = {}

        # Feature candidates
        feature_runs = {
            "xsa": "run003_xsa",
            "leaky_relu": "run004_leakyrelu",
            "quant_mlp5_attn6": "run005a_quant_mlp5_attn6",
            "quant_attn5_mlp6": "run005b_quant_attn5_mlp6",
            "film": "run006_film",
            "star_relu": "run008a_star_relu",
            "gated_mlp": "run008b_true_gated_mlp",
            "gqa": "run009_gqa",
            "recurrence": "run010_recurrence",
        }

        for feature_name, run_id in feature_runs.items():
            entry = self.registry.get(run_id)
            if entry and entry.status == "completed":
                gate_status = self._infer_gate_status(entry)
                candidates[feature_name] = FeatureCandidate(
                    feature_name=feature_name,
                    run_id=run_id,
                    gate_status=gate_status,
                    val_bpb=entry.val_bpb if entry.val_bpb is not None else 1.5,  # Default
                    delta_bpb=entry.delta_bpb if entry.delta_bpb is not None else 0.0,
                    ms_per_step=entry.ms_per_step if entry.ms_per_step is not None else 10.0,
                    delta_ms=entry.delta_ms if entry.delta_ms is not None else 0.0,
                    artifact_bytes=entry.artifact_bytes if entry.artifact_bytes is not None else 10_000_000,
                    priority_score=self._compute_priority_score(entry),
                )

        return candidates

    def _infer_gate_status(self, entry: RunEntry) -> GateStatus:
        """Infer Gate-Status aus Run-Ergebnissen."""
        if not entry:
            return GateStatus.PENDING
            
        # Check if killed
        if entry.status == "killed":
            return GateStatus.FAIL

        # Check notes for gate status (from Phase 2 evaluation)
        notes = entry.notes.lower() if entry.notes else ""
        if "pass" in notes:
            return GateStatus.PASS
        elif "watch" in notes:
            return GateStatus.WATCH
        elif "fail" in notes:
            return GateStatus.FAIL

        # Infer from metrics (with null safety)
        if entry.delta_bpb is not None:
            if entry.delta_bpb < -0.03:  # Significant improvement
                return GateStatus.PASS
            elif entry.delta_bpb > 0.05:  # Significant regression
                return GateStatus.FAIL
            else:
                return GateStatus.WATCH

        # Default for runs without metrics
        if entry.status == "completed":
            return GateStatus.WATCH  # Assume neutral until proven otherwise
        
        return GateStatus.PENDING

    def _compute_priority_score(self, entry: RunEntry) -> float:
        """Compute priority score for feature ranking."""
        score = 0.0

        # BPB improvement (lower is better)
        if entry and entry.delta_bpb is not None:
            score -= entry.delta_bpb * 100  # Negative delta = improvement

        # Speed improvement (lower ms is better)
        if entry and entry.delta_ms is not None and entry.delta_ms < 0:
            score += abs(entry.delta_ms) * 0.5

        # Artifact size (smaller is better)
        if entry and entry.artifact_bytes and entry.artifact_bytes < 10_000_000:
            score += 1.0

        return score

    def select_best_tokenizer(self) -> FeatureCandidate | None:
        """Select best tokenizer from Phase 1 results."""
        candidates = self.analyze_phase1_results()
        tokenizer_candidates = [
            c for c in candidates.values()
            if c.feature_name.startswith("tokenizer_") and c.is_combinable()
        ]

        if not tokenizer_candidates:
            return None

        return max(tokenizer_candidates, key=lambda c: c.priority_score)

    def select_best_activation(self) -> FeatureCandidate | None:
        """Select best activation from Phase 2 results."""
        candidates = self.analyze_phase2_results()
        activation_candidates = [
            c for c in candidates.values()
            if c.feature_name in ["leaky_relu", "star_relu"] and c.is_combinable()
        ]

        if not activation_candidates:
            return None

        return max(activation_candidates, key=lambda c: c.priority_score)

    def select_best_attention(self) -> FeatureCandidate | None:
        """Select best attention from Phase 2 results."""
        candidates = self.analyze_phase2_results()
        attention_candidates = [
            c for c in candidates.values()
            if c.feature_name in ["gqa", "xsa"] and c.is_combinable()
        ]

        if not attention_candidates:
            return None

        return max(attention_candidates, key=lambda c: c.priority_score)

    def select_best_quant_strategy(self) -> FeatureCandidate | None:
        """Select best quantization strategy from run005a/b."""
        candidates = self.analyze_phase2_results()
        quant_candidates = [
            c for c in candidates.values()
            if c.feature_name in ["quant_mlp5_attn6", "quant_attn5_mlp6"]
            and c.is_combinable()
        ]

        if not quant_candidates:
            return None

        return max(quant_candidates, key=lambda c: c.priority_score)

    def build_best_combo(self, is_quantized: bool = False) -> ComboConfig:
        """Build best combo configuration from validated features.

        Args:
            is_quantized: Whether to include quantization

        Returns:
            ComboConfig for run016 or run017
        """
        # Select best features
        tokenizer = self.select_best_tokenizer()
        activation = self.select_best_activation()
        attention = self.select_best_attention()
        quant_strategy = self.select_best_quant_strategy() if is_quantized else None

        # Build combo config
        combo_id = "run017_best_combo_quantized" if is_quantized else "run016_best_combo_a"
        parent_id = "run016_best_combo_a" if is_quantized else "run001_control"

        combo = ComboConfig(
            combo_id=combo_id,
            parent_run_id=parent_id,
            seed=42,
        )

        # Apply selected features
        selected = []

        if tokenizer and tokenizer.tokenizer_type != "byte":
            combo.tokenizer_type = tokenizer.tokenizer_type
            if "4k" in tokenizer.tokenizer_type:
                combo.tokenizer_vocab_size = 4096
            elif "8k" in tokenizer.tokenizer_type:
                combo.tokenizer_vocab_size = 8192
            selected.append(tokenizer)

        if activation and activation.feature_name in ["leaky_relu", "star_relu"]:
            combo.activation = "leaky_relu_squared" if "leaky" in activation.feature_name else "star_relu"
            selected.append(activation)

        if attention and attention.feature_name == "gqa":
            combo.attention_type = "gqa"
            selected.append(attention)
        elif attention and attention.feature_name == "xsa":
            combo.attention_type = "gqa"  # XSA requires GQA
            combo.xsa_enabled = True
            selected.append(attention)

        if quant_strategy and is_quantized:
            combo.quant_enabled = True
            if "mlp5" in quant_strategy.feature_name:
                combo.quant_mlp_dtype = "int5"
                combo.quant_attention_dtype = "int6"
            else:
                combo.quant_mlp_dtype = "int6"
                combo.quant_attention_dtype = "int5"
            selected.append(quant_strategy)

        combo.selected_features = selected

        # Enforce max 3 new features
        if len(combo.selected_features) > combo.max_new_features:
            combo.selected_features = sorted(
                combo.selected_features,
                key=lambda c: c.priority_score,
                reverse=True
            )[:combo.max_new_features]

        return combo

    def check_gate_freeze(self) -> tuple[bool, list[str]]:
        """Check if gate-freeze conditions are met.

        Returns:
            Tuple of (ready, list of blocking reasons)
        """
        blocking = []

        # Check Phase 1 runs completed
        phase1_runs = [
            "run001_control",
            "run001b_frontierish_control",
            "run002a_bigram_4k",
            "run002b_bigram_8k",
            "run002c_trigram_small",
        ]

        for run_id in phase1_runs:
            entry = self.registry.get(run_id)
            if not entry or entry.status not in ["completed", "killed"]:
                blocking.append(f"Phase 1 run {run_id} not completed")

        # Check Phase 2 runs completed
        phase2_runs = [
            "run003_xsa",
            "run004_leakyrelu",
            "run005a_quant_mlp5_attn6",
            "run005b_quant_attn5_mlp6",
            "run006_film",
            "run007_ttt",
            "run008a_star_relu",
            "run008b_true_gated_mlp",
            "run009_gqa",
            "run010_recurrence",
        ]

        for run_id in phase2_runs:
            entry = self.registry.get(run_id)
            if not entry or entry.status not in ["completed", "killed"]:
                blocking.append(f"Phase 2 run {run_id} not completed")

        return len(blocking) == 0, blocking

    def generate_combo_run(
        self,
        is_quantized: bool = False,
        output_dir: str | Path = "configs/runs",
    ) -> ComboConfig | None:
        """Generate combo run configuration.

        Args:
            is_quantized: Whether to generate quantized combo
            output_dir: Directory to save config

        Returns:
            Generated ComboConfig or None if gate-freeze not met
        """
        # Check gate-freeze
        ready, blocking = self.check_gate_freeze()
        if not ready:
            print(f"Gate-freeze not met: {blocking}")
            return None

        # Build combo
        combo = self.build_best_combo(is_quantized)

        # Save config
        output_path = Path(output_dir) / f"{combo.combo_id}.yaml"
        combo.save(output_path)

        print(f"Generated combo config: {output_path}")
        print(f"Selected features: {[f.feature_name for f in combo.selected_features]}")

        return combo


def create_combo_builder(registry: RunRegistry | None = None) -> DynamicComboBuilder:
    """Create a dynamic combo builder."""
    return DynamicComboBuilder(registry or RunRegistry())


def generate_phase3_combos(
    output_dir: str = "configs/runs",
    force: bool = False,
) -> tuple[ComboConfig | None, ComboConfig | None]:
    """Generate Phase 3 combo configurations.

    Args:
        output_dir: Directory to save configs
        force: Generate even if gate-freeze not met

    Returns:
        Tuple of (best_combo, quantized_combo)
    """
    registry = RunRegistry()
    builder = create_combo_builder(registry)

    # Check gate-freeze
    ready, blocking = builder.check_gate_freeze()
    if not ready and not force:
        print("Gate-freeze conditions not met:")
        for reason in blocking:
            print(f"  - {reason}")
        print("\nUse force=True to generate anyway (with placeholder values)")
        return None, None

    # Generate combos
    best_combo = builder.generate_combo_run(is_quantized=False, output_dir=output_dir)
    quantized_combo = builder.generate_combo_run(is_quantized=True, output_dir=output_dir)

    return best_combo, quantized_combo
