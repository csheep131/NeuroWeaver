"""Feature Gates für Phase 2 Features.

Dieses Modul verwaltet die Feature-Gates für alle Phase 2 Features:
- XSA (Cross-Sequence Attention)
- FiLM (Feature-wise Linear Modulation)
- TTT (Test-Time Training)
- GQA (Grouped Query Attention)
- Recurrence (Recurrent Blocks)
- Gated MLP (SwiGLU/GeGLU)
- Aktivierungsfunktionen (LeakyReLU, Star-ReLU)

Jedes Feature hat:
- Status (ENABLED, DISABLED, KILLED, INVALID)
- Abhängigkeiten
- Konfiguration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class FeatureStatus(Enum):
    """Status eines Features."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    KILLED = "killed"
    INVALID = "invalid"


class FeatureType(Enum):
    """Verfügbare Feature-Typen."""

    # Tokenizer
    BYTE = "byte"
    BIGRAM_HASH = "bigram_hash"
    TRIGRAM_HASH = "trigram_hash"

    # Aktivierungen
    GELU = "gelu"
    LEAKY_RELU = "leaky_relu"
    LEAKY_RELU_SQUARED = "leaky_relu_squared"
    STAR_RELU = "star_relu"
    REPU = "relu_squared"

    # Attention
    STANDARD = "standard"
    GQA = "gqa"
    MQA = "mqa"
    MHA = "mha"

    # Features
    XSA = "xsa"
    FILM = "film"
    TTT = "ttt"
    GATED_MLP = "gated_mlp"
    RECURRENCE = "recurrence"

    # Quantisierung
    INT6 = "int6"
    INT5 = "int5"
    MIXED = "mixed"


@dataclass
class FeatureConfig:
    """Konfiguration für ein Feature."""

    feature_type: FeatureType
    status: FeatureStatus = FeatureStatus.DISABLED
    config: dict[str, Any] = field(default_factory=dict)

    def is_enabled(self) -> bool:
        """Check if feature is enabled."""
        return self.status == FeatureStatus.ENABLED

    def is_disabled(self) -> bool:
        """Check if feature is disabled."""
        return self.status == FeatureStatus.DISABLED

    def is_killed(self) -> bool:
        """Check if feature is killed."""
        return self.status == FeatureStatus.KILLED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_type": self.feature_type.value,
            "status": self.status.value,
            "config": self.config,
        }


@dataclass
class FeatureDependency:
    """Abhängigkeit für ein Feature."""

    feature: str
    condition: Callable[[dict[str, Any]], bool] = lambda x: True
    required: bool = False


@dataclass
class FeatureDefinition:
    """Definition eines Features mit Metadaten."""

    name: str
    feature_type: FeatureType
    description: str
    default_status: FeatureStatus = FeatureStatus.DISABLED
    dependencies: list[FeatureDependency] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)
    kill_criteria: list[dict[str, Any]] = field(default_factory=list)

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validiere Feature-Konfiguration.

        Returns:
            Tuple of (valid, list of errors)
        """
        errors = []

        # Check dependencies
        for dep in self.dependencies:
            if dep.required and not dep.condition(config):
                errors.append(f"Required dependency '{dep.feature}' not satisfied")

        # Check config schema
        for key, expected_type in self.config_schema.items():
            if key in config and not isinstance(config[key], expected_type):
                errors.append(
                    f"Config key '{key}' should be {expected_type}, got {type(config[key])}"
                )

        return len(errors) == 0, errors


# Feature-Definitionen für Phase 2
PHASE2_FEATURES: dict[str, FeatureDefinition] = {
    "xsa": FeatureDefinition(
        name="Cross-Sequence Attention",
        feature_type=FeatureType.XSA,
        description="Attention über Sequenzgrenzen hinweg für lange Abhängigkeiten",
        default_status=FeatureStatus.DISABLED,
        dependencies=[
            FeatureDependency(
                feature="gqa",  # XSA requires GQA to be enabled
                condition=lambda c: c.get("model", {}).get("attention", {}).get("type")
                in ["gqa", "standard"],
                required=True,
            )
        ],
        config_schema={
            "layers": (str, list),
            "window": int,
        },
        kill_criteria=[
            {"metric": "ms_per_step", "threshold": 30.0, "operator": "increase_percent"},
            {"metric": "val_bpb", "threshold": 0.02, "operator": "min_improvement"},
        ],
    ),
    "film": FeatureDefinition(
        name="Feature-wise Linear Modulation",
        feature_type=FeatureType.FILM,
        description="Kontextabhängige Skalierung und Verschiebung von Aktivierungen",
        default_status=FeatureStatus.DISABLED,
        dependencies=[],
        config_schema={
            "layers": (str, list),
            "cond_dim": int,
        },
        kill_criteria=[
            {"metric": "ms_per_step", "threshold": 25.0, "operator": "increase_percent"},
            {"metric": "artifact_bytes", "threshold": 2_000_000, "operator": "max_increase"},
            {"metric": "val_bpb", "threshold": 0.02, "operator": "min_improvement"},
        ],
    ),
    "ttt": FeatureDefinition(
        name="Test-Time Training",
        feature_type=FeatureType.TTT,
        description="Adaptive Inferenz durch Mini-Updates während der Vorhersage",
        default_status=FeatureStatus.DISABLED,
        dependencies=[],
        config_schema={
            "layers": (str, list),
            "steps": int,
            "lr": float,
        },
        kill_criteria=[
            {"metric": "ms_per_step", "threshold": 50.0, "operator": "increase_percent"},
            {"metric": "val_bpb", "threshold": 0.03, "operator": "min_improvement"},
        ],
    ),
    "gqa": FeatureDefinition(
        name="Grouped Query Attention",
        feature_type=FeatureType.GQA,
        description="Reduziert KV-Cache durch geteilte KV-Heads",
        default_status=FeatureStatus.DISABLED,
        dependencies=[],
        config_schema={
            "gqa_groups": int,
            "kv_sharing": bool,
        },
        kill_criteria=[
            {"metric": "val_bpb", "threshold": 0.04, "operator": "max_regression"},
        ],
    ),
    "recurrence": FeatureDefinition(
        name="Recurrent Blocks",
        feature_type=FeatureType.RECURRENCE,
        description="Recurrente Verarbeitung für effiziente Sequenzmodellierung",
        default_status=FeatureStatus.DISABLED,
        dependencies=[],
        config_schema={
            "type": str,
            "depth": int,
            "loop_embeddings": bool,
        },
        kill_criteria=[
            {"metric": "ms_per_step", "threshold": 40.0, "operator": "increase_percent"},
            {"metric": "val_bpb", "threshold": 0.02, "operator": "min_improvement_long_seq"},
        ],
    ),
    "gated_mlp": FeatureDefinition(
        name="Gated MLP",
        feature_type=FeatureType.GATED_MLP,
        description="Gated MLP (SwiGLU/GeGLU) für höhere Expressivität",
        default_status=FeatureStatus.DISABLED,
        dependencies=[],
        config_schema={
            "type": str,  # swiglu, geglu
        },
        kill_criteria=[
            {"metric": "ms_per_step", "threshold": 30.0, "operator": "increase_percent"},
            {"metric": "artifact_bytes", "threshold": 0.3, "operator": "max_increase_percent"},
            {"metric": "val_bpb", "threshold": 0.02, "operator": "min_improvement"},
        ],
    ),
    "leaky_relu": FeatureDefinition(
        name="LeakyReLU² Aktivierung",
        feature_type=FeatureType.LEAKY_RELU_SQUARED,
        description="LeakyReLU² für bessere BPB/Throughput-Tradeoff",
        default_status=FeatureStatus.DISABLED,
        dependencies=[],
        config_schema={
            "leakiness": float,
        },
        kill_criteria=[
            {"metric": "val_bpb", "threshold": 0.05, "operator": "max_regression"},
        ],
    ),
    "star_relu": FeatureDefinition(
        name="Star-ReLU Aktivierung",
        feature_type=FeatureType.STAR_RELU,
        description="Star-ReLU / ReLU² für bessere numerische Stabilität",
        default_status=FeatureStatus.DISABLED,
        dependencies=[],
        config_schema={
            "beta": float,
        },
        kill_criteria=[
            {"metric": "val_bpb", "threshold": 0.04, "operator": "max_regression"},
        ],
    ),
}


class FeatureGateManager:
    """Manager für Feature-Gates.

    Verwaltet den Status und die Konfiguration aller Features.
    """

    def __init__(self):
        self.features: dict[str, FeatureConfig] = {}
        self._initialize_default_features()

    def _initialize_default_features(self) -> None:
        """Initialisiere Default-Features."""
        for name, definition in PHASE2_FEATURES.items():
            self.features[name] = FeatureConfig(
                feature_type=definition.feature_type,
                status=definition.default_status,
            )

    def enable(self, name: str, config: dict[str, Any] | None = None) -> bool:
        """Enable a feature.

        Args:
            name: Feature name
            config: Optional feature configuration

        Returns:
            True if successfully enabled, False otherwise
        """
        if name not in self.features:
            return False

        definition = PHASE2_FEATURES.get(name)
        if definition is None:
            return False

        # Check dependencies
        full_config = self._get_full_config()
        for dep in definition.dependencies:
            # Check if the dependency condition is satisfied
            condition_met = dep.condition(full_config)
            
            # For required dependencies, condition MUST be met
            if dep.required and not condition_met:
                return False
            
            # For optional dependencies that specify a feature name,
            # check if that feature is enabled (if required)
            if dep.feature and dep.required:
                dep_feature_status = self.get_status(dep.feature)
                if dep_feature_status != FeatureStatus.ENABLED:
                    return False

        # Enable feature
        self.features[name].status = FeatureStatus.ENABLED
        if config:
            self.features[name].config = config

        return True

    def disable(self, name: str) -> bool:
        """Disable a feature."""
        if name not in self.features:
            return False

        self.features[name].status = FeatureStatus.DISABLED
        self.features[name].config = {}
        return True

    def kill(self, name: str, reason: str = "") -> bool:
        """Kill a feature (permanent disable)."""
        if name not in self.features:
            return False

        self.features[name].status = FeatureStatus.KILLED
        self.features[name].config = {"kill_reason": reason}
        return True

    def get_status(self, name: str) -> FeatureStatus | None:
        """Get status of a feature."""
        if name not in self.features:
            return None
        return self.features[name].status

    def is_enabled(self, name: str) -> bool:
        """Check if feature is enabled."""
        status = self.get_status(name)
        return status == FeatureStatus.ENABLED if status else False

    def get_config(self, name: str) -> dict[str, Any] | None:
        """Get configuration of a feature."""
        if name not in self.features:
            return None
        return self.features[name].config.copy()

    def _get_full_config(self) -> dict[str, Any]:
        """Get full configuration including all features."""
        config = {}
        for name, feature in self.features.items():
            config[name] = feature.to_dict()
        return config

    def validate_all(self) -> dict[str, tuple[bool, list[str]]]:
        """Validate all enabled features.

        Returns dict of feature_name -> (valid, errors).
        """
        results = {}
        full_config = self._get_full_config()

        for name, feature in self.features.items():
            if feature.is_enabled():
                definition = PHASE2_FEATURES.get(name)
                if definition:
                    valid, errors = definition.validate(full_config)
                    results[name] = (valid, errors)

        return results

    def get_enabled_features(self) -> list[str]:
        """Get list of enabled feature names."""
        return [
            name
            for name, feature in self.features.items()
            if feature.is_enabled()
        ]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all features."""
        return {
            "total": len(self.features),
            "enabled": len(self.get_enabled_features()),
            "disabled": sum(
                1 for f in self.features.values() if f.is_disabled()
            ),
            "killed": sum(
                1 for f in self.features.values() if f.is_killed()
            ),
            "features": {
                name: feature.to_dict()
                for name, feature in self.features.items()
            },
        }


def create_feature_gate_manager() -> FeatureGateManager:
    """Create a new feature gate manager."""
    return FeatureGateManager()


def features_from_config(config: dict[str, Any]) -> FeatureGateManager:
    """Create feature gate manager from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Configured FeatureGateManager
    """
    manager = FeatureGateManager()
    model_cfg = config.get("model", {})

    # Check XSA
    xsa_cfg = model_cfg.get("xsa", {})
    if xsa_cfg.get("enabled", False):
        manager.enable("xsa", xsa_cfg)

    # Check FiLM
    film_cfg = model_cfg.get("film", {})
    if film_cfg.get("enabled", False):
        manager.enable("film", film_cfg)

    # Check TTT
    ttt_cfg = model_cfg.get("ttt", {})
    if ttt_cfg.get("enabled", False):
        manager.enable("ttt", ttt_cfg)

    # Check Gated MLP
    gated_cfg = model_cfg.get("gated_mlp", {})
    if gated_cfg.get("enabled", False):
        manager.enable("gated_mlp", gated_cfg)

    # Check Recurrence
    rec_cfg = model_cfg.get("recurrence", {})
    if rec_cfg.get("enabled", False):
        manager.enable("recurrence", rec_cfg)

    # Check activation features
    activation = model_cfg.get("activation", "gelu")
    if activation == "leaky_relu_squared" or activation == "leaky_relu":
        manager.enable(
            "leaky_relu", {"leakiness": model_cfg.get("leakiness", 0.01)}
        )
    elif activation == "star_relu":
        manager.enable("star_relu", {"beta": model_cfg.get("beta", 0.5)})

    # Check GQA
    attn_cfg = model_cfg.get("attention", {})
    if attn_cfg.get("type") == "gqa":
        manager.enable(
            "gqa",
            {
                "gqa_groups": attn_cfg.get("gqa_groups", 4),
                "kv_sharing": attn_cfg.get("kv_sharing", True),
            },
        )

    return manager
