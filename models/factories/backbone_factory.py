"""Backbone Factory for creating model variants.

This module provides a declarative model builder that creates
backbone architectures based on configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.config import Config

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Protocol for model instances."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def train(self) -> None:
        ...

    def eval(self) -> None:
        ...


@dataclass
class ArchitectureConfig:
    """Architecture configuration for backbone."""

    # Basic architecture
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: int = 4
    max_seq_len: int = 1024
    vocab_size: int = 256

    # Recurrence
    recurrence_enabled: bool = False
    recurrence_depth: int = 1
    recurrence_tied: bool = False
    recurrence_type: str = "tied"
    loop_embeddings: bool = False

    # Attention
    attention_type: str = "gqa"  # gqa, mha, mqa, none
    kv_heads: int = 4
    gqa_groups: int = 4
    kv_sharing: bool = True
    use_rope: bool = True
    partial_rope: bool = False

    # Activations
    activation: str = "gelu"  # gelu, leaky_relu, leaky_relu_squared, star_relu, gated
    leakiness: float = 0.01  # Für LeakyReLU
    beta: float = 0.5  # Für Star-ReLU

    # Feature gates
    xsa_enabled: bool = False
    xsa_layers: list[int] | str | None = None  # None = all, "last_N", oder Liste
    xsa_window: int = 2048
    film_enabled: bool = False
    film_layers: str | list[int] = "all"
    film_cond_dim: int = 64
    ttt_enabled: bool = False
    ttt_layers: str | list[int] = "last_2"
    ttt_steps: int = 1
    ttt_lr: float = 0.0001
    gated_mlp_enabled: bool = False
    gated_mlp_type: str = "swiglu"  # swiglu, geglu

    @classmethod
    def from_config(cls, config: Config) -> "ArchitectureConfig":
        """Create from main Config object."""
        model_cfg = config.model
        return cls(
            d_model=model_cfg.get("d_model", 512),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=model_cfg.get("num_heads", 8),
            mlp_ratio=model_cfg.get("mlp_ratio", 4),
            max_seq_len=model_cfg.get("max_seq_len", 1024),
            vocab_size=model_cfg.get("vocab_size", 256),
            recurrence_enabled=model_cfg.get("recurrence", {}).get("enabled", False),
            recurrence_depth=model_cfg.get("recurrence", {}).get("depth", 1),
            recurrence_tied=model_cfg.get("recurrence", {}).get("tied", False),
            recurrence_type=model_cfg.get("recurrence", {}).get("type", "tied"),
            loop_embeddings=model_cfg.get("recurrence", {}).get("loop_embeddings", False),
            attention_type=model_cfg.get("attention", {}).get("type", "gqa"),
            kv_heads=model_cfg.get("attention", {}).get("kv_heads", 4),
            gqa_groups=model_cfg.get("attention", {}).get("gqa_groups", 4),
            kv_sharing=model_cfg.get("attention", {}).get("kv_sharing", True),
            use_rope=model_cfg.get("attention", {}).get("rope", True),
            partial_rope=model_cfg.get("attention", {}).get("partial_rope", False),
            activation=model_cfg.get("activation", "gelu"),
            leakiness=model_cfg.get("leakiness", 0.01),
            beta=model_cfg.get("beta", 0.5),
            xsa_enabled=model_cfg.get("xsa", {}).get("enabled", False),
            xsa_layers=model_cfg.get("xsa", {}).get("layers"),
            xsa_window=model_cfg.get("xsa", {}).get("window", 2048),
            film_enabled=model_cfg.get("film", {}).get("enabled", False),
            film_layers=model_cfg.get("film", {}).get("layers", "all"),
            film_cond_dim=model_cfg.get("film", {}).get("cond_dim", 64),
            ttt_enabled=model_cfg.get("ttt", {}).get("enabled", False),
            ttt_layers=model_cfg.get("ttt", {}).get("layers", "last_2"),
            ttt_steps=model_cfg.get("ttt", {}).get("steps", 1),
            ttt_lr=model_cfg.get("ttt", {}).get("lr", 0.0001),
            gated_mlp_enabled=model_cfg.get("gated_mlp", {}).get("enabled", False),
            gated_mlp_type=model_cfg.get("gated_mlp", {}).get("type", "swiglu"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "recurrence_enabled": self.recurrence_enabled,
            "recurrence_depth": self.recurrence_depth,
            "recurrence_tied": self.recurrence_tied,
            "recurrence_type": self.recurrence_type,
            "loop_embeddings": self.loop_embeddings,
            "attention_type": self.attention_type,
            "kv_heads": self.kv_heads,
            "gqa_groups": self.gqa_groups,
            "kv_sharing": self.kv_sharing,
            "use_rope": self.use_rope,
            "partial_rope": self.partial_rope,
            "activation": self.activation,
            "leakiness": self.leakiness,
            "beta": self.beta,
            "xsa_enabled": self.xsa_enabled,
            "xsa_layers": self.xsa_layers,
            "xsa_window": self.xsa_window,
            "film_enabled": self.film_enabled,
            "film_layers": self.film_layers,
            "film_cond_dim": self.film_cond_dim,
            "ttt_enabled": self.ttt_enabled,
            "ttt_layers": self.ttt_layers,
            "ttt_steps": self.ttt_steps,
            "ttt_lr": self.ttt_lr,
            "gated_mlp_enabled": self.gated_mlp_enabled,
            "gated_mlp_type": self.gated_mlp_type,
        }


@dataclass
class ModelSpec:
    """Specification for model creation."""

    arch_config: ArchitectureConfig
    feature_gates: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    @property
    def head_dim(self) -> int:
        """Get head dimension."""
        return self.arch_config.d_model // self.arch_config.num_heads

    @property
    def d_ff(self) -> int:
        """Get feedforward dimension."""
        return self.arch_config.d_model * self.arch_config.mlp_ratio

    @property
    def num_parameters_estimate(self) -> int:
        """Estimate total parameter count."""
        cfg = self.arch_config

        # Embedding
        embed_params = cfg.vocab_size * cfg.d_model

        # Attention per layer
        qkv_params = cfg.d_model * (cfg.d_model + 2 * cfg.kv_heads * self.head_dim)
        out_params = cfg.d_model * cfg.d_model
        attn_params_per_layer = qkv_params + out_params

        # MLP per layer
        mlp_params_per_layer = cfg.d_model * self.d_ff * 2

        # Total
        total = embed_params
        total += cfg.num_layers * (attn_params_per_layer + mlp_params_per_layer)

        # Add for features
        if cfg.xsa_enabled:
            total += cfg.num_layers * cfg.d_model * cfg.d_model

        if cfg.film_enabled:
            total += cfg.num_layers * cfg.d_model * 2

        return total

    @property
    def num_parameters_millions(self) -> float:
        """Get parameter count in millions."""
        return self.num_parameters_estimate / 1_000_000.0

    def summary(self) -> str:
        """Get human-readable summary."""
        cfg = self.arch_config
        features = []
        if cfg.xsa_enabled:
            features.append("XSA")
        if cfg.film_enabled:
            features.append("FiLM")
        if cfg.ttt_enabled:
            features.append("TTT")
        if cfg.recurrence_enabled:
            features.append(f"Recurrence(d={cfg.recurrence_depth})")

        features_str = ", ".join(features) if features else "None"

        return (
            f"ModelSpec: {cfg.d_model}d x {cfg.num_layers}L x {cfg.num_heads}H | "
            f"{self.num_parameters_millions:.2f}M params | "
            f"Activation: {cfg.activation} | Attention: {cfg.attention_type} | "
            f"Features: {features_str}"
        )


class BackboneFactory:
    """Factory for creating backbone models.

    This factory creates models declaratively from configuration,
    supporting various architecture variants and feature gates.
    """

    def __init__(self, use_rust: bool = True):
        """Initialize factory.

        Args:
            use_rust: Whether to use Rust implementations when available
        """
        self.use_rust = use_rust
        self._created_models: list[ModelSpec] = []

    def create(self, config: Config | ArchitectureConfig | dict) -> Any:
        """Create a backbone model from configuration.

        Args:
            config: Either a full Config, ArchitectureConfig, or config dict

        Returns:
            Created model instance
        """
        # Handle dict input
        if isinstance(config, dict):
            from core.config import _parse_config
            config_obj = _parse_config(config)
            arch_config = ArchitectureConfig.from_config(config_obj)
        elif isinstance(config, Config):
            arch_config = ArchitectureConfig.from_config(config)
        else:
            # Assume it's an ArchitectureConfig
            arch_config = config

        spec = ModelSpec(arch_config=arch_config)
        self._created_models.append(spec)

        # Try to use Rust implementation
        if self.use_rust:
            try:
                import rust_core
            except (ImportError, AttributeError) as e:
                logger.warning(
                    f"Rust module 'rust_core' not available or has errors: {e}. "
                    "Falling back to Python implementation."
                )
                return self._create_python_stub(spec)
            
            try:
                rust_config = rust_core.BackboneConfig(
                    d_model=arch_config.d_model,
                    num_layers=arch_config.num_layers,
                    num_heads=arch_config.num_heads,
                    mlp_ratio=arch_config.mlp_ratio,
                    max_seq_len=arch_config.max_seq_len,
                    vocab_size=arch_config.vocab_size,
                    use_rope=arch_config.use_rope,
                    use_xsa=arch_config.xsa_enabled,
                    use_film=arch_config.film_enabled,
                    # Extended Phase 2 features
                    use_ttt=arch_config.ttt_enabled,
                    ttt_layers=arch_config.ttt_layers if isinstance(arch_config.ttt_layers, list) else [arch_config.num_layers - 2, arch_config.num_layers - 1],
                    ttt_steps=arch_config.ttt_steps,
                    ttt_lr=arch_config.ttt_lr,
                    use_gated_mlp=arch_config.gated_mlp_enabled,
                    gated_mlp_type=arch_config.gated_mlp_type,
                    recurrence_enabled=arch_config.recurrence_enabled,
                    recurrence_depth=arch_config.recurrence_depth,
                    recurrence_type=arch_config.recurrence_type,
                    loop_embeddings=arch_config.loop_embeddings,
                    activation=arch_config.activation,
                    leakiness=arch_config.leakiness,
                    beta=arch_config.beta,
                )

                model = rust_core.Backbone(rust_config)
                return model

            except AttributeError as e:
                logger.warning(
                    f"Rust module missing expected attribute: {e}. "
                    "Falling back to Python implementation."
                )
                return self._create_python_stub(spec)

        # Fallback to Python stub
        return self._create_python_stub(spec)

    def _create_python_stub(self, spec: ModelSpec) -> dict[str, Any]:
        """Create a Python stub model for testing.

        In a real implementation, this would create actual PyTorch modules.
        """
        return {
            "spec": spec,
            "config": spec.arch_config.to_dict(),
            "num_parameters": spec.num_parameters_estimate,
            "is_stub": True,
        }

    def get_created_specs(self) -> list[ModelSpec]:
        """Get list of all created model specifications."""
        return self._created_models.copy()

    def validate_config(self, config: ArchitectureConfig) -> list[str]:
        """Validate architecture configuration.

        Returns list of warnings/errors.
        """
        warnings = []

        cfg = config

        # Check head dimension
        if cfg.d_model % cfg.num_heads != 0:
            warnings.append(
                f"d_model ({cfg.d_model}) not divisible by num_heads ({cfg.num_heads})"
            )

        # Check KV heads for GQA
        if cfg.attention_type == "gqa":
            if cfg.num_heads % cfg.kv_heads != 0:
                warnings.append(
                    f"For GQA, num_heads ({cfg.num_heads}) should be "
                    f"divisible by kv_heads ({cfg.kv_heads})"
                )
            # Check GQA groups
            if hasattr(cfg, 'gqa_groups') and cfg.gqa_groups:
                if cfg.num_heads % cfg.gqa_groups != 0:
                    warnings.append(
                        f"For GQA, num_heads ({cfg.num_heads}) should be "
                        f"divisible by gqa_groups ({cfg.gqa_groups})"
                    )

        # Check XSA dependencies
        if cfg.xsa_enabled:
            if cfg.attention_type == "none":
                warnings.append("XSA requires attention to be enabled")
            # Validate xsa_layers format
            if cfg.xsa_layers is not None:
                if isinstance(cfg.xsa_layers, str):
                    if not cfg.xsa_layers.startswith("last_"):
                        warnings.append(
                            f"Invalid xsa_layers format: {cfg.xsa_layers} "
                            "(should be 'last_N' or list)"
                        )

        # Check FiLM config
        if cfg.film_enabled:
            if cfg.film_cond_dim <= 0:
                warnings.append(f"film_cond_dim must be positive, got {cfg.film_cond_dim}")

        # Check TTT config
        if cfg.ttt_enabled:
            if cfg.ttt_steps <= 0:
                warnings.append(f"ttt_steps must be positive, got {cfg.ttt_steps}")
            if cfg.ttt_lr <= 0:
                warnings.append(f"ttt_lr must be positive, got {cfg.ttt_lr}")

        # Check Gated MLP config
        if cfg.gated_mlp_enabled:
            if cfg.gated_mlp_type not in ["swiglu", "geglu"]:
                warnings.append(
                    f"Unknown gated_mlp_type: {cfg.gated_mlp_type} "
                    "(should be 'swiglu' or 'geglu')"
                )

        # Check Recurrence config
        if cfg.recurrence_enabled:
            if cfg.recurrence_depth <= 0:
                warnings.append(
                    f"recurrence_depth must be positive, got {cfg.recurrence_depth}"
                )
            if cfg.recurrence_type not in ["tied", "stacked"]:
                warnings.append(
                    f"Unknown recurrence_type: {cfg.recurrence_type} "
                    "(should be 'tied' or 'stacked')"
                )

        # Check activation
        valid_activations = [
            "gelu", "leaky_relu", "leaky_relu_squared",
            "star_relu", "gated", "relu", "silu"
        ]
        if cfg.activation not in valid_activations:
            warnings.append(f"Unknown activation: {cfg.activation}")

        # Check activation-specific params
        if cfg.activation in ["leaky_relu", "leaky_relu_squared"]:
            if not (0 <= cfg.leakiness <= 1):
                warnings.append(
                    f"leakiness should be in [0, 1], got {cfg.leakiness}"
                )
        elif cfg.activation == "star_relu":
            if not (0 <= cfg.beta <= 1):
                warnings.append(f"beta should be in [0, 1], got {cfg.beta}")

        return warnings


def create_backbone(
    config: Config,
    use_rust: bool = True,
) -> tuple[Any, ModelSpec]:
    """Convenience function to create a backbone.

    Args:
        config: Configuration object
        use_rust: Whether to use Rust implementations

    Returns:
        Tuple of (model, spec)
    """
    factory = BackboneFactory(use_rust=use_rust)
    model = factory.create(config)
    spec = factory.get_created_specs()[-1]
    return model, spec
