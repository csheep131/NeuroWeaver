"""Configuration loading and management."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Configuration container with hash computation."""

    run_id: str = ""
    parent_run_id: str | None = None
    seed: int = 42

    # Model config
    model: dict[str, Any] = field(default_factory=dict)

    # Tokenizer config
    tokenizer: dict[str, Any] = field(default_factory=dict)

    # Training config
    training: dict[str, Any] = field(default_factory=dict)

    # Eval config
    eval: dict[str, Any] = field(default_factory=dict)

    # Quantization config
    quant: dict[str, Any] = field(default_factory=dict)

    # Feature gates
    features: dict[str, Any] = field(default_factory=dict)

    # Raw config for hashing
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def config_hash(self) -> str:
        """Compute a hash of the configuration."""
        if not hasattr(self, '_cached_hash'):
            config_str = json.dumps(self._raw, sort_keys=True)
            self._cached_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return self._cached_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return self._raw.copy()

    def save(self, path: str | Path) -> None:
        """Save resolved config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._raw, f, default_flow_style=False)


def load_base_config(base_path: str | Path = "configs/base.yaml") -> Config:
    """Load the base configuration."""
    path = Path(base_path)
    if not path.exists():
        # Return empty config if base doesn't exist
        return Config()

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f) or {}

    return _parse_config(raw_config)


def load_config(config_path: str | Path) -> Config:
    """Load a run configuration, merging with base config."""
    config_path = Path(config_path)

    # Load run-specific config
    with open(config_path, "r") as f:
        run_config = yaml.safe_load(f) or {}

    # Load base config and merge
    base_config = load_base_config()
    merged = _deep_merge(base_config._raw, run_config)

    return _parse_config(merged)


def merge_configs(base: Config, override: dict[str, Any]) -> Config:
    """Merge a base config with overrides."""
    merged = _deep_merge(base._raw, override)
    return _parse_config(merged)


def _parse_config(raw: dict[str, Any]) -> Config:
    """Parse raw dict into Config dataclass."""
    config = Config(
        run_id=raw.get("run_id", ""),
        parent_run_id=raw.get("parent_run_id"),
        seed=raw.get("seed", 42),
        model=raw.get("model", {}),
        tokenizer=raw.get("tokenizer", {}),
        training=raw.get("training", {}),
        eval=raw.get("eval", {}),
        quant=raw.get("quant", {}),
        features=raw.get("features", {}),
        _raw=raw,
    )
    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
