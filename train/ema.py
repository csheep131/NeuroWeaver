"""Exponential Moving Average for model weights."""

from typing import Any


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: Any, decay: float = 0.9999):
        """Initialize EMA.

        Args:
            model: The model to track
            decay: Decay factor (closer to 1 = more smoothing)
        """
        self.model = model
        self.decay = decay
        self.shadow_params: dict[str, Any] = {}
        self.initialized = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
        self.initialized = True

    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        if not self.initialized:
            self._initialize()
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self) -> None:
        """Apply shadow parameters to the model."""
        if not self.initialized:
            raise RuntimeError("EMA not initialized. Call _initialize() first.")

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow_params[name])

    def restore_original(self, original_params: dict[str, Any]) -> None:
        """Restore original parameters.

        Args:
            original_params: Dictionary of original parameters to restore
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in original_params:
                param.data.copy_(original_params[name])

    def get_state_dict(self) -> dict[str, Any]:
        """Get the EMA shadow parameters as a state dict."""
        return self.shadow_params.copy()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load EMA shadow parameters from a state dict."""
        self.shadow_params = {k: v.clone() for k, v in state_dict.items()}
        self.initialized = True
