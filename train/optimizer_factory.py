"""Optimizer factory for creating optimizers."""

from typing import Any


def create_optimizer(
    model: Any,
    optimizer_type: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    **kwargs: Any,
) -> Any:
    """Create an optimizer for the model.

    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
        learning_rate: Learning rate
        weight_decay: Weight decay
        beta1: Beta1 for Adam optimizers
        beta2: Beta2 for Adam optimizers
        eps: Epsilon for numerical stability
        **kwargs: Additional optimizer arguments

    Returns:
        The created optimizer
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for optimizer creation")

    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps,
            **kwargs,
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            betas=(beta1, beta2),
            eps=eps,
            **kwargs,
        )
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get("nesterov", True),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
