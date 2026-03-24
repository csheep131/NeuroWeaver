"""Learning rate schedulers."""

import math
from typing import Any


def create_scheduler(
    optimizer: Any,
    scheduler_type: str = "cosine",
    num_warmup_steps: int = 100,
    num_training_steps: int = 10000,
    min_lr_ratio: float = 0.1,
    learning_rate: float = 3e-4,
    **kwargs: Any,
) -> Any:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler ("cosine", "linear", "constant")
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR ratio for cosine/linear schedulers
        **kwargs: Additional scheduler arguments

    Returns:
        The created scheduler (or a wrapper that implements step())
    """
    try:
        import torch
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            ConstantLR,
            SequentialLR,
            LambdaLR,
        )
    except ImportError:
        raise ImportError("PyTorch is required for scheduler creation")

    scheduler_type = scheduler_type.lower()

    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )

    if scheduler_type == "constant":
        main_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=1)
    elif scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=learning_rate * min_lr_ratio,
        )
    elif scheduler_type == "linear":
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=num_training_steps - num_warmup_steps,
        )
    elif scheduler_type == "inverse_sqrt":
        # Inverse square root decay
        def lr_lambda(step: int) -> float:
            if step < num_warmup_steps:
                return (step + 1) / num_warmup_steps
            return math.sqrt(num_warmup_steps / (step + 1))

        main_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Combine warmup and main scheduler
    combined_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[num_warmup_steps],
    )

    return combined_scheduler
