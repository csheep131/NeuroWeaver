"""Training utilities."""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.logging import RunLogger


class ModelProtocol(Protocol):
    """Protocol for models."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def train(self) -> None:
        ...

    def eval(self) -> None:
        ...


class OptimizerProtocol(Protocol):
    """Protocol for optimizers."""

    def step(self) -> None:
        ...

    def zero_grad(self) -> None:
        ...


@dataclass
class TrainConfig:
    """Training configuration."""

    num_steps: int = 1000
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    eval_every: int = 100
    save_every: int = 500
    grad_clip: float | None = 1.0
    ema_decay: float | None = None
    device: str = "auto"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        """Create from dictionary."""
        return cls(
            num_steps=d.get("num_steps", 1000),
            batch_size=d.get("batch_size", 32),
            learning_rate=d.get("learning_rate", 3e-4),
            warmup_steps=d.get("warmup_steps", 100),
            eval_every=d.get("eval_every", 100),
            save_every=d.get("save_every", 500),
            grad_clip=d.get("grad_clip", 1.0),
            ema_decay=d.get("ema_decay"),
            device=d.get("device", "auto"),
        )

    def get_device(self) -> torch.device:
        """Get torch device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)


@dataclass
class TrainState:
    """Current training state."""

    step: int = 0
    best_loss: float = float("inf")
    total_time_seconds: float = 0.0
    ms_per_step: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "best_loss": self.best_loss,
            "total_time_seconds": self.total_time_seconds,
            "ms_per_step": self.ms_per_step,
        }


class Trainer:
    """Trainer for models."""

    def __init__(
        self,
        model: nn.Module | ModelProtocol,
        optimizer: OptimizerProtocol,
        config: TrainConfig,
        logger: RunLogger,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        self.state = TrainState()

        # Device
        self.device = config.get_device()
        if isinstance(self.model, nn.Module):
            self.model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, batch: Any) -> float:
        """Perform a single training step.

        Args:
            batch: A batch of data (tokens, targets, mask)

        Returns:
            Loss for this step
        """
        # Check if batch is a Batch object from data loader
        if hasattr(batch, 'tokens') and hasattr(batch, 'targets'):
            tokens = batch.tokens
            targets = batch.targets
            mask = getattr(batch, 'mask', None)  # FIX: Extract mask if available
        else:
            # Fallback for tuple format
            tokens, targets = batch[0], batch[1]
            mask = batch[2] if len(batch) > 2 else None  # FIX: Extract mask from tuple

        # Convert to torch tensors if numpy
        if hasattr(tokens, 'numpy'):
            import numpy as np
            tokens = torch.from_numpy(tokens).long()
            targets = torch.from_numpy(targets).long()
            if mask is not None:
                mask = torch.from_numpy(mask).float()
        elif not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens).long()
            targets = torch.tensor(targets).long()
            if mask is not None:
                mask = torch.tensor(mask).float()

        # Move to device
        tokens = tokens.to(self.device)
        targets = targets.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Training mode
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        if isinstance(self.model, nn.Module):
            # PyTorch model - returns (logits, loss) tuple
            output = self.model(tokens)
            # Handle tuple return (logits, loss)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # FIX: Compute loss with mask support
            # Flatten logits and targets
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            if mask is not None:
                # Apply mask: set masked positions to -1 (ignored in loss)
                mask_flat = mask.view(-1)
                targets_flat = targets_flat.clone()
                targets_flat[mask_flat == 0] = -1  # -1 is ignore_index
            
            loss = self.criterion(
                logits_flat,
                targets_flat
            )
        else:
            # Fallback: use model's forward method
            output = self.model.forward(tokens)
            # Handle tuple return
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # FIX: Same masking logic for fallback
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            if mask is not None:
                mask_flat = mask.view(-1)
                targets_flat = targets_flat.clone()
                targets_flat[mask_flat == 0] = -1
            
            loss = self.criterion(
                logits_flat,
                targets_flat
            )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

        # Optimizer step
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        train_loader: Any,
        eval_fn: callable | None = None,
    ) -> TrainState:
        """Run training loop.

        Args:
            train_loader: Data loader for training data
            eval_fn: Optional evaluation function to call periodically

        Returns:
            Final TrainState
        """
        self.logger.log_info(f"Starting training for {self.config.num_steps} steps")
        start_time = time.time()

        # Use deque with maxlen to limit memory usage
        step_times = collections.deque(maxlen=1000)

        for step in range(self.config.num_steps):
            step_start = time.perf_counter()

            # Get batch (placeholder - assumes iterable)
            try:
                batch = next(train_loader)
            except (StopIteration, TypeError):
                # Reset loader if exhausted
                train_loader = iter(train_loader)
                batch = next(train_loader)

            # Training step
            loss = self.train_step(batch)

            step_time = (time.perf_counter() - step_start) * 1000
            step_times.append(step_time)

            # Update state
            self.state.step = step + 1
            self.state.best_loss = min(self.state.best_loss, loss)

            # Log step
            if step % 10 == 0:
                self.logger.log_step(step, {"loss": loss, "step_time_ms": step_time})

            # Evaluation
            if eval_fn and step % self.config.eval_every == 0:
                self.logger.log_info(f"Running evaluation at step {step}")
                eval_metrics = eval_fn(self.model, step)
                self.logger.log_eval(step, eval_metrics)

            # Progress update
            if step % 100 == 0:
                # Calculate average from last 100 steps efficiently
                window_size = min(len(step_times), 100)
                if window_size > 0:
                    # Use list slicing on deque for the last 100 elements
                    recent_times = list(step_times)[-window_size:]
                    avg_step_time = sum(recent_times) / window_size
                    self.state.ms_per_step = avg_step_time
                    elapsed = time.time() - start_time
                    eta = elapsed / (step + 1) * (self.config.num_steps - step - 1)
                    self.logger.log_info(
                        f"Step {step}/{self.config.num_steps} | "
                        f"loss: {loss:.4f} | "
                        f"ms/step: {avg_step_time:.2f} | "
                        f"ETA: {eta:.0f}s"
                    )

        # Final state
        self.state.total_time_seconds = time.time() - start_time
        if step_times:
            self.state.ms_per_step = sum(step_times) / len(step_times)

        self.logger.log_info(
            f"Training completed: {self.state.step} steps in {self.state.total_time_seconds:.1f}s"
        )
        self.logger.log_info(f"Best loss: {self.state.best_loss:.4f}")
        self.logger.log_info(f"Average ms/step: {self.state.ms_per_step:.2f}")

        return self.state

    def save_checkpoint(self, path: str | Path) -> None:
        """Save a checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Placeholder - actual implementation depends on framework
        self.logger.log_info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a checkpoint."""
        path = Path(path)
        # Placeholder - actual implementation depends on framework
        self.logger.log_info(f"Checkpoint loaded from {path}")
