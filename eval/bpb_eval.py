"""BPB (Bits Per Byte) evaluation."""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F


class TokenizerProtocol(Protocol):
    """Protocol for tokenizers."""

    def encode(self, text: str) -> list[int]:
        """Encode text to tokens."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens to text."""
        ...

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...


@dataclass
class EvalResult:
    """Result of BPB evaluation."""

    val_bpb: float
    total_bytes: int
    total_bits: int
    samples_evaluated: int
    ms_per_step: float | None = None
    steps_evaluated: int = 0
    val_loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "val_bpb": self.val_bpb,
            "total_bytes": self.total_bytes,
            "total_bits": self.total_bits,
            "samples_evaluated": self.samples_evaluated,
            "ms_per_step": self.ms_per_step,
            "steps_evaluated": self.steps_evaluated,
            "val_loss": self.val_loss,
        }


class BPBEvaluator:
    """Evaluator for Bits Per Byte metric."""

    def __init__(self, tokenizer: TokenizerProtocol):
        self.tokenizer = tokenizer

    def compute_bpb(
        self,
        model: Any,
        data: list[str] | Any,
        batch_size: int = 1,
        sliding_window: int | None = None,
        stride: int | None = None,
        device: str = "auto",
    ) -> EvalResult:
        """Compute BPB on given data.

        Args:
            model: The model to evaluate (should have a forward method returning logits)
            data: List of text samples or data loader
            batch_size: Batch size for evaluation
            sliding_window: Sliding window size (optional)
            stride: Stride for sliding window (optional)
            device: Device for evaluation

        Returns:
            EvalResult with BPB and statistics
        """
        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        device = torch.device(device)

        # Move model to device and eval mode
        if hasattr(model, 'to'):
            model.to(device)
        model.eval()

        total_bits = 0
        total_bytes = 0
        samples_evaluated = 0
        steps_evaluated = 0
        total_time = 0.0
        total_loss = 0.0

        # Check if data is a data loader (has __iter__ and is not a list)
        if hasattr(data, '__iter__') and not isinstance(data, list):
            # Data loader
            for batch in data:
                start_time = time.perf_counter()

                with torch.no_grad():
                    if hasattr(batch, 'tokens') and hasattr(batch, 'targets'):
                        tokens = torch.from_numpy(batch.tokens).long().to(device)
                        targets = torch.from_numpy(batch.targets).long().to(device)
                        num_bytes = batch.num_bytes
                    else:
                        tokens = batch[0].to(device)
                        targets = batch[1].to(device)
                        num_bytes = tokens.numel()

                    # Forward pass (handle tuple output)
                    output = model(tokens)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output

                    # Compute loss (cross-entropy)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )

                    # Compute bits from loss (loss is in nats, convert to bits)
                    bits = loss.item() * num_bytes / math.log(2)

                step_time = time.perf_counter() - start_time
                total_time += step_time

                total_bits += bits
                total_bytes += num_bytes
                total_loss += loss.item()
                steps_evaluated += 1
                samples_evaluated += batch_size

        elif isinstance(data, list):
            # List of text samples
            for sample in data:
                # Count bytes in original text
                sample_bytes = len(sample.encode("utf-8"))
                total_bytes += sample_bytes

                if sliding_window is not None:
                    # Sliding window evaluation
                    windows = self._create_windows(
                        sample, sliding_window, stride or sliding_window // 2
                    )
                    for window in windows:
                        start_time = time.perf_counter()
                        bits = self._compute_bits_for_text(model, window, device)
                        total_time += time.perf_counter() - start_time
                        total_bits += bits
                        steps_evaluated += 1
                else:
                    # Full sample evaluation
                    start_time = time.perf_counter()
                    bits = self._compute_bits_for_text(model, sample, device)
                    total_time += time.perf_counter() - start_time
                    total_bits += bits
                    steps_evaluated += 1

                samples_evaluated += 1
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Compute BPB
        val_bpb = total_bits / total_bytes if total_bytes > 0 else float("inf")

        # Compute ms per step
        ms_per_step = (total_time * 1000 / steps_evaluated) if steps_evaluated > 0 else None

        # Average loss
        val_loss = total_loss / steps_evaluated if steps_evaluated > 0 else None

        return EvalResult(
            val_bpb=val_bpb,
            total_bytes=total_bytes,
            total_bits=total_bits,
            samples_evaluated=samples_evaluated,
            ms_per_step=ms_per_step,
            steps_evaluated=steps_evaluated,
            val_loss=val_loss,
        )

    def _create_windows(self, text: str, window_size: int, stride: int) -> list[str]:
        """Create sliding windows from text."""
        windows = []
        start = 0
        while start < len(text):
            window = text[start : start + window_size]
            if window:  # Only add non-empty windows
                windows.append(window)
            start += stride
        return windows

    def _compute_bits_for_text(self, model: Any, text: str) -> int:
        """Compute bits needed to encode text.

        This is a placeholder implementation. In a real implementation,
        this would use the model's logits to compute cross-entropy.

        BPB = -log2(p(correct_token)) averaged over all bytes
        """
        # For now, use a simple estimation based on vocabulary size
        # Real implementation would use model predictions
        tokens = self.tokenizer.encode(text)
        vocab_size = self.tokenizer.vocab_size

        # Bits per token (uniform distribution baseline)
        bits_per_token = -sum(
            self._estimate_log2_probability(model, tokens, i)
            for i in range(len(tokens))
        )

        # Convert to bits per byte
        text_bytes = len(text.encode("utf-8"))
        return int(bits_per_token * len(text) / max(text_bytes, 1))

    def _estimate_log2_probability(self, model: Any, tokens: list[int], position: int) -> float:
        """Estimate log2 probability of correct token at position.

        Placeholder - would use actual model logits in real implementation.
        """
        # Placeholder: assume uniform distribution
        import math
        vocab_size = self.tokenizer.vocab_size
        return -math.log2(1.0 / vocab_size)
