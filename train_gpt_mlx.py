"""Train GPT with MLX for Apple Silicon.

This is the MLX version of train_gpt.py, optimized for Apple Silicon (M1/M2/M3).
Use this for local testing and smoke tests before running on H100 GPUs.

Challenge Constraints:
- 16MB artifact size limit (code + compressed model weights)
- 10 minute training time on 8xH100s
- Evaluated by compression on FineWeb validation set (bits per byte)

Usage:
    # Smoke test (200 iterations)
    RUN_ID=mlx_smoke ITERATIONS=200 python train_gpt_mlx.py
    
    # Full training
    RUN_ID=baseline_v1 ITERATIONS=2000 TRAIN_BATCH_TOKENS=8192 python train_gpt_mlx.py
    
    # With validation
    VAL_LOSS_EVERY=200 VAL_BATCH_SIZE=8192 python train_gpt_mlx.py

Environment Variables:
    RUN_ID: Run identifier (default: mlx_baseline)
    ITERATIONS: Number of training iterations (default: 200)
    TRAIN_BATCH_TOKENS: Batch size in tokens (default: 8192)
    VAL_LOSS_EVERY: Evaluate every N steps (default: 0, only at end)
    VAL_BATCH_SIZE: Validation batch size (default: 8192)
    DATA_PATH: Path to FineWeb dataset (default: ./data/datasets/fineweb10B_sp1024/)
    TOKENIZER_PATH: Path to tokenizer (default: ./data/tokenizers/fineweb_1024_bpe.model)
    VOCAB_SIZE: Vocabulary size (default: 1024)
"""

import argparse
import io
import json
import math
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Type aliases for type hints
Array = None  # Will be set below

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    MLX_AVAILABLE = True
    Array = mx.array
except ImportError:
    print("Warning: MLX not installed. Install with: pip install mlx")
    print("This file is only usable on Apple Silicon with MLX installed.")
    MLX_AVAILABLE = False
    mx = None
    nn = None
    optim = None
    tree_flatten = None
    tree_unflatten = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Model and training configuration."""

    # Model architecture (Challenge-optimized: 9L x 384d x 6H for <16MB)
    d_model: int = 384
    num_layers: int = 9
    num_heads: int = 6
    mlp_ratio: int = 4
    vocab_size: int = 1024
    max_seq_len: int = 1024
    use_rope: bool = True
    partial_rope: bool = False
    rope_dim: int = 64  # For partial RoPE

    # Attention
    attention_type: str = "gqa"
    kv_heads: int = 3  # GQA: 6Q, 3KV (2:1 ratio)
    use_xsa: bool = False
    xsa_layers: list[int] = None

    # Activations
    activation: str = "leaky_relu_squared"
    leakiness: float = 0.5

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 200
    batch_tokens: int = 8192
    grad_clip: float = 1.0

    # Evaluation
    val_every: int = 0  # 0 = only at end
    val_batch_size: int = 8192

    # Paths
    data_path: str = "./data/datasets/fineweb10B_sp1024/"
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model"

    # Run metadata
    run_id: str = "mlx_baseline"
    seed: int = 42
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            run_id=os.getenv("RUN_ID", "mlx_baseline"),
            max_steps=int(os.getenv("ITERATIONS", "200")),
            batch_tokens=int(os.getenv("TRAIN_BATCH_TOKENS", "8192")),
            val_every=int(os.getenv("VAL_LOSS_EVERY", "0")),
            val_batch_size=int(os.getenv("VAL_BATCH_SIZE", "8192")),
            data_path=os.getenv("DATA_PATH", "./data/datasets/fineweb10B_sp1024/"),
            tokenizer_path=os.getenv("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"),
            vocab_size=int(os.getenv("VOCAB_SIZE", "1024")),
        )


# ============================================================================
# Model Components (MLX) - Nur verfügbar wenn MLX installiert ist
# ============================================================================

if MLX_AVAILABLE:

    def leaky_relu_squared(x: Array, leakiness: float = 0.01) -> Array:
        """LeakyReLU squared activation."""
        return mx.power(nn.leaky_relu(x, negative_slope=leakiness), 2)

    class Rope(nn.Module):
        """Rotary Positional Embeddings."""

        def __init__(self, dim: int, max_seq_len: int = 1024, partial: bool = False, partial_dim: int = 64):
            super().__init__()
            self.dim = dim
            self.partial = partial
            self.partial_dim = partial_dim if partial else dim

            inv_freq = 1.0 / (10000 ** (mx.arange(0, self.partial_dim, 2).astype(mx.float32) / self.partial_dim))
            self.inv_freq = inv_freq

            self.max_seq_len = max_seq_len
            self._cos_cached = None
            self._sin_cached = None

        def _update_cache(self, seq_len: int):
            if self._cos_cached is None or seq_len > self._cos_cached.shape[0]:
                t = mx.arange(seq_len, dtype=self.inv_freq.dtype)
                freqs = mx.outer(t, self.inv_freq)
                emb = mx.concatenate([freqs, freqs], axis=-1)
                self._cos_cached = mx.cos(emb)
                self._sin_cached = mx.sin(emb)

        def __call__(self, x: mx.array) -> mx.array:
            """Apply RoPE to query/key tensors."""
            self._update_cache(x.shape[1])

            if self.partial:
                # Apply RoPE only to first partial_dim dimensions
                x_rot = x[..., : self.partial_dim]
                x_pass = x[..., self.partial_dim :]

                cos = self._cos_cached[: x.shape[1], :][mx.newaxis, mx.newaxis, :]
                sin = self._sin_cached[: x.shape[1], :][mx.newaxis, mx.newaxis, :]

                x_rotated = x_rot * cos + self._rotate_half(x_rot) * sin
                return mx.concatenate([x_rotated, x_pass], axis=-1)
            else:
                cos = self._cos_cached[: x.shape[1], :][mx.newaxis, mx.newaxis, :]
                sin = self._sin_cached[: x.shape[1], :][mx.newaxis, mx.newaxis, :]
                return x * cos + self._rotate_half(x) * sin

        @staticmethod
        def _rotate_half(x: mx.array) -> mx.array:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return mx.concatenate([-x2, x1], axis=-1)

    class Attention(nn.Module):
        """Multi-head / Grouped Query Attention."""

        def __init__(self, cfg: Config):
            super().__init__()
            self.d_model = cfg.d_model
            self.num_heads = cfg.num_heads
            self.kv_heads = cfg.kv_heads if cfg.attention_type != "mha" else cfg.num_heads
            self.head_dim = cfg.d_model // cfg.num_heads
            self.use_rope = cfg.use_rope

            # QKV projections
            self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.k_proj = nn.Linear(cfg.d_model, self.kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(cfg.d_model, self.kv_heads * self.head_dim, bias=False)
            self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

            # RoPE
            if cfg.use_rope:
                self.rope = Rope(
                    dim=self.head_dim,
                    max_seq_len=cfg.max_seq_len,
                    partial=cfg.partial_rope,
                    partial_dim=cfg.rope_dim,
                )

            # GQA repeats
            self.gqa_repeats = cfg.num_heads // self.kv_heads

        def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
            B, T, C = x.shape

            # QKV projections
            q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = self.k_proj(x).reshape(B, T, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, T, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)

            # Apply RoPE
            if self.use_rope:
                q = self.rope(q)
                k = self.rope(k)

            # GQA: repeat KV heads
            if self.gqa_repeats > 1:
                k = mx.repeat(k, self.gqa_repeats, axis=1)
                v = mx.repeat(v, self.gqa_repeats, axis=1)

            # Scaled dot-product attention
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.swapaxes(-2, -1)) * scale

            if mask is not None:
                attn = mx.where(mask == 0, -1e9, attn)

            attn = mx.softmax(attn, axis=-1)

            # Output
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
            return self.out_proj(out)

    class MLP(nn.Module):
        """Feed-forward network."""

        def __init__(self, cfg: Config):
            super().__init__()
            hidden_dim = cfg.d_model * cfg.mlp_ratio
            self.c_fc = nn.Linear(cfg.d_model, hidden_dim, bias=False)
            self.c_proj = nn.Linear(hidden_dim, cfg.d_model, bias=False)
            self.activation = cfg.activation
            self.leakiness = cfg.leakiness

        def __call__(self, x: mx.array) -> mx.array:
            x = self.c_fc(x)

            if self.activation == "gelu":
                x = nn.gelu(x)
            elif self.activation == "leaky_relu_squared":
                x = leaky_relu_squared(x, self.leakiness)
            elif self.activation == "star_relu":
                x = mx.sqrt(mx.maximum(x, 0)) * 0.5  # Simplified Star-ReLU
            else:
                x = nn.relu(x)

            return self.c_proj(x)

    class Block(nn.Module):
        """Transformer block."""

        def __init__(self, cfg: Config, layer_idx: int):
            super().__init__()
            self.ln_1 = nn.LayerNorm(cfg.d_model)
            self.attn = Attention(cfg)
            self.ln_2 = nn.LayerNorm(cfg.d_model)
            self.mlp = MLP(cfg)

            # XSA for specific layers
            self.use_xsa = cfg.use_xsa and cfg.xsa_layers and layer_idx in cfg.xsa_layers

        def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
            x = x + self.attn(self.ln_1(x), mask)
            x = x + self.mlp(self.ln_2(x))
            return x

    class GPT(nn.Module):
        """GPT model for Parameter Golf Challenge (MLX version)."""

        def __init__(self, cfg: Config):
            super().__init__()
            self.cfg = cfg
            self.d_model = cfg.d_model
            self.vocab_size = cfg.vocab_size

            # Token embeddings
            self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

            # Transformer blocks
            self.blocks = [Block(cfg, i) for i in range(cfg.num_layers)]
            self.ln_f = nn.LayerNorm(cfg.d_model)

        def __call__(self, idx: mx.array, targets: Optional[mx.array] = None) -> tuple:
            B, T = idx.shape

            # Create causal mask
            mask = mx.tril(mx.ones((T, T))).reshape(1, 1, T, T)

            # Forward pass
            x = self.token_embedding(idx)

            for block in self.blocks:
                x = block(x, mask)

            x = self.ln_f(x)

            # Output projection (tied embeddings)
            logits = x @ self.token_embedding.weight.T

            # Compute loss if targets provided
            loss = None
            if targets is not None:
                loss = nn.losses.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                ).mean()

            return logits, loss

        def num_parameters(self) -> int:
            """Get parameter count."""
            return sum(p.size for p in self.parameters())

    # ========================================================================
    # Compression & Evaluation
    # ========================================================================

    def compress_model(model: GPT) -> tuple:
        """Compress model weights and return size.

        Returns:
            Tuple of (compressed_size_bytes, compressed_data)
        """
        # Get parameters
        params = model.parameters()

        # Convert to numpy and quantize to INT8
        quantized = {}
        scales = {}

        for key, value in tree_flatten(params):
            arr = np.array(value)
            if arr.dtype == np.float32:
                # Quantize to int8
                scale = np.abs(arr).max() / 127
                quantized[key] = np.round(arr / scale).clip(-127, 127).astype(np.uint8)
                scales[key] = scale
            else:
                quantized[key] = arr

        # Save to bytes
        buffer = io.BytesIO()
        np.savez(buffer, quantized=quantized, scales=scales)
        quantized_bytes = buffer.getvalue()

        # Compress with zlib
        compressed = zlib.compress(quantized_bytes, level=9)

        return len(compressed), compressed

    def compute_bpb(model: GPT, data_loader, val_batches: int = 10) -> float:
        """Compute bits per byte on validation data."""
        total_loss = 0.0
        num_batches = 0

        for i in range(val_batches):
            x, y = data_loader.get_batch(data_loader.batch_tokens)

            _, loss = model(x, y)

            if loss is not None:
                total_loss += float(loss)
                num_batches += 1

        if num_batches == 0:
            return float("inf")

        # Convert loss (nats) to BPB (bits per byte)
        avg_loss = total_loss / num_batches
        bpb = avg_loss / math.log(2)

        return bpb

    # ========================================================================
    # Training
    # ========================================================================

    def train(config: Config):
        """Main training function."""

        # Set seed
        mx.random.seed(config.seed)

        # Print configuration
        print("=" * 60)
        print("Parameter Golf Challenge - Training (MLX)")
        print("=" * 60)
        print(f"Run ID: {config.run_id}")
        print(f"Device: Apple Silicon")
        print(f"Model: {config.num_layers}L x {config.d_model}d x {config.num_heads}H")
        print(f"Vocab size: {config.vocab_size}")
        print(f"Max steps: {config.max_steps}")
        print(f"Batch tokens: {config.batch_tokens}")
        print("=" * 60)

        # Initialize dataset
        train_data = FineWebDataset(
            data_path=config.data_path,
            tokenizer_path=config.tokenizer_path,
            seq_len=config.max_seq_len,
        )

        # Initialize model
        model = GPT(config)
        num_params = model.num_parameters()

        print(f"Model parameters: {num_params / 1_000_000:.2f}M")
        print(f"Model created")

        # Optimizer
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler with warmup
        def get_lr(step: int) -> float:
            if step < config.warmup_steps:
                return config.learning_rate * step / config.warmup_steps
            return config.learning_rate

        # Training loop
        start_time = time.time()
        step = 0
        best_val_bpb = float("inf")

        print("Starting training...")

        while step < config.max_steps:
            # Get learning rate
            lr = get_lr(step)

            # Get batch
            x, y = train_data.get_batch(config.batch_tokens)

            # Forward pass and compute loss
            def loss_fn(m):
                _, loss = m(x, y)
                return loss

            # Compute gradients
            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad_fn(model)

            # Gradient clipping
            total_norm = 0.0
            for g in tree_flatten(grads):
                total_norm += float((g[1] ** 2).sum())
            total_norm = total_norm ** 0.5

            if total_norm > config.grad_clip:
                scale = config.grad_clip / total_norm
                grads = tree_unflatten([(k, v * scale) for k, v in tree_flatten(grads)])

            # Update parameters
            optimizer.update(model, grads)

            # Logging
            if step % 10 == 0:
                step_time = (time.time() - start_time) / (step + 1) * 1000
                print(f"Step {step}/{config.max_steps} | Loss: {float(loss):.4f} | LR: {lr:.6f} | ms/step: {step_time:.1f}")

            # Validation
            if config.val_every > 0 and step % config.val_every == 0:
                val_bpb = compute_bpb(model, train_data)
                print(f"Step {step} | Val BPB: {val_bpb:.4f}")

                if val_bpb < best_val_bpb:
                    best_val_bpb = val_bpb

            step += 1

        # Final evaluation
        print("\n" + "=" * 60)
        print("Training completed")
        print("=" * 60)

        total_time = time.time() - start_time
        print(f"Total steps: {step}")
        print(f"Total time: {total_time:.1f}s")
        print(f"ms/step: {total_time / step * 1000:.1f}")

        # Final validation
        val_bpb = compute_bpb(model, train_data)
        print(f"Final Val BPB: {val_bpb:.4f}")

        # Compression
        compressed_size, _ = compress_model(model)
        print(f"Compressed model size: {compressed_size / 1_000_000:.2f} MB")

        # Final metrics
        print("\n" + "=" * 60)
        print("FINAL METRICS")
        print("=" * 60)
        print(f"val_bpb: {val_bpb:.4f}")
        print(f"compressed_size_bytes: {compressed_size}")
        print(f"meets_16mb_limit: {compressed_size < 16_000_000}")
        print("=" * 60)

    def main():
        """Main entry point."""
        parser = argparse.ArgumentParser(description="Train GPT with MLX for Parameter Golf Challenge")
        parser.add_argument("--run_id", type=str, default=None, help="Run identifier")
        parser.add_argument("--config", type=str, default=None, help="Path to config file (JSON)")

        args = parser.parse_args()

        # Load configuration
        config = Config.from_env()

        # Override from arguments
        if args.run_id:
            config.run_id = args.run_id

        # Load from config file if provided
        if args.config:
            with open(args.config) as f:
                config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Start training
        train(config)


# ============================================================================
# Data Loading
# ============================================================================

class FineWebDataset:
    """FineWeb dataset for training (MLX version)."""
    
    def __init__(self, data_path: str, tokenizer_path: str, seq_len: int = 1024):
        self.data_path = Path(data_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.seq_len = seq_len
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Load data shards
        self.shards = self._load_shards()
        self.current_shard = 0
        self.current_pos = 0
    
    def _load_tokenizer(self):
        """Load SentencePiece tokenizer."""
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(str(self.tokenizer_path))
        except ImportError:
            print("Warning: sentencepiece not installed, using byte-level fallback")
            self.sp = None
    
    def _load_shards(self) -> list:
        """Load data shards from directory."""
        if not self.data_path.exists():
            print(f"Warning: Data path {self.data_path} does not exist")
            return []
        
        shards = sorted(self.data_path.glob("*.bin"))
        print(f"Found {len(shards)} data shards in {self.data_path}")
        return shards
    
    def _load_shard(self, shard_idx: int) -> np.ndarray:
        """Load a single shard."""
        if shard_idx >= len(self.shards):
            # Cycle back to first shard
            shard_idx = shard_idx % len(self.shards)
        
        shard_path = self.shards[shard_idx]
        data = np.memmap(str(shard_path), dtype=np.uint16, mode="r")
        return data
    
    def get_batch(self, batch_tokens: int) -> tuple:
        """Get a batch of tokens."""
        # Calculate tokens per sequence
        tokens_per_seq = self.seq_len + 1  # +1 for target
        num_sequences = batch_tokens // tokens_per_seq

        # Load data
        if not self.shards:
            # Generate random data for testing
            if not MLX_AVAILABLE:
                raise RuntimeError("MLX not available for data generation")
            tokens = mx.random.randint(0, self.vocab_size, (num_sequences, self.seq_len + 1))
        else:
            # Load from shards
            data = self._load_shard(self.current_shard)

            # Get batch from current position
            start_idx = self.current_pos
            end_idx = start_idx + num_sequences * tokens_per_seq

            if end_idx > len(data):
                # Move to next shard
                self.current_shard += 1
                self.current_pos = 0
                return self.get_batch(batch_tokens)

            if not MLX_AVAILABLE:
                raise RuntimeError("MLX not available for data loading")
            tokens = mx.array(data[start_idx:end_idx].astype(np.int32))
            self.current_pos = end_idx

        # Reshape
        tokens = tokens.reshape(num_sequences, self.seq_len + 1)

        # Split into inputs and targets
        x = tokens[:, :-1]
        y = tokens[:, 1:]

        return x, y
    
    @property
    def vocab_size(self) -> int:
        if self.sp:
            return self.sp.GetPieceSize()
        return 256  # Byte-level fallback


# ============================================================================
# MLX Training Functions - Nur verfügbar wenn MLX installiert ist
# ============================================================================

if not MLX_AVAILABLE:
    def _mlx_not_available():
        """Raise error when MLX is not available."""
        raise RuntimeError(
            "MLX is not installed. Install with: pip install mlx\n"
            "This file requires MLX for training and model operations."
        )

    # Stub functions to provide clear error messages
    def compress_model(*args, **kwargs):  # type: ignore
        """Compress model weights - requires MLX."""
        _mlx_not_available()

    def compute_bpb(*args, **kwargs):  # type: ignore
        """Compute bits per byte - requires MLX."""
        _mlx_not_available()

    def train(*args, **kwargs):  # type: ignore
        """Train model - requires MLX."""
        _mlx_not_available()

    def main():  # type: ignore
        """Main entry point - requires MLX."""
        _mlx_not_available()


if __name__ == "__main__":
    main()
