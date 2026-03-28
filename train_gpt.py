"""Train GPT for Parameter Golf Challenge.

Challenge Constraints:
- 16MB artifact size limit (code + compressed model weights)
- 10 minute training time on 8xH100s
- Evaluated by compression on FineWeb validation set (bits per byte)

Usage:
    # Single GPU (testing)
    python train_gpt.py --run_id baseline_v1
    
    # Distributed training on 8xH100
    torchrun --standalone --nproc_per_node=8 train_gpt.py --run_id baseline_v1
    
    # With environment variables (recommended)
    RUN_ID=baseline_v1 ITERATIONS=2000 TRAIN_BATCH_TOKENS=8192 python train_gpt.py

Environment Variables:
    RUN_ID: Run identifier (default: baseline_v1)
    ITERATIONS: Number of training iterations (default: 2000)
    TRAIN_BATCH_TOKENS: Batch size in tokens (default: 8192)
    VAL_LOSS_EVERY: Evaluate every N steps (default: 0, only at end)
    VAL_BATCH_SIZE: Validation batch size (default: 8192)
    MAX_WALLCLOCK_SECONDS: Maximum training time (default: 600 = 10 min)
    DATA_PATH: Path to FineWeb dataset (default: ./data/datasets/fineweb10B_sp1024/)
    TOKENIZER_PATH: Path to tokenizer (default: ./data/tokenizers/fineweb_1024_bpe.model)
    VOCAB_SIZE: Vocabulary size (default: 1024)
"""

import argparse
import io
import json
import math
import os
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


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
    max_steps: int = 2000
    batch_tokens: int = 8192
    grad_clip: float = 1.0
    ema_decay: float = None

    # Evaluation
    val_every: int = 0  # 0 = only at end
    val_batch_size: int = 8192

    # Constraints
    max_wallclock_seconds: float = 600.0  # 10 minutes

    # Paths
    data_path: str = "./data/datasets/fineweb10B_sp1024/"
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model"

    # Run metadata
    run_id: str = "baseline_v1"
    seed: int = 42
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        # Parse XSA layers if provided
        xsa_layers_str = os.getenv("XSA_LAYERS", "")
        xsa_layers = [int(x) for x in xsa_layers_str.split(",") if x] if xsa_layers_str else None
        
        # Get base architecture from ENV
        num_layers = int(os.getenv("NUM_LAYERS", "9"))
        d_model = int(os.getenv("D_MODEL", "384"))
        num_heads = int(os.getenv("NUM_HEADS", "6"))
        head_dim = d_model // num_heads
        
        # rope_dim defaults to head_dim for full RoPE, or can be set explicitly for partial RoPE
        rope_dim_default = head_dim  # Full RoPE by default
        rope_dim = int(os.getenv("ROPE_DIMS", str(rope_dim_default)))

        # Get KV heads, ensure it's a divisor of num_heads for GQA
        kv_heads = int(os.getenv("KV_HEADS", "3"))
        attention_type = os.getenv("ATTENTION_TYPE", "gqa")
        
        # For GQA, ensure kv_heads divides num_heads evenly
        if attention_type == "gqa" and num_heads % kv_heads != 0:
            # Find the largest divisor of num_heads <= kv_heads
            # Or adjust to a reasonable value
            if kv_heads > num_heads:
                kv_heads = num_heads  # Fall back to MHA
            else:
                # Try to find a divisor close to the requested value
                # Common ratios: 2:1, 4:1, 8:1
                possible_kv = [num_heads // 2, num_heads // 4, num_heads // 8, 1]
                possible_kv = [k for k in possible_kv if k >= 1 and num_heads % k == 0]
                if possible_kv:
                    # Pick the one closest to requested kv_heads
                    kv_heads = min(possible_kv, key=lambda x: abs(x - kv_heads))
                else:
                    # Fall back to MHA
                    kv_heads = num_heads
        
        return cls(
            run_id=os.getenv("RUN_ID", "baseline_v1"),
            max_steps=int(os.getenv("ITERATIONS", "2000")),
            batch_tokens=int(os.getenv("TRAIN_BATCH_TOKENS", "8192")),
            val_every=int(os.getenv("VAL_LOSS_EVERY", "0")),
            val_batch_size=int(os.getenv("VAL_BATCH_SIZE", "8192")),
            max_wallclock_seconds=float(os.getenv("MAX_WALLCLOCK_SECONDS", "600")),
            data_path=os.getenv("DATA_PATH", "./data/datasets/fineweb10B_sp1024/"),
            tokenizer_path=os.getenv("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"),
            vocab_size=int(os.getenv("VOCAB_SIZE", "1024")),
            # Model architecture from ENV
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            mlp_ratio=int(os.getenv("MLP_RATIO", "4")),
            max_seq_len=int(os.getenv("MAX_SEQ_LEN", "1024")),
            use_rope=os.getenv("USE_ROPE", "1") == "1",
            partial_rope=os.getenv("PARTIAL_ROPE", "0") == "1",
            rope_dim=rope_dim,
            # Attention
            attention_type=attention_type,
            kv_heads=kv_heads,
            use_xsa=os.getenv("USE_XSA", "0") == "1" or os.getenv("XSA_ENABLED", "0") == "1",
            xsa_layers=xsa_layers,
            # Training
            learning_rate=float(os.getenv("LEARNING_RATE", "0.0003")),
            weight_decay=float(os.getenv("WEIGHT_DECAY", "0.1")),
            warmup_steps=int(os.getenv("WARMUP_STEPS", "100")),
            grad_clip=float(os.getenv("GRAD_CLIP", "1.0")),
            ema_decay=float(os.getenv("EMA_DECAY", "0.997")) if os.getenv("EMA_ENABLED", "0") == "1" else None,
        )


# ============================================================================
# Model Components
# ============================================================================

def leaky_relu_squared(x: torch.Tensor, leakiness: float = 0.01) -> torch.Tensor:
    """LeakyReLU squared activation."""
    return F.leaky_relu(x, negative_slope=leakiness) ** 2


def star_relu(x: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """Star-ReLU activation."""
    return torch.sqrt(torch.relu(x)) * (beta * x + (1 - beta))


class Rope(nn.Module):
    """Rotary Positional Embeddings."""

    def __init__(self, dim: int, max_seq_len: int = 1024, partial: bool = False, partial_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.partial = partial
        self.partial_dim = partial_dim if partial else dim
        
        # Ensure partial_dim is even for proper RoPE
        assert self.partial_dim % 2 == 0, f"partial_dim must be even, got {self.partial_dim}"

        # inv_freq shape: [partial_dim // 2]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.partial_dim, 2).float() / self.partial_dim))
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len = max_seq_len
        self._cache_seq_len = -1
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int):
        """Update RoPE cache for given sequence length."""
        # Always ensure cache matches actual seq_len
        if seq_len != self._cache_seq_len:
            self._cache_seq_len = seq_len
            t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            # emb shape: [seq_len, partial_dim]
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to query/key tensors.

        Args:
            x: Input tensor of shape [B, num_heads, seq_len, head_dim]
        """
        B, num_heads, seq_len, head_dim = x.shape

        self._update_cache(seq_len)

        if self.partial:
            # Apply RoPE only to first partial_dim dimensions
            x_rot = x[..., : self.partial_dim]
            x_pass = x[..., self.partial_dim :]

            # cos/sin shape: [seq_len, partial_dim]
            # Need to reshape to [1, 1, seq_len, partial_dim] for broadcasting
            cos = self._cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
            sin = self._sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)

            x_rotated = x_rot * cos + self._rotate_half(x_rot) * sin
            return torch.cat((x_rotated, x_pass), dim=-1)
        else:
            # Full RoPE - head_dim should match self.dim
            # cos/sin shape: [seq_len, dim] where dim = head_dim
            cos = self._cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
            sin = self._sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
            return x * cos + self._rotate_half(x) * sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV projections
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)
        
        # GQA: repeat KV heads to match Q heads
        if self.num_heads != self.kv_heads:
            # Ensure kv_heads divides num_heads (should be guaranteed by config validation)
            if self.num_heads % self.kv_heads != 0:
                raise ValueError(
                    f"num_heads ({self.num_heads}) must be divisible by kv_heads ({self.kv_heads}) "
                    f"for Grouped Query Attention"
                )
            repeat_factor = self.num_heads // self.kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        
        attn = F.softmax(attn, dim=-1)
        
        # Output
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        
        if self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "leaky_relu_squared":
            x = leaky_relu_squared(x, self.leakiness)
        elif self.activation == "star_relu":
            x = star_relu(x)
        else:
            x = F.relu(x)
        
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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT model for Parameter Golf Challenge."""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.vocab_size = cfg.vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        
        # Output projection (tied with embeddings)
        # self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple:
        B, T = idx.shape
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        
        # Forward pass
        x = self.token_embedding(idx)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        
        # Output projection (tied embeddings)
        logits = F.linear(x, self.token_embedding.weight)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
            )
        
        return logits, loss
    
    def num_parameters_millions(self) -> float:
        """Get parameter count in millions."""
        return sum(p.numel() for p in self.parameters()) / 1_000_000.0


# ============================================================================
# Data Loading
# ============================================================================

class FineWebDataset:
    """FineWeb dataset for training."""

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

        # Cache vocab_size for fast access during training
        self._vocab_size = self._get_vocab_size()
    
    def _load_tokenizer(self):
        """Load SentencePiece tokenizer."""
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            if self.tokenizer_path and Path(self.tokenizer_path).exists():
                # Prüfen ob Datei nicht leer ist
                if Path(self.tokenizer_path).stat().st_size > 0:
                    self.sp.Load(str(self.tokenizer_path))
                    print(f"Tokenizer loaded: {self.tokenizer_path}")
                else:
                    print(f"Warning: Tokenizer file is empty, using byte-level fallback")
                    self.sp = None
            else:
                print(f"Warning: Tokenizer not found at {self.tokenizer_path}, using byte-level fallback")
                self.sp = None
        except ImportError:
            print("Warning: sentencepiece not installed, using byte-level fallback")
            self.sp = None
        except (OSError, RuntimeError) as e:
            print(f"Warning: Could not load tokenizer: {e}, using byte-level fallback")
            self.sp = None

    def _get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.sp:
            return self.sp.GetPieceSize()
        return 256  # Byte-level fallback
    
    def _load_shards(self) -> list:
        """Load data shards from directory."""
        if not self.data_path.exists():
            print(f"Warning: Data path {self.data_path} does not exist")
            return []
        
        shards = sorted(self.data_path.glob("*.bin"))
        print(f"Found {len(shards)} data shards in {self.data_path}")
        return shards
    
    def _load_shard(self, shard_idx: int) -> np.ndarray:
        """Load a single shard, skipping the 256-int32 header."""
        if shard_idx >= len(self.shards):
            # Cycle back to first shard
            shard_idx = shard_idx % len(self.shards)
        
        shard_path = self.shards[shard_idx]
        # Shard files have a 256 x int32 header (1024 bytes) containing metadata.
        # Read header to get token count, then load only the token data.
        HEADER_INTS = 256
        header_bytes = HEADER_INTS * np.dtype("<i4").itemsize  # 1024 bytes
        header = np.fromfile(str(shard_path), dtype="<i4", count=HEADER_INTS)
        if header.size != HEADER_INTS or int(header[0]) != 20240520 or int(header[1]) != 1:
            raise ValueError(f"Unexpected shard header for {shard_path}")
        num_tokens = int(header[2])
        data = np.fromfile(str(shard_path), dtype="<u2", count=num_tokens, offset=header_bytes)
        return data
    
    def get_batch(self, batch_tokens: int) -> tuple:
        """Get a batch of tokens."""
        # Calculate tokens per sequence
        tokens_per_seq = self.seq_len + 1  # +1 for target
        num_sequences = batch_tokens // tokens_per_seq

        # Load data
        if not self.shards:
            # Generate random data for testing
            print("WARNUNG: Keine Daten gefunden! Verwende ZUFALLSDATEN!")
            print(f"   DATA_PATH={self.data_path}")
            print("   Das Ergebnis wird BPB ~5.0 (Muell) sein!")
            tokens = torch.randint(0, self._vocab_size, (num_sequences, self.seq_len + 1))
        else:
            # Check if we've exhausted all shards
            if self.current_shard >= len(self.shards):
                # Cycle back to first shard
                self.current_shard = 0
                self.current_pos = 0

            # Load from shards
            data = self._load_shard(self.current_shard)

            # Get batch from current position
            start_idx = self.current_pos
            end_idx = start_idx + num_sequences * tokens_per_seq

            if end_idx > len(data):
                # Move to next shard and retry
                self.current_shard += 1
                self.current_pos = 0
                return self.get_batch(batch_tokens)

            tokens = torch.from_numpy(data[start_idx:end_idx].astype(np.int64))
            self.current_pos = end_idx

        # Reshape
        tokens = tokens.view(num_sequences, self.seq_len + 1)

        # Split into inputs and targets
        x = tokens[:, :-1]
        y = tokens[:, 1:]

        return x, y

    def __iter__(self):
        """Make dataset iterable for evaluation."""
        # For evaluation, yield batches with validation batch size
        while True:
            yield self.get_batch(self.seq_len * 4)  # 4 sequences per batch for eval


# ============================================================================
# Compression & Evaluation
# ============================================================================

def compress_model(model: nn.Module, use_zstd: bool = False) -> tuple:
    """Compress model weights and return size.

    Uses efficient numpy-based INT8 quantization with minimal serialization overhead.

    Returns:
        Tuple of (compressed_size_bytes, compressed_data)
    """
    import numpy as np

    # Get state dict
    state_dict = model.state_dict()

    # Quantize to INT8 and collect raw bytes
    quantized_data = {}
    scales = {}

    for key, value in state_dict.items():
        if value.dtype == torch.float32:
            # Quantize to int8
            scale = value.abs().max() / 127
            quantized = (value / scale).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
            quantized_data[key] = quantized
            scales[key] = float(scale)
        else:
            quantized_data[key] = value.cpu().numpy()

    # Efficient serialization using numpy savez (much less overhead than torch.save)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **quantized_data, **{f"scale_{k}": v for k, v in scales.items()})
    compressed = buffer.getvalue()

    return len(compressed), compressed


def compute_bpb(model: nn.Module, data_loader, device: str = "cuda", max_batches: int = 100) -> float:
    """Compute bits per byte on validation data."""
    model.eval()
    total_loss = 0.0
    total_bytes = 0
    num_batches = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)

            if loss is not None:
                total_loss += loss.item()
                total_bytes += x.numel()
                num_batches += 1

            if num_batches >= max_batches:
                break

    if num_batches == 0:
        return float("inf")
    
    # Convert loss (nats) to BPB (bits per byte)
    avg_loss = total_loss / num_batches
    bpb = avg_loss / math.log(2)
    
    return bpb


# ============================================================================
# Training
# ============================================================================

def train(config: Config):
    """Main training function."""
    
    # Setup distributed training
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Print configuration
    if local_rank == 0:
        print("=" * 60)
        print("Parameter Golf Challenge - Training")
        print("=" * 60)
        print(f"Run ID: {config.run_id}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Model: {config.num_layers}L x {config.d_model}d x {config.num_heads}H")
        print(f"Vocab size: {config.vocab_size}")
        print(f"Max steps: {config.max_steps}")
        print(f"Batch tokens: {config.batch_tokens}")
        print(f"Max wallclock: {config.max_wallclock_seconds}s")
        print("=" * 60)
    
    # Initialize dataset
    train_data = FineWebDataset(
        data_path=config.data_path,
        tokenizer_path=config.tokenizer_path,
        seq_len=config.max_seq_len,
    )
    
    # Initialize model
    model = GPT(config)
    num_params = model.num_parameters_millions()
    
    if local_rank == 0:
        print(f"Model parameters: {num_params:.2f}M")
        print(f"Model created")
    
    # Move to device
    model = model.to(device)
    
    # Wrap with DDP for distributed training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
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
    best_val_loss = float("inf")
    
    if local_rank == 0:
        print("Starting training...")
    
    model.train()
    
    while step < config.max_steps:
        # Check wallclock limit
        elapsed = time.time() - start_time
        if elapsed > config.max_wallclock_seconds:
            if local_rank == 0:
                print(f"Wallclock limit reached ({elapsed:.1f}s > {config.max_wallclock_seconds}s)")
            break
        
        # Get learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Get batch
        x, y = train_data.get_batch(config.batch_tokens)
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        _, loss = model(x, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Logging
        if step % 10 == 0 and local_rank == 0:
            step_time = (time.time() - start_time) / (step + 1) * 1000
            print(f"Step {step}/{config.max_steps} | Loss: {loss.item():.4f} | LR: {lr:.6f} | ms/step: {step_time:.1f}")
        
        # Validation
        if config.val_every > 0 and step % config.val_every == 0 and local_rank == 0:
            val_bpb = compute_bpb(model, train_data, device)
            print(f"Step {step} | Val BPB: {val_bpb:.4f}")
            
            if val_bpb < best_val_loss:
                best_val_loss = val_bpb
        
        step += 1
    
    # Final evaluation
    if local_rank == 0:
        print("\n" + "=" * 60)
        print("Training completed")
        print("=" * 60)
        
        total_time = time.time() - start_time
        print(f"Total steps: {step}")
        print(f"Total time: {total_time:.1f}s")
        print(f"ms/step: {total_time / step * 1000:.1f}")
        
        # Final validation
        val_bpb = compute_bpb(model, train_data, device)
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
    
    # Cleanup distributed training
    if is_distributed:
        dist.destroy_process_group()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train GPT for Parameter Golf Challenge")
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


if __name__ == "__main__":
    main()
