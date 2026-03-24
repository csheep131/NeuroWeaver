"""PyTorch-Modell-Integration für Ablation Machine.

Dieses Modul integriert die Rust-Backbone-Implementierung
mit PyTorch für Training und Evaluation.

Features:
- Rust-Backend in PyTorch Module wrapper
- Forward-Pass mit Logits
- Cross-Entropy Loss
- Gradient Checkpointing (optional)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Konfiguration für PyTorch-Modell."""

    # Architektur
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: int = 4
    vocab_size: int = 256
    max_seq_len: int = 1024

    # Attention
    attention_type: str = "gqa"
    kv_heads: int = 4
    use_rope: bool = True

    # Aktivierung
    activation: str = "gelu"

    # Features
    xsa_enabled: bool = False
    film_enabled: bool = False
    recurrence_enabled: bool = False
    gated_mlp_enabled: bool = False

    # Training
    dropout: float = 0.0
    use_gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        model_cfg = d.get("model", d)
        num_heads = model_cfg.get("num_heads", 8)
        attention_type = model_cfg.get("attention", {}).get("type", "gqa")
        kv_heads = model_cfg.get("attention", {}).get("kv_heads", 4)
        
        # FIX: For standard attention, kv_heads must equal num_heads
        if attention_type == "standard":
            kv_heads = num_heads
        
        return cls(
            d_model=model_cfg.get("d_model", 512),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=num_heads,
            mlp_ratio=model_cfg.get("mlp_ratio", 4),
            vocab_size=model_cfg.get("vocab_size", 256),
            max_seq_len=model_cfg.get("max_seq_len", 1024),
            attention_type=attention_type,
            kv_heads=kv_heads,
            use_rope=model_cfg.get("attention", {}).get("rope", True),
            activation=model_cfg.get("activation", "gelu"),
            xsa_enabled=model_cfg.get("xsa", {}).get("enabled", False),
            film_enabled=model_cfg.get("film", {}).get("enabled", False),
            recurrence_enabled=model_cfg.get("recurrence", {}).get("enabled", False),
            gated_mlp_enabled=model_cfg.get("gated_mlp", {}).get("enabled", False),
            dropout=d.get("dropout", 0.0),
            use_gradient_checkpointing=d.get("gradient_checkpointing", False),
        )


class AblationModel(nn.Module):
    """PyTorch-Modell für Ablation Machine.

    Wrapper um Rust-Backbone oder native PyTorch-Implementierung.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Versuche Rust-Backend zu laden
        self.use_rust = False
        self.rust_backbone = None

        try:
            import rust_core

            # Prüfe ob Backbone verfügbar ist (und nicht nur Stub-Klassen)
            if hasattr(rust_core, "Backbone") and hasattr(rust_core, "BackboneConfig"):
                # Teste ob es echte Implementierungen sind (keine Stubs)
                try:
                    rust_config = rust_core.BackboneConfig(
                        d_model=config.d_model,
                        num_layers=config.num_layers,
                        num_heads=config.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        max_seq_len=config.max_seq_len,
                        vocab_size=config.vocab_size,
                        use_rope=config.use_rope,
                        use_xsa=config.xsa_enabled,
                        use_film=config.film_enabled,
                    )
                    self.rust_backbone = rust_core.Backbone(rust_config)
                    self.use_rust = True
                except RuntimeError:
                    # Rust-Backend ist nur ein Stub (nicht kompiliert)
                    pass
        except (ImportError, AttributeError) as e:
            print(f"Rust-Backend nicht verfügbar: {e}. Verwende PyTorch-Implementierung.")

        # PyTorch-Implementierung
        if not self.use_rust:
            self._init_pytorch_model()

    def _init_pytorch_model(self):
        """Initialisiere PyTorch-Modell."""
        cfg = self.config

        # Embedding
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # Transformer-Blöcke
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )

        # Layer-Norm
        self.norm = nn.LayerNorm(cfg.d_model)

        # Output-Projection
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Initialisierung
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Gewichte initialisieren."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward-Pass.

        Args:
            tokens: Input-Token-IDs (batch_size, seq_len)
            targets: Optional Target-Token-IDs für Loss-Berechnung

        Returns:
            Tuple of (logits, loss)
        """
        batch_size, seq_len = tokens.shape

        if self.use_rust and self.rust_backbone:
            # Rust-Backend Forward-Pass
            # Hinweis: Rust-Backend muss als torch.autograd.Function gewrapped werden
            # für korrekte Gradientenberechnung
            logits = self._rust_forward(tokens)
        else:
            # PyTorch-Forward-Pass
            logits = self._pytorch_forward(tokens)

        # Loss berechnen
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def _rust_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Rust-Backend Forward-Pass."""
        # TODO: Rust-Backend als torch.autograd.Function implementieren
        # Für jetzt: Placeholder der native PyTorch-Logik verwendet
        return self._pytorch_forward(tokens)

    def _pytorch_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """PyTorch-Forward-Pass."""
        device = tokens.device
        batch_size, seq_len = tokens.shape

        # Embeddings
        tok_emb = self.token_embedding(tokens)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=device))
        x = tok_emb + pos_emb

        # Transformer-Blöcke
        for layer in self.layers:
            x = layer(x)

        # Layer-Norm
        x = self.norm(x)

        # Output-Projection
        logits = self.output_proj(x)

        return logits

    def num_parameters(self) -> int:
        """Anzahl der Parameter."""
        return sum(p.numel() for p in self.parameters())

    def num_parameters_millions(self) -> float:
        """Anzahl der Parameter in Millionen."""
        return self.num_parameters() / 1_000_000.0


class TransformerBlock(nn.Module):
    """Transformer-Block mit Attention und MLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Attention
        self.attention = SelfAttention(config)

        # MLP
        if config.gated_mlp_enabled:
            self.mlp = GatedMLP(config)
        else:
            self.mlp = StandardMLP(config)

        # Layer-Norm
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass."""
        # Attention mit Residual
        x = x + self.dropout(self.attention(self.norm1(x)))

        # MLP mit Residual
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x


class SelfAttention(nn.Module):
    """Self-Attention mit GQA-Unterstützung."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        d_model = config.d_model
        num_heads = config.num_heads
        kv_heads = config.kv_heads if config.attention_type == "gqa" else num_heads

        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.kv_heads = kv_heads

        # Q, K, V Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim)

        # Output Projection
        self.out_proj = nn.Linear(d_model, d_model)

        # RoPE (optional)
        self.use_rope = config.use_rope
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim)

        # Scale
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass."""
        batch_size, seq_len, _ = x.shape

        # Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.kv_heads, self.head_dim)

        # Transpose für Attention
        q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE (optional)
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        # Attention
        if self.num_heads == self.kv_heads:
            # Standard Multi-Head Attention
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            # GQA: Repeat KV-Heads
            q_per_kv = self.num_heads // self.kv_heads
            q = q.reshape(batch_size, self.kv_heads, q_per_kv, seq_len, self.head_dim)
            k = k.unsqueeze(2)
            v = v.unsqueeze(2)

            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
            attn = attn.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # Output
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(attn)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # RoPE-Frequenzen
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE zu x (batch, heads, seq, dim)."""
        seq_len = x.shape[2]
        dim = x.shape[3]

        # Positionen
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)

        # RoPE anwenden - cos/sin haben shape (seq_len, dim/2)
        cos = freqs.cos()
        sin = freqs.sin()

        # Rotate x
        x_rot = self._rotate(x, cos, sin)

        return x_rot

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate x mit RoPE (Rotary Position Embedding).
        
        Correct implementation: rotates pairs of dimensions.
        For x = [x0, x1, x2, x3, ...], rotates as:
        [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        """
        # Split into pairs of dimensions (even and odd indices)
        # x shape: (..., dim) -> x1: (..., dim/2), x2: (..., dim/2)
        x1 = x[..., 0::2]  # Even indices: x0, x2, x4, ...
        x2 = x[..., 1::2]  # Odd indices: x1, x3, x5, ...
        
        # Rotate: [-x2, x1] for the paired dimensions
        # x1' = x1 * cos - x2 * sin
        # x2' = x1 * sin + x2 * cos
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        
        # Interleave back: [x1', x2', x1', x2', ...]
        # Stack and reshape to interleave
        rotated = torch.stack([rot_x1, rot_x2], dim=-1)
        return rotated.flatten(-2)


class StandardMLP(nn.Module):
    """Standard MLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model = config.d_model
        d_ff = d_model * config.mlp_ratio

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        # Aktivierung
        self.activation = self._get_activation(config.activation)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "leaky_relu" or activation == "leaky_relu_squared":
            return nn.LeakyReLU(0.01)
        elif activation == "star_relu":
            return StarReLU(beta=0.5)
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "silu":
            return nn.SiLU()
        else:
            return nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class GatedMLP(nn.Module):
    """Gated MLP (SwiGLU)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model = config.d_model
        d_ff = d_model * config.mlp_ratio

        # Gate-Varianten
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass."""
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class StarReLU(nn.Module):
    """Star-ReLU Aktivierung."""

    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass: (1-beta) * ReLU(x) + beta * ReLU(x)^2."""
        relu_x = F.relu(x)
        return (1 - self.beta) * relu_x + self.beta * relu_x.pow(2)


def create_model(config: ModelConfig | dict[str, Any]) -> AblationModel:
    """Erstelle Modell aus Konfiguration.

    Args:
        config: ModelConfig oder Dictionary

    Returns:
        AblationModel Instanz
    """
    if isinstance(config, dict):
        config = ModelConfig.from_dict(config)

    return AblationModel(config)
