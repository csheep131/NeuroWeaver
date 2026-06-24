# SOTA Reference — Pflichtlektüre für alle Agenten

**Stand: 2026-03-28** | **Aktueller SOTA: val_bpb = 1.1194** (PR #549, abaybektursun)

---

## WICHTIG: Nicht von Null anfangen!

Die aktuelle `train_gpt.py` in diesem Repo IST bereits der SOTA.
**Agenten sollen GEZIELTE VERBESSERUNGEN machen, nicht alles neu schreiben.**

Wenn du `train_gpt.py` komplett neu schreibst, landest du bei ~4.0 BPB statt ~1.12.
Die Differenz kommt von ~15 Features die ALLE zusammenwirken.

---

## Architektur des aktuellen SOTA (1920 Zeilen)

### Modell: 11L x 512d x 8H GPT mit U-Net Skips

```
Hyperparameter Wert Warum

num_layers 11 Mehr Kapazität, passt noch in 16MB mit INT6
model_dim 512 Sweet spot für 16MB
num_heads 8 GQA: 8 query heads, 4 KV heads
num_kv_heads 4 Spart Parameter für KV-Projektion
mlp_mult 3.0 3x MLP statt 4x → mehr Layer möglich
vocab_size 1024 SentencePiece BPE, 1024 tokens
train_seq_len 2048 Auf H100; 1024 auf RTX 3050
logit_softcap 30.0 Stabilisiert Training
rope_base 10000.0 Standard RoPE
rope_dims 16 Partial RoPE: nur 16 von 64 dims
xsa_last_n 4 Cross-Scale Attention auf letzten 4 Layern
tie_embeddings true Spart ~262K params
bigram_vocab_size 2048 BigramHash embedding
bigram_dim 128 Projiziert auf model_dim
ve_enabled true Value Embeddings auf Layer 9,10
ve_dim 128
ln_scale true Layer-wise LN scaling: 1/√(i+1)
```

### Optimizer: Parallel Muon + Adam

```
Muon (für 4 Parameter-Banks):
- Newton-Schulz Orthogonalisierung (5 Schritte)
- lr=0.025, momentum=0.99 (warmup von 0.92)
- weight_decay=0.04
- Parallel: reduce-scatter → NS5 → all-gather

Adam (für Embeddings, Scalars, Controls):
- embed_lr=0.035 (tied), scalar_lr=0.025
- beta1=0.9, beta2=0.95, eps=1e-8
- weight_decay=0.04

Warmdown: Letzte 3500 Steps, LR → 0
Late QAT: Aktiviert wenn LR-Scale < 0.15
```

### Parameter Banking (KRITISCH!)

Statt einzelne nn.Linear pro Layer werden 4 "Bank"-Tensoren verwendet:
```python
qo_bank: [2*num_layers, model_dim, model_dim] # Q und Out Projektionen
kv_bank: [2*num_layers, kv_dim, model_dim] # K und V Projektionen
mlp_up_bank: [num_layers, mlp_dim, model_dim] # MLP Up
mlp_down_bank: [num_layers, model_dim, mlp_dim] # MLP Down
```
→ Ermöglicht batched Newton-Schulz im Muon Optimizer
→ Ermöglicht Parallel Muon mit reduce-scatter/all-gather

### Schlüssel-Features (ALLE nötig, nicht optional!)

1. **RMSNorm** statt LayerNorm (schneller, weniger params)
2. **SmearGate**: Lernt Token-Smoothing zwischen benachbarten Positionen
3. **BigramHash**: Hash-basierte Bigram-Embeddings → zusätzlicher Kontext
4. **U-Net Skip Connections**: Encoder-Decoder mit Skip-Weights
5. **XSA (Cross-Scale Attention)**: Auf letzten 4 Layern, V-Projektion subtrahiert
6. **EMA Weight Averaging**: decay=0.997, besser als SWA
7. **Value Embeddings**: Auf Layer 9,10, injiziert Token-Identität in V
8. **Partial RoPE**: Nur 16 von 64 Head-Dims → Rest ist absolute Position
9. **Logit Softcapping**: tanh(logits/30)*30 → stabilisiert
10. **Warmup/Warmdown**: 20 warmup Steps, 3500 warmdown Steps
11. **Late QAT**: INT6 QAT erst in der Warmdown-Phase
12. **LeakyReLU²**: `leaky_relu(x, 0.5).square()` statt GELU
13. **CastedLinear**: Weights in FP32, Forward in BF16
14. **torch.compile**: fullgraph=True für Performance

### Komprimierung: INT6 + LZMA

```
1. Training in BF16/FP32
2. EMA Averaging nach Training
3. Unbank 3D Banks → individuelle 2D Tensoren
4. INT6 per-row Quantisierung (±31 clip) für Attention+MLP
5. INT8 per-row für Rest
6. FP16 für kleine Tensoren (<65536 params)
7. FP32 für Control-Tensoren (scales, gates, etc.)
8. LZMA preset=6 Komprimierung
→ Ergebnis: ~12.5 MB (unter 16MB Limit)
```

### Evaluation Pipeline

```
1. Standard Eval: Full validation set, batch-weise
2. Sliding Window Eval: stride=64, maximaler Kontext pro Token
3. BPB Berechnung: bits_per_token × tokens_per_byte (SentencePiece-aware)
4. Optional: Test-Time Training (TTT) mit Score-First Protokoll
```

---

## Wie man den SOTA VERBESSERT (nicht ersetzt!)

### DO: Gezielte Änderungen

```
Ein Feature hinzufügen/verbessern (z.B. bessere Quantisierung)
Hyperparameter tunen (z.B. LR, Warmdown)
Neuen Trick einbauen der mit dem Rest kompatibel ist
Komprimierung verbessern (z.B. INT5 für manche Tensoren)
Eval verbessern (z.B. besseres TTT-Protokoll)
Architektur-Tweak (z.B. mehr Layer wenn Platz durch bessere Quant)
```

### DON'T: Alles neu schreiben

```
train_gpt.py von Grund auf neu schreiben
Parameter Banking entfernen
Muon Optimizer durch AdamW ersetzen
Features weglassen "weil einfacher"
Standard-nn.Linear statt CastedLinear verwenden
EMA/SWA weglassen
Quantisierung weglassen oder vereinfachen
```

### Ideen für nächste Verbesserungen (von Leaderboard abgeleitet)

1. **Bessere Quantisierung**: INT5 für MLP, INT6 für Attention → mehr Platz für Layer
2. **Mehr Layer**: 12L statt 11L wenn Quant klein genug
3. **Besseres TTT**: Mehr Epochs, bessere LR-Schedule
4. **Depth Recurrence**: Shared weights über Layer-Gruppen
5. **Gated Attention**: Sigmoid-Gate pro Head
6. **Value Residual**: V0 + learnable mix
7. **Besserer Tokenizer**: Größeres Vocab wenn Platz
8. **Custom CUDA Kernels**: Fused attention/MLP

---

## Technische Details für Agenten

### Flash Attention 3 → SDPA Fallback

Der SOTA verwendet Flash Attention 3 (nur H100). Für andere GPUs (RTX 3050, etc.)
ist ein SDPA-Fallback eingebaut:

```python
try:
from flash_attn_interface import flash_attn_func as flash_attn_3_func
_USE_FA3 = True
except ImportError:
_USE_FA3 = False

# In CausalSelfAttention.forward():
if _USE_FA3:
y = flash_attn_3_func(q, k, v, causal=True) # (B,T,H,D)
else:
# SDPA: transpose to (B,H,T,D), expand GQA heads
y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
```

### Grad Accumulation

```
8 GPUs: grad_accum_steps = 1
4 GPUs: grad_accum_steps = 2
1 GPU: grad_accum_steps = 8
```

### Distributed Training

Parallel Muon handles bank grads via reduce-scatter/all-gather.
Non-bank grads are manually all-reduced before Adam steps.
**Kein DDP!** (DDP würde die Banks doppelt synchronisieren)

---

Letzte Aktualisierung: 2026-03-28
