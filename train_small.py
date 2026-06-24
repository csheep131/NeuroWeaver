from __future__ import annotations
import copy
import glob
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
try:
import zstandard
_COMPRESSOR = "zstd"
except ImportError:
_COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Flash Attention 3 with SDPA fallback for non-H100 GPUs
try:
from flash_attn_interface import flash_attn_func as flash_attn_3_func
_USE_FA3 = True
except ImportError:
_USE_FA3 = False

class Hyperparameters:
data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
train_files = os.path.join(data_path, "fineweb_train_*.bin")
val_files = os.path.join(data_path, "fineweb_val_*.bin")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
seed = int(os.environ.get("SEED", 1337))
val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
iterations = int(os.environ.get("ITERATIONS", 15000))
warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
warmup_steps = int(os.environ.get("WARMUP_STEPS", 0)) # Auf 0 gesetzt für Eval-Only
train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 131072))
train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 1024))
max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
num_layers = int(os.environ.get("NUM_LAYERS", 11))
num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
model_dim = int(os.environ.get("MODEL_DIM", 512))
num_heads = int(os.environ.get("NUM_HEADS", 8))
mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
embed_lr = float(os.environ.get("EMBED_LR", 0.6))
head_lr = float(os.environ.get("HEAD_LR", 0.008))
tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
beta1 = float(os.environ.get("BETA1", 0.9))
beta2 = float(os.environ.get("BETA2", 0.95))
adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
eval_stride = int(os.environ.get("EVAL_STRIDE", 32))
mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 2))
mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.3))
muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
swa_every = int(os.environ.get("SWA_EVERY", 50))
lawa_enabled = bool(int(os.environ.get("LAWA_ENABLED", "0")))
lawa_k = int(os.environ.get("LAWA_K", 10))
lawa_freq = int(os.environ.get("LAWA_FREQ", 100))
muon_wd = float(os.environ.get("MUON_WD", 0.04))
adam_wd = float(os.environ.get("ADAM_WD", 0.04))
qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
rope_dims = int(os.environ.get("ROPE_DIMS", 16))
ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
ve_dim = int(os.environ.get("VE_DIM", 128))
ve_layers = os.environ.get("VE_LAYERS", "9,10")
gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))
ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
ttt_lr = float(os.environ.get("TTT_LR", 0.002))
ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))

# --- Optimizer & Logic Utilities ---

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
a, b, c = (3.4445, -4.7750, 2.0315)
was_2d = G.ndim == 2
if was_2d: G = G.unsqueeze(0)
X = G.bfloat16()
transposed = X.size(-2) > X.size(-1)
if transposed: X = X.mT
X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
for _ in range(steps):
A = X @ X.mT
B = b * A + c * (A @ A)
X = a * X + B @ X
if transposed: X = X.mT
if was_2d: X = X.squeeze(0)
return X

class Muon(torch.optim.Optimizer):
def __init__(self, params, lr: float, momentum: float, backend_steps: int,
nesterov: bool = True, weight_decay: float = 0.0):
super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
nesterov=nesterov, weight_decay=weight_decay))
self._built = False

def _build(self):
self._distributed = dist.is_available() and dist.is_initialized()
self._world_size = dist.get_world_size() if self._distributed else 1
ws = self._world_size
self._bank_meta = []
for group in self.param_groups:
for p in group["params"]:
B = p.shape[0]
padded_B = ((B + ws - 1) // ws) * ws
shard_B = padded_B // ws
dev = p.device
self._bank_meta.append({
'p': p, 'B': B,
'padded_grad': torch.zeros(padded_B, *p.shape[1:], device=dev, dtype=torch.bfloat16),
'shard': torch.zeros(shard_B, *p.shape[1:], device=dev, dtype=torch.bfloat16),
'shard_mom': torch.zeros(shard_B, *p.shape[1:], device=dev, dtype=torch.bfloat16),
'full_update': torch.zeros(padded_B, *p.shape[1:], device=dev, dtype=torch.bfloat16),
'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
})
self._bank_meta.sort(key=lambda m: -m['p'].numel())
self._built = True

def launch_reduce_scatters(self):
if not self._built: self._build()
if not self._distributed: return
self._rs_futures = []
for m in self._bank_meta:
p = m['p']
if p.grad is None:
self._rs_futures.append(None)
continue
pg = m['padded_grad']
pg[:m['B']].copy_(p.grad.bfloat16())
if pg.shape[0] > m['B']: pg[m['B']:].zero_()
self._rs_futures.append(dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True))

@torch.no_grad()
def step(self):
if not self._built: self._build()
for group in self.param_groups:
lr, momentum, steps, nesterov = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
wd = group.get("weight_decay", 0.0)
sharded = self._distributed and hasattr(self, '_rs_futures')
prev_ag, prev_m = None, None
for i, m in enumerate(self._bank_meta):
p = m['p']
if p.grad is None: continue
if prev_ag:
prev_ag.wait()
if wd > 0.0: prev_m['p'].data.mul_(1.0 - lr * wd)
prev_m['p'].add_(prev_m['full_update'][:prev_m['B']].to(dtype=prev_m['p'].dtype), alpha=-lr * prev_m['scale'])
if sharded and self._rs_futures[i]:
self._rs_futures[i].wait()
g, buf = m['shard'], m['shard_mom']
else:
g = p.grad.bfloat16()
if "mom" not in self.state[p]: self.state[p]["mom"] = torch.zeros_like(g)
buf = self.state[p]["mom"]
buf.mul_(momentum).add_(g)
update = zeropower_via_newtonschulz5(g.add(buf, alpha=momentum) if nesterov else buf, steps=steps)
if sharded:
prev_ag, prev_m = dist.all_gather_into_tensor(m['full_update'], update, async_op=True), m
else:
if wd > 0.0: p.data.mul_(1.0 - lr * wd)
p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])
if prev_ag:
prev_ag.wait()
if wd > 0.0: prev_m['p'].data.mul_(1.0 - lr * wd)
prev_m['p'].add_(prev_m['full_update'][:prev_m['B']].to(dtype=prev_m['p'].dtype), alpha=-lr * prev_m['scale'])
if hasattr(self, '_rs_futures'): del self._rs_futures

# --- Tokenizer & Data ---

def build_sentencepiece_luts(sp, vocab_size, device):
size = max(int(sp.vocab_size()), vocab_size)
b_np, s_np, bound_np = np.zeros(size, dtype=np.int16), np.zeros(size, dtype=np.bool_), np.ones(size, dtype=np.bool_)
for i in range(int(sp.vocab_size())):
if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i): continue
bound_np[i] = False
if sp.is_byte(i): b_np[i] = 1; continue
p = sp.id_to_piece(i)
if p.startswith("\u2581"): s_np[i] = True; p = p[1:]
b_np[i] = len(p.encode("utf-8"))
return torch.tensor(b_np, device=device), torch.tensor(s_np, device=device), torch.tensor(bound_np, device=device)

def load_data_shard(file):
header = np.fromfile(file, dtype="<i4", count=256)
return torch.from_numpy(np.fromfile(file, dtype="<u2", count=int(header[2]), offset=1024).astype(np.uint16))

def load_validation_tokens(pattern, seq_len):
files = sorted(glob.glob(pattern))
tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
return tokens[:((tokens.numel()-1)//seq_len)*seq_len + 1]

class DistributedTokenLoader:
def __init__(self, pattern, rank, world_size, device):
self.rank, self.world_size, self.device = rank, world_size, device
self.files = sorted(glob.glob(pattern))
self.f_idx, self.pos = 0, 0
self.tokens = load_data_shard(self.files[0])
def next_batch(self, global_tokens, seq_len, grad_accum):
local = global_tokens // (self.world_size * grad_accum) + 1
chunk_size = local * self.world_size
if self.pos + chunk_size >= self.tokens.numel():
self.f_idx = (self.f_idx + 1) % len(self.files)
self.tokens, self.pos = load_data_shard(self.files[self.f_idx]), 0
chunk = self.tokens[self.pos : self.pos + chunk_size]
self.pos += chunk_size
t = chunk[self.rank * local : (self.rank+1) * local].to(dtype=torch.int64)
return t[:-1].reshape(-1, seq_len).to(self.device), t[1:].reshape(-1, seq_len).to(self.device)
def take(self, n):
# Einfaches Vorspulen für Resume
self.pos += n
while self.pos >= self.tokens.numel():
self.pos -= self.tokens.numel()
self.f_idx = (self.f_idx + 1) % len(self.files)
self.tokens = load_data_shard(self.files[self.f_idx])

# --- Model Modules ---

class CastedLinear(nn.Linear):
_qat_enabled = False
def forward(self, x):
w = self.weight.to(x.dtype)
if CastedLinear._qat_enabled and self.training:
with torch.no_grad():
s = (self.weight.abs().amax(dim=1) / 31.0).clamp_min(1e-4)
w_q = (torch.clamp(torch.round(self.weight / s[:, None]), -32, 31) * s[:, None]).to(x.dtype)
w = w + (w_q - w).detach()
return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

def restore_low_dim_params_to_fp32(module):
for name, p in module.named_parameters():
if (p.ndim < 2 or "scale" in name or "bias" in name) and p.dtype != torch.float32:
p.data = p.data.float()

class Rotary(nn.Module):
def __init__(self, dim, base=10000.0, rope_dims=0):
super().__init__()
self.inv_freq = 1.0 / (base ** (torch.arange(0, rope_dims or dim, 2).float() / (rope_dims or dim)))
self._cos, self._sin, self._seq = None, None, 0
def forward(self, n, dev, dtype):
if self._seq != n or self._cos.device != dev:
t = torch.arange(n, device=dev, dtype=torch.float32)
freqs = torch.outer(t, self.inv_freq.to(dev))
self._cos, self._sin, self._seq = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :], n
# FIX: .clone() verhindert Inference Mode Fehler bei TTT
return self._cos.to(dtype).clone(), self._sin.to(dtype).clone()

def apply_rotary(x, cos, sin, rope_dims=0):
r_dim = rope_dims or x.size(-1)
x_r, x_p = x[..., :r_dim], x[..., r_dim:]
half = r_dim // 2
x1, x2 = x_r[..., :half], x_r[..., half:]
out_r = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
return torch.cat((out_r, x_p), dim=-1)

class Attention(nn.Module):
def __init__(self, dim, h, h_kv, base, gain):
super().__init__()
self.h, self.h_kv, self.d = h, h_kv, dim // h
self.q_gain = nn.Parameter(torch.full((h,), gain))
self.rope = Rotary(self.d, base=base, rope_dims=16)
self.use_xsa = False
def forward(self, x, q_w, k_w, v_w, out_w, v_embed=None):
B, T, _ = x.shape
q = F.linear(x, q_w.to(x.dtype)).view(B, T, self.h, self.d)
k = F.linear(x, k_w.to(x.dtype)).view(B, T, self.h_kv, self.d)
v = (F.linear(x, v_w.to(x.dtype)) + (v_embed if v_embed is not None else 0)).view(B, T, self.h_kv, self.d)
q, k = F.rms_norm(q, (self.d,)), F.rms_norm(k, (self.d,))
cos, sin = self.rope(T, x.device, q.dtype)
q, k = apply_rotary(q, cos, sin, 16), apply_rotary(k, cos, sin, 16)
q = q * self.q_gain.to(q.dtype)[None, None, :, None]
if _USE_FA3:
y = flash_attn_3_func(q, k, v, causal=True)
else:
if self.h != self.h_kv:
k = k.repeat_interleave(self.h // self.h_kv, dim=2)
v = v.repeat_interleave(self.h // self.h_kv, dim=2)
y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
if self.use_xsa:
y_g = y.view(B, T, self.h_kv, self.h // self.h_kv, self.d)
vn = F.normalize(v, dim=-1).unsqueeze(-2)
y = (y_g - (y_g * vn).sum(-1, keepdim=True) * vn).view(B, T, -1)
return F.linear(y.reshape(B, T, -1), out_w.to(x.dtype))

class Block(nn.Module):
def __init__(self, dim, h, h_kv, mult, base, gain, idx):
super().__init__()
self.attn = Attention(dim, h, h_kv, base, gain)
self.attn_norm, self.mlp_norm = nn.LayerNorm(dim, elementwise_affine=False), nn.LayerNorm(dim, elementwise_affine=False)
self.attn_scale, self.mlp_scale = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.ones(dim))
self.mix = nn.Parameter(torch.tensor([1.0, 0.0]))
self.scale = 1.0 / math.sqrt(idx + 1)
def forward(self, x, x0, q, k, v, o, up, dw, ve=None):
m = self.mix.to(x.dtype)
xi = m[0] * x + m[1] * x0
x = xi + self.attn_scale.to(x.dtype) * self.attn(F.rms_norm(xi, (xi.size(-1),)) * self.scale, q, k, v, o, ve)
return x + self.mlp_scale.to(x.dtype) * F.linear(torch.square(F.leaky_relu(F.linear(F.rms_norm(x, (x.size(-1),)) * self.scale, up.to(x.dtype)), 0.5)), dw.to(x.dtype))

class GPT(nn.Module):
def __init__(self, args):
super().__init__()
self.args = args
self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
self.num_layers = args.num_layers
kv_dim = args.num_kv_heads * (args.model_dim // args.num_heads)
self.qo_bank = nn.Parameter(torch.empty(2 * args.num_layers, args.model_dim, args.model_dim))
self.kv_bank = nn.Parameter(torch.empty(2 * args.num_layers, kv_dim, args.model_dim))
self.mlp_up_bank = nn.Parameter(torch.empty(args.num_layers, int(args.mlp_mult * args.model_dim), args.model_dim))
self.mlp_down_bank = nn.Parameter(torch.empty(args.num_layers, args.model_dim, int(args.mlp_mult * args.model_dim)))
self.blocks = nn.ModuleList([Block(args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.rope_base, args.qk_gain_init, i) for i in range(args.num_layers)])
self.mtp_heads = nn.ModuleList([CastedLinear(args.model_dim, args.vocab_size, bias=False) for _ in range(args.mtp_num_heads)])
for h in self.mtp_heads: nn.init.zeros_(h.weight); h._zero_init = True
self.final_norm = nn.LayerNorm(args.model_dim, elementwise_affine=False)
self._init_weights()
def _init_weights(self):
nn.init.normal_(self.tok_emb.weight, std=self.args.tied_embed_init_std)
s = 1.0 / math.sqrt(2 * self.num_layers)
for i in range(self.num_layers):
for b in [self.qo_bank, self.kv_bank, self.mlp_up_bank]: nn.init.orthogonal_(b.data[i])
nn.init.zeros_(self.qo_bank.data[self.num_layers+i]); nn.init.orthogonal_(self.kv_bank.data[self.num_layers+i])
nn.init.zeros_(self.mlp_down_bank.data[i])
self.qo_bank.data[self.num_layers+i].mul_(s); self.mlp_down_bank.data[i].mul_(s)
def forward(self, input_ids, target_ids, current_softcap=None):
cap = torch.as_tensor(current_softcap or self.args.logit_softcap, dtype=torch.float32, device=input_ids.device)
x = F.rms_norm(self.tok_emb(input_ids), (self.args.model_dim,))
x0, n = x, self.num_layers
for i, b in enumerate(self.blocks):
x = b(x, x0, self.qo_bank[i], self.kv_bank[i], self.kv_bank[n+i], self.qo_bank[n+i], self.mlp_up_bank[i], self.mlp_down_bank[i])
logits_proj = F.linear(F.rms_norm(x, (x.size(-1),)), self.tok_emb.weight)
logits = cap * torch.tanh(logits_proj / cap)
loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")
if self.training and self.args.mtp_num_heads > 0:
for k, head in enumerate(self.mtp_heads):
off = k + 1
if off >= x.size(1): continue
m_logits = cap * torch.tanh(head(x[:, :-off, :].reshape(-1, x.size(-1))) / cap)
loss += self.args.mtp_loss_weight * F.cross_entropy(m_logits.float(), target_ids[:, off:].reshape(-1))
return loss
def forward_logits(self, input_ids):
x = F.rms_norm(self.tok_emb(input_ids), (self.args.model_dim,))
x0, n = x, self.num_layers
for i, b in enumerate(self.blocks):
x = b(x, x0, self.qo_bank[i], self.kv_bank[i], self.kv_bank[n+i], self.qo_bank[n+i], self.mlp_up_bank[i], self.mlp_down_bank[i])
return self.args.logit_softcap * torch.tanh(F.linear(F.rms_norm(x, (x.size(-1),)), self.tok_emb.weight) / self.args.logit_softcap)

# --- Eval & TTT Functions ---

def eval_val(args, model, rank, world_size, device, val_tokens, lut_b, lut_s, lut_bound):
model.eval()
local_n = (val_tokens.numel() - 1) // world_size
local_tokens = val_tokens[rank * local_n : (rank + 1) * local_n + 1].to(device)
x, y = local_tokens[:-1].reshape(-1, args.eval_seq_len), local_tokens[1:].reshape(-1, args.eval_seq_len)
l_sum, t_cnt, b_cnt = 0, 0, 0
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
for i in range(x.size(0)):
loss = model(x[i:i+1], y[i:i+1]).detach()
l_sum += loss.item() * y[i:i+1].numel()
t_cnt += y[i:i+1].numel()
tb = lut_b[y[i:i+1]] + (lut_s[y[i:i+1]] & ~lut_bound[x[i:i+1]]).int()
b_cnt += tb.sum().item()
return l_sum / t_cnt, (l_sum / t_cnt / math.log(2)) * (t_cnt / b_cnt)

def eval_val_sliding_ttt(args, model, rank, world_size, device, val_tokens, lut_b, lut_s, lut_bound):
# Optimierte TTT-Logik ohne Inference-Mode Fehler
stride, chunk_size = args.eval_stride, args.ttt_chunk_tokens
total = val_tokens.numel() - 1
loss_sum, t_cnt, b_cnt = 0, 0, 0

# Unfreeze only late blocks for TTT speed
ttt_params = []
for n, p in model.named_parameters():
if any(f"blocks.{i}." in n for i in range(args.num_layers - args.ttt_freeze_blocks, args.num_layers)):
p.requires_grad_(True); ttt_params.append(p)
else: p.requires_grad_(False)

opt = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

for ci in range(0, total, chunk_size):
ce = min(ci + chunk_size, total)
# 1. Score (Inference)
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
# Inkrementelles Scoring...
tokens = val_tokens[ci : ce + 1].to(device)
logits = model.forward_logits(tokens[:-1].unsqueeze(0))
nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[1:], reduction="none")
loss_sum += nll.sum().item()
t_cnt += nll.numel()
tb = lut_b[tokens[1:]] + (lut_s[tokens[1:]] & ~lut_bound[tokens[:-1]]).int()
b_cnt += tb.sum().item()

# 2. Train (Legal TTT)
model.train()
for _ in range(args.ttt_epochs):
opt.zero_grad()
with torch.autocast("cuda", dtype=torch.bfloat16):
# FIX: .clone() zur Sicherheit gegen Graph-Leaks
loss = model(tokens[:-1].unsqueeze(0).clone(), tokens[1:].unsqueeze(0).clone())
loss.backward()
torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
opt.step()

return loss_sum / t_cnt, (loss_sum / t_cnt / math.log(2)) * (t_cnt / b_cnt)

# --- Main ---

def main():
args = Hyperparameters()
rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
if dist.is_available() and world_size > 1: dist.init_process_group("nccl")

# Fix für Dynamo / Compiler
import torch._dynamo
torch._dynamo.config.recompile_limit = 1024

sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
lut_b, lut_s, lut_bound = build_sentencepiece_luts(sp, args.vocab_size, device)
val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)

base_model = GPT(args).to(device).bfloat16()
restore_low_dim_params_to_fp32(base_model)

# Resume / Load
if os.path.exists("final_model.pt"):
print(f" Loading weights from final_model.pt (Rank {rank})")
base_model.load_state_dict(torch.load("final_model.pt", map_location=device), strict=False)

# Compiler (OHNE mark_dynamic um den AssertionError zu umgehen)
model = torch.compile(base_model, dynamic=False, fullgraph=True)

# --- EVAL ONLY MODE ---
print(f" Starting Final Evaluation (Rank {rank})")
l, bpb = eval_val(args, model, rank, world_size, device, val_tokens, lut_b, lut_s, lut_bound)
print(f"DIAGNOSTIC: Loss={l:.4f}, BPB={bpb:.4f}")

if args.ttt_enabled:
print(f" Starting Legal TTT...")
l_ttt, bpb_ttt = eval_val_sliding_ttt(args, base_model, rank, world_size, device, val_tokens, lut_b, lut_s, lut_bound)
print(f"FINAL TTT BPB: {bpb_ttt:.6f}")

# --- Quantisierung & Save ---
if rank == 0:
# Hier deine mixed_quantize_int6 Logik ausführen und speichern
print(" Quantizing and saving final_model.int6.ptz...")
# (Da du den Quantisierungs-Code bereits hast, hier einfügen oder Datei-Sicherung nutzen)

if dist.is_available() and world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
main()