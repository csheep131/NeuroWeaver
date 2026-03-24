# Trainingsplan — Ablation Machine

Dieses Dokument beschreibt die notwendigen Schritte für echte Trainingsläufe in allen 3 Phasen.

**Erstellt:** 2026-03-24  
**Status:** Ready für Training

---

## Voraussetzungen ✅

### Abgeschlossen (2026-03-24)

- [x] **Virtuelles Environment** (`.venv/`)
  - Python 3.11+
  - pip 26.0.1+

- [x] **PyTorch Installation**
  - Version: 2.11.0+cu130
  - CUDA Support: ✅ Verfügbar
  - CPU-Fallback: ✅ Verfügbar

- [x] **Rust-Core kompiliert**
  - maturin: 1.12.6
  - PyO3: 0.20 (kompatibel)
  - Build-Status: ✅ SUCCESS

- [x] **Alle Run-Konfigurationen**
  - Phase 1: 5 Runs ✅
  - Phase 2: 10 Runs ✅
  - Phase 3: 2 Combos ✅

---

## Phase 1: Baseline Training

### Ziel
Echte Trainingsläufe für alle 5 Phase-1-Runs mit tatsächlichen BPB-Metriken.

### Notwendige Komponenten

#### 1.1 Daten-Loader

```python
# TODO: Implementierung erforderlich
# Datei: data/text_loader.py

class TextDataLoader:
    """Lädt und tokenisiert Textdaten für Training."""
    
    def __init__(self, data_path: str, tokenizer, seq_len: int):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    
    def __iter__(self):
        # Yield batches of tokens
        pass
```

**Erforderliche Features:**
- [ ] Textdateien einlesen (UTF-8)
- [ ] Tokenisierung (Byte/Bigram/Trigram)
- [ ] Batching mit seq_len
- [ ] Shuffle für Training

#### 1.2 PyTorch Modell-Integration

```python
# TODO: Implementierung erforderlich
# Datei: train/pytorch_model.py

import torch
import torch.nn as nn
from rust_core import Backbone  # Rust-Backend

class AblationModel(nn.Module):
    """PyTorch-Modell mit Rust-Backend."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Rust-Backend initialisieren
        self.backbone = Backbone(config)
    
    def forward(self, x):
        return self.backbone.forward(x)
```

**Erforderliche Features:**
- [ ] Rust-Backend in PyTorch integrieren
- [ ] Forward-Pass implementieren
- [ ] Loss-Berechnung (Cross-Entropy)
- [ ] Gradient-Checkpointing (optional für VRAM)

#### 1.3 Training Loop

```python
# TODO: Erweitern in train/trainer.py

def train_step(self, batch) -> float:
    """Echter Training-Schritt."""
    # 1. Forward pass
    logits = self.model(batch)
    
    # 2. Loss berechnen
    loss = self.criterion(logits, targets)
    
    # 3. Backward pass
    loss.backward()
    
    # 4. Gradient clipping
    if self.config.grad_clip:
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.grad_clip
        )
    
    # 5. Optimizer step
    self.optimizer.step()
    self.optimizer.zero_grad()
    
    return loss.item()
```

**Erforderliche Features:**
- [ ] Forward/Backward-Pass
- [ ] Gradient Clipping
- [ ] Optimizer (AdamW)
- [ ] Learning Rate Scheduler

#### 1.4 Evaluation

```python
# TODO: Erweitern in eval/bpb_eval.py

def compute_bpb(self, model, data_loader) -> float:
    """Echte BPB-Berechnung."""
    total_bits = 0
    total_bytes = 0
    
    for batch in data_loader:
        with torch.no_grad():
            logits = model(batch)
            # Cross-Entropy → Bits
            bits = self._compute_bits(logits, batch)
            total_bits += bits
            total_bytes += batch.num_bytes
    
    return total_bits / total_bytes
```

### Run-Plan Phase 1

| Run | Proxy-Steps | Echte Steps | Seq-Len | VRAM (geschätzt) | Zeit (geschätzt) |
|-----|-------------|-------------|---------|------------------|------------------|
| run001_control | 50 | 10.000 | 256 | ~4 GB | ~30 Min |
| run001b_frontierish | 50 | 10.000 | 256 | ~4 GB | ~30 Min |
| run002a_bigram_4k | 50 | 10.000 | 256 | ~4 GB | ~30 Min |
| run002b_bigram_8k | 50 | 10.000 | 256 | ~4 GB | ~30 Min |
| run002c_trigram | 50 | 10.000 | 256 | ~4 GB | ~30 Min |

### Erfolgskriterien Phase 1

- [ ] Alle 5 Runs starten ohne Fehler
- [ ] BPB-Werte werden konsistent geschrieben
- [ ] `val_bpb < 1.60` für alle Runs
- [ ] Kein OOM bei lokaler 8GB GPU
- [ ] Training konvergiert (BPB sinkt über Steps)

---

## Phase 2: Feature Training

### Ziel
Echte Trainingsläufe für alle 10 Phase-2-Runs mit Feature-spezifischen Metriken.

### Zusätzliche Komponenten

#### 2.1 Feature-Implementierungen

**XSA (Cross-Sequence Attention):**
```python
# TODO: rust-core/src/models.rs erweitern
# oder Python-Implementierung in train/xsa.py

class CrossSequenceAttention(nn.Module):
    """XSA für lange Abhängigkeiten."""
    
    def __init__(self, d_model, window_size=2048):
        super().__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)
    
    def forward(self, x, memory=None):
        # Cross-Sequence Attention Logik
        pass
```

**FiLM (Feature-wise Linear Modulation):**
```python
# TODO: train/film.py

class FiLMLayer(nn.Module):
    """FiLM für konditionale Skalierung."""
    
    def __init__(self, d_model, cond_dim=64):
        super().__init__()
        self.scale = nn.Linear(cond_dim, d_model)
        self.shift = nn.Linear(cond_dim, d_model)
    
    def forward(self, x, condition):
        gamma = self.scale(condition)
        beta = self.shift(condition)
        return gamma * x + beta
```

**TTT (Test-Time Training):**
```python
# TODO: train/ttt.py

class TestTimeTraining(nn.Module):
    """TTT für adaptive Inferenz."""
    
    def __init__(self, base_model, lr=1e-4):
        super().__init__()
        self.model = base_model
        self.lr = lr
    
    def adapt(self, x, steps=1):
        # Mini-Updates während Inferenz
        pass
```

#### 2.2 Quantisierung

```python
# TODO: quant/fake_quant.py

class FakeQuantizer:
    """Fake-Quantisierung für Training."""
    
    def __init__(self, bits=6):
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
    
    def quantize(self, x):
        # Fake-Quantisierung (straight-through estimator)
        scale = (x.max() - x.min()) / (self.qmax - self.qmin)
        x_int = torch.round(x / scale).clamp(self.qmin, self.qmax)
        return x_int * scale  # Straight-through
```

### Run-Plan Phase 2

| Run | Feature | Steps | VRAM | Zeit | Priorität |
|-----|---------|-------|------|------|-----------|
| run003_xsa | XSA | 10.000 | ~6 GB | ~45 Min | P2 |
| run004_leakyrelu | LeakyReLU² | 10.000 | ~4 GB | ~30 Min | P1 |
| run005a_quant | INT5/6 Mixed | 10.000 | ~4 GB | ~40 Min | P2 |
| run005b_quant | INT6/5 Mixed | 10.000 | ~4 GB | ~40 Min | P2 |
| run006_film | FiLM | 10.000 | ~5 GB | ~35 Min | P3 |
| run007_ttt | TTT | 10.000 | ~6 GB | ~50 Min | P4 |
| run008a_star_relu | Star-ReLU | 10.000 | ~4 GB | ~30 Min | P3 |
| run008b_gated_mlp | SwiGLU | 10.000 | ~5 GB | ~35 Min | P3 |
| run009_gqa | GQA | 10.000 | ~4 GB | ~30 Min | P1 |
| run010_recurrence | Recurrent | 10.000 | ~5 GB | ~45 Min | P1 |

### Erfolgskriterien Phase 2

- [ ] Alle 10 Runs starten ohne Fehler
- [ ] Feature-spezifische Metriken werden geschrieben
- [ ] Gate-Status kann bestimmt werden (PASS/WATCH/FAIL)
- [ ] Δ BPB vs. Parent wird berechnet
- [ ] Quant-Gap wird für Quant-Runs berechnet

---

## Phase 3: Combo Training

### Ziel
Echte Trainingsläufe für dynamische Feature-Kombinationen.

### Automatische Combo-Erstellung

```bash
# Combo Builder ausführen
source .venv/bin/activate
python3 -c "
from orchestrator import generate_phase3_combos
best_combo, quant_combo = generate_phase3_combos()
print(f'Best Combo: {best_combo.combo_id}')
print(f'Quant Combo: {quant_combo.combo_id}')
"
```

### Combo-Konfiguration (dynamisch)

Der Combo Builder wählt automatisch:
- **Besten Tokenizer** aus Phase 1 (niedrigste BPB)
- **Beste Aktivierung** aus Phase 2 (beste BPB/ms)
- **Beste Attention** aus Phase 2 (beste Effizienz)
- **Beste Quant-Strategie** aus run005a/b (kleinstes Quant-Gap)

### Run-Plan Phase 3

| Run | Typ | Steps | VRAM | Zeit | Submission |
|-----|-----|-------|------|------|------------|
| run016_best_combo_a | Nicht-quantisiert | 10.000 | ~6 GB | ~45 Min | Vorbereitung |
| run017_best_combo_quantized | Quantisiert | 10.000 | ~5 GB | ~45 Min | ✅ Bereit |

### Erfolgskriterien Phase 3

- [ ] Combo ist besser als beste Einzel-Features (Synergie)
- [ ] `artifact_bytes < 16 MB` (Challenge-Limit)
- [ ] `val_bpb < 1.50` (Challenge-Ziel)
- [ ] Quant-Gap < 0.05 (für quantisierte Combos)
- [ ] Submission Bundle kann erstellt werden

---

## Multi-Seed Validierung (H100)

### Nur für finale Submission

**Auf H100 Hardware:**
```bash
# Multi-Seed für Top-Kandidaten
python3 -m runs.run --config configs/runs/run017_best_combo_quantized.yaml --seeds 42,43,44
```

### Erfolgskriterien

- [ ] 3 Seeds mit `val_bpb < 1.50`
- [ ] `σ < 0.03 BPB` über 3 Seeds
- [ ] Alle Seeds konvergieren stabil
- [ ] Submission Bundle mit allen 3 Seeds

---

## Checkliste vor Trainingsstart

### System-Voraussetzungen

- [ ] `.venv/` aktiviert: `source .venv/bin/activate`
- [ ] PyTorch installiert: `python3 -c "import torch; print(torch.__version__)"`
- [ ] Rust-Core kompiliert: `python3 -c "import rust_core; print('OK')"`
- [ ] CUDA verfügbar (optional): `python3 -c "import torch; print(torch.cuda.is_available())"`

### Daten-Voraussetzungen

- [ ] Trainingsdaten vorhanden (Pfad in Config)
- [ ] Eval-Daten vorhanden
- [ ] Tokenizer getestet

### Code-Voraussetzungen

- [ ] `train/trainer.py` mit echtem Forward/Backward-Pass
- [ ] `train/pytorch_model.py` mit Modell-Integration
- [ ] `data/text_loader.py` mit Daten-Loader
- [ ] `eval/bpb_eval.py` mit echter BPB-Berechnung

---

## Start-Kommandos

### Smoke-Test (bereits erfolgreich)

```bash
source .venv/bin/activate
python3 -m runs.run --config configs/runs/run001_control.yaml --smoke-test
```

### Proxy-Run (bereits erfolgreich)

```bash
source .venv/bin/activate
python3 -m runs.run --config configs/runs/run001_control.yaml --local-proxy
```

### Echtes Training (sobald implementiert)

```bash
source .venv/bin/activate

# Phase 1
python3 -m runs.run --config configs/runs/run001_control.yaml --mode train

# Phase 2
python3 -m runs.run --config configs/runs/run004_leakyrelu.yaml --mode train

# Phase 3
python3 -m runs.run --config configs/runs/run016_best_combo_a.yaml --mode train
```

---

## Troubleshooting

### OOM (Out of Memory)

```yaml
# In Config anpassen:
local_proxy:
  seq_len: 128  # Reduzieren (256 → 128)
  microbatch: 1  # Minimieren
  grad_accumulation: 16  # Erhöhen für effektive Batch-Größe
```

### Langsames Training

```yaml
# In Config anpassen:
training:
  batch_size: 64  # Erhöhen (wenn VRAM verfügbar)
  # Oder:
  # flash_attention: true  # Wenn GPU es unterstützt
```

### Rust-Core Fehler

```bash
# Neu kompilieren
cd rust-core
source ../.venv/bin/activate
maturin develop --release
```

---

## Nächste Schritte

1. **Daten-Loader implementieren** (data/text_loader.py)
2. **PyTorch-Modell integrieren** (train/pytorch_model.py)
3. **Training Loop erweitern** (train/trainer.py)
4. **BPB-Evaluation implementieren** (eval/bpb_eval.py)
5. **Ersten echten Training-Run starten**

---

*Dokument erstellt: 2026-03-24*  
*Letzte Aktualisierung: 2026-03-24*  
*Status: Ready für Implementierung*
