# Submission Guide — Parameter Golf Challenge

Dieser Guide beschreibt den kompletten Prozess für eine Submission zur Parameter Golf Challenge mit NeuroWeave.

## Challenge Übersicht

### Ziel
Trainiere das beste Sprachmodell, das in ein **16MB Artifact** passt und in **unter 10 Minuten auf 8xH100** trainiert.

### Bewertung
- **Metrik:** Bits Per Byte (BPB) auf FineWeb Validation Set
- **Tokenizer-agnostisch:** BPB wird auf Byte-Ebene berechnet
- **Artifact-Size:** Code + komprimierte Modell-Gewichte (dezimale 16MB = 16.000.000 Bytes)

### Constraints
| Constraint | Limit |
|------------|-------|
| **Artifact Size** | < 16.000.000 Bytes |
| **Training Time** | < 10 Minuten auf 8xH100 |
| **Evaluation Time** | < 10 Minuten auf 8xH100 |
| **External Compute** | Nicht erlaubt (kein Cheating) |

---

## Quick Start

### 1. Repository klonen

```bash
git clone https://github.com/neuro-weave/NeuroWeave.git
cd NeuroWeave
```

### 2. Dependencies installieren

```bash
# Core Dependencies
pip install torch numpy pyyaml

# Dataset & Tokenizer
pip install datasets tqdm sentencepiece

# Optional: MLX für Apple Silicon (lokales Testing)
pip install mlx
```

### 3. Dataset vorbereiten

```bash
# FineWeb mit 1024-token Vocabulary herunterladen
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

Dies erstellt:
- `./data/datasets/fineweb10B_sp1024/` — Tokenisierte Datensätze
- `./data/tokenizers/fineweb_1024_bpe.model` — SentencePiece Tokenizer

### 4. Smoke Test (lokal)

```bash
# Apple Silicon (MLX)
RUN_ID=smoke_test ITERATIONS=200 python train_gpt_mlx.py

# Oder mit PyTorch (CPU/GPU)
RUN_ID=smoke_test ITERATIONS=200 python train_gpt.py
```

### 5. Training auf H100

```bash
# Auf 8xH100 mit torchrun
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  --run_id baseline_v1
```

Oder mit Environment Variables:

```bash
RUN_ID=baseline_v1 \
ITERATIONS=2000 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=200 \
MAX_WALLCLOCK_SECONDS=600 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/train \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Submission Checklist

Eine vollständige Submission muss folgende Kriterien erfüllen:

### ✅ Pflicht-Dateien

```
records/<submission_name>/
├── README.md           # Beschreibung der Submission
├── submission.json     # Metadaten (Name, GitHub ID, val_bpb, etc.)
├── train_gpt.py        # Trainingsskript (muss lauffähig sein)
├── requirements.txt    # Dependencies (falls nicht Standard)
└── logs/
    ├── run1.log        # Trainings-Log (Seed 1)
    ├── run2.log        # Trainings-Log (Seed 2)
    └── run3.log        # Trainings-Log (Seed 3, empfohlen)
```

### ✅ Metriken

| Metrik | requirement |
|--------|-------------|
| **val_bpb** | Muss dokumentiert sein |
| **compressed_size_bytes** | < 16.000.000 |
| **training_time** | < 600 Sekunden |
| **statistical_significance** | 3 Runs für p < 0.01 |

### ✅ Reproduzierbarkeit

- [ ] Skript läuft ohne manuelle Intervention
- [ ] Alle Dependencies sind dokumentiert
- [ ] Dataset-Zugriff ist beschrieben
- [ ] Logs zeigen vollständigen Training-Verlauf

---

## Submission.json Template

```json
{
  "name": "Your Submission Name",
  "github_id": "your-github-username",
  "val_bpb": 1.234,
  "compressed_size_bytes": 15000000,
  "description": "Brief description of your approach",
  "architecture": {
    "num_layers": 11,
    "d_model": 512,
    "num_heads": 8,
    "kv_heads": 4,
    "mlp_ratio": 4,
    "vocab_size": 1024,
    "activation": "leaky_relu_squared",
    "attention_type": "gqa",
    "use_rope": true
  },
  "training": {
    "optimizer": "adamw",
    "learning_rate": 0.0003,
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "max_steps": 2000,
    "batch_tokens": 8192
  },
  "seeds": [42, 1, 2],
  "logs": ["logs/run1.log", "logs/run2.log", "logs/run3.log"],
  "created_at": "2026-03-25",
  "notes": "Any additional notes"
}
```

---

## Compression & Artifact Size

### Berechnung

Die Artifact-Size wird wie folgt berechnet:

```python
import zlib
import torch

# Modell speichern
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
raw_bytes = buffer.getvalue()

# Mit zlib komprimieren (level 9)
compressed = zlib.compress(raw_bytes, level=9)
artifact_size = len(compressed)
```

### Tipps für kleinere Artifacts

1. **Quantisierung:** INT8 statt FP32 (4x Reduktion)
2. **Weight Tying:** Embeddings teilen (Input = Output)
3. **Parameter Tying:** Layer-Gewichte wiederverwenden
4. **Low-Rank Training:** LoRA-ähnliche Techniken
5. **Pruning:** Unwichtige Gewichte entfernen

---

## Evaluation

### Bits Per Byte (BPB)

BPB wird auf Byte-Ebene berechnet:

```
BPB = Cross-Entropy-Loss (nats) / ln(2)
```

### Sliding Window Evaluation

Für bessere Genauigkeit kann Sliding Window verwendet werden:

```python
# In train_gpt.py: Sliding Window aktivieren
VAL_LOSS_EVERY=200 \
python train_gpt.py
```

### 3-Seed Validierung

Für statistische Signifikanz:

```bash
# Seed 1
RUN_ID=baseline_s1 SEED=42 python train_gpt.py

# Seed 2
RUN_ID=baseline_s2 SEED=1 python train_gpt.py

# Seed 3
RUN_ID=baseline_s3 SEED=2 python train_gpt.py

# Durchschnitt und Standardabweichung berechnen
```

---

## Challenge-spezifische Features

### Empfohlene Architektur-Entscheidungen

| Feature | Empfehlung | Begründung |
|---------|------------|------------|
| **Layers** | 10-12 | Mehr Layer = bessere Qualität |
| **d_model** | 512-768 | Balance zwischen Capacity und Size |
| **Activation** | LeakyReLU² | Besser als GELU in Challenge |
| **Attention** | GQA | Effizienter als MHA |
| **Positional** | RoPE | Bewährt in Challenge |
| **Quantisierung** | INT8 QAT | Erhält Qualität bei kleiner Size |

### Experimentelle Features

- **TTT (Test-Time Training):** Adaptive Inference
- **XSA (Cross-Sequence Attention):** Längerer Kontext
- **FiLM (Feature-wise Linear Modulation):** Conditional Computing
- **Recurrence:** Tiefe durch Wiederholung

---

## Häufige Fehler

### ❌ Artifact zu groß

**Problem:** Compressed Size > 16MB

**Lösungen:**
- INT8 Quantisierung verwenden
- Weniger Parameter (smaller d_model oder weniger Layer)
- Weight Tying aktivieren

### ❌ Training zu langsam

**Problem:** > 10 Minuten auf 8xH100

**Lösungen:**
- Batch-Größe erhöhen
- Gradient Accumulation reduzieren
- Mixed Precision (AMP) verwenden

### ❌ val_bpb zu hoch

**Problem:** BPB > 1.50 (schlechter als Baseline)

**Lösungen:**
- Mehr Training Steps
- Learning Rate anpassen
- Architektur verbessern (mehr Layer, bessere Activation)

---

## Einreichung

### Pull Request erstellen

1. **Fork** das NeuroWeave Repository
2. **Branch** erstellen: `submission/<your-name>-<date>`
3. **Submission** Ordner hinzufügen: `records/<submission_name>/`
4. **PR** erstellen mit Beschreibung

### PR-Beschreibung Template

```markdown
## Submission: <Name>

**val_bpb:** X.XXXX
**compressed_size:** XX MB
**training_time:** XX min on 8xH100

## Approach

Brief description of your approach and key innovations.

## Architecture Changes

- List key architectural changes

## Training Details

- Learning rate: X
- Batch size: X
- Steps: X

## Logs

Attached are logs from 3 independent runs.
```

---

## Support

- **Discord:** [OpenAI Parameter Golf Channels](https://discord.gg/openai)
- **Issues:** [GitHub Issues](https://github.com/neuro-weave/NeuroWeave/issues)
- **Compute Grants:** [OpenAI Compute Grant Form](https://forms.openai.com/compute-grant)

---

## Ressourcen

- [Challenge Regeln](../../regeln.md)
- [RunPod Setup](runpod_setup.md)
- [Architecture Guide](../guides/architecture.md)
- [Compression Guide](compression.md)
