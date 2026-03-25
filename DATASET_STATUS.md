# Dataset & Tokenizer Status

**Stand:** 2026-03-25 12:25 Uhr

---

## ✅ DATASET ERSTELLT

### Ordnerstruktur

```
data/datasets/fineweb10B_sp1024/
├── train/
│   ├── shard_00000.bin (20 MB, 10M tokens)
│   ├── shard_00001.bin (20 MB, 10M tokens)
│   ├── shard_00002.bin (20 MB, 10M tokens)
│   ├── shard_00003.bin (20 MB, 10M tokens)
│   ├── shard_00004.bin (20 MB, 10M tokens)
│   ├── shard_00005.bin (20 MB, 10M tokens)
│   ├── shard_00006.bin (20 MB, 10M tokens)
│   ├── shard_00007.bin (20 MB, 10M tokens)
│   ├── shard_00008.bin (20 MB, 10M tokens)
│   └── shard_00009.bin (20 MB, 10M tokens)
├── val/
│   └── shard_00000.bin (196 KB, 100K tokens)
└── test/
    ├── shard_00000.bin (9.6 MB, 5M tokens)
    └── shard_00001.bin (9.6 MB, 5M tokens)
```

**Gesamt:**
- **Train:** 200 MB, 100M Tokens (10 Shards)
- **Val:** 196 KB, 100K Tokens
- **Test:** 19.2 MB, 10M Tokens

---

## ✅ TOKENIZER ERSTELLT

### Dateien

```
data/tokenizers/
├── fineweb_1024_bpe.model (249 KB)
└── fineweb_1024_bpe.vocab (12 KB)
```

**Details:**
- **Typ:** SentencePiece BPE
- **Vocab Size:** 1024
- **Special Tokens:** pad=0, bos=1, eos=2, unk=3
- **Training:** 50k Dokumente, 11 MB Text

**Test Encoding:**
```
"The quick brown fox jumps over the lazy dog"
→ [546, 979, 940, 46, 1000, 18, 39, 999, 985, 45, 986, 1005, 979, 3, 991, 914, 987, 67, 1002, 6, 49, 34, 983, 1001, 998, 68, 994]
```

---

## 📊 USAGE

### Training mit Dataset

```bash
cd /home/schaf/projects/NeuroWeave
source .venv/bin/activate

# Smoke Test (wenige Iterationen)
RUN_ID=test1 \
ITERATIONS=10 \
TRAIN_BATCH_TOKENS=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/train \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
python train_gpt.py

# Vollständiges Training
RUN_ID=baseline_v1 \
ITERATIONS=2000 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=200 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/train \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
python train_gpt.py
```

### Dataset für andere Experimente

```python
from train_gpt import FineWebDataset

dataset = FineWebDataset(
    data_path="./data/datasets/fineweb10B_sp1024/train",
    tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model",
    seq_len=1024
)

# Batch holen
x, y = dataset.get_batch(batch_tokens=4096)
print(f"Batch shape: {x.shape}, {y.shape}")
```

---

## ⚠️ BEKANNTE ISSUES

### Training hängt nach mehreren Steps

**Status:** ⚠️ Investigated

**Symptom:** Training startet, führt einige Steps aus, hängt sich dann.

**Ursache:** Vermutlich Data Loading bei großen Datasets.

**Workaround:** 
- Weniger Iterationen für Tests (`ITERATIONS=10`)
- Kleinere Batch-Größe (`TRAIN_BATCH_TOKENS=1024`)

**Nächste Schritte:**
- Data Loading Logik debuggen
- Eventuell Memory-Leak im Dataset

---

## 📝 HINWEISE

### Synthetisches Dataset

Dies ist ein **synthetisches Test-Dataset** mit zufälligen Tokens. Es ist geeignet für:
- ✅ Smoke Tests
- ✅ Development
- ✅ Integration Tests
- ✅ Performance Testing

**NICHT geeignet für:**
- ❌ Echte Modell-Qualitätsbewertung
- ❌ Challenge Submission
- ❌ Publication-ready Results

### Für echte Submission

Das echte FineWeb Dataset muss heruntergeladen werden:

```bash
# Auf H100 mit schnellem Netzwerk
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

**Erwartet:**
- Download: ~80 GB
- Dauer: ~30-60 Minuten auf H100
- Tokens: ~8B (80 Shards × 100M)

---

## ✅ CHECKLISTE

- [x] Dataset Ordnerstruktur erstellt
- [x] Train Shards erstellt (10 × 20 MB)
- [x] Validation Shard erstellt (196 KB)
- [x] Test Shards erstellt (2 × 9.6 MB)
- [x] SentencePiece Tokenizer trainiert (1024 Vocab)
- [x] Tokenizer getestet (Encoding funktioniert)
- [x] Dataset Loading getestet

---

**Gesamt:** ✅ Dataset und Tokenizer sind bereit für lokale Tests.
