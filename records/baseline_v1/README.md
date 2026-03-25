# NeuroWeave Baseline v1

**Submission for Parameter Golf Challenge**

## Summary

| Metric | Value |
|--------|-------|
| **val_bpb** | _pending_ |
| **Compressed Size** | ~12.9 MB ✅ |
| **Parameters** | ~15M |
| **Training Time** | < 10 min on 8xH100 |
| **Artifact Size** | < 16 MB ✅ |

## Architecture

This submission uses a 9-layer transformer optimized for the 16MB constraint:

### Core Architecture
- **Layers:** 9
- **Model Dimension:** 384
- **Attention Heads:** 6 (GQA with 3 KV heads)
- **MLP Ratio:** 4x
- **Vocabulary:** 1024 (SentencePiece BPE)
- **Sequence Length:** 1024

### Key Features

| Feature | Setting | Description |
|---------|---------|-------------|
| **Activation** | LeakyReLU² | LeakyReLU (leakiness=0.5) squared |
| **Attention** | GQA | Grouped Query Attention (6Q, 3KV) |
| **Positional** | RoPE | Rotary Positional Embeddings |
| **Weight Tying** | Yes | Input/output embeddings tied |

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 3e-4 |
| **Weight Decay** | 0.1 |
| **Warmup Steps** | 100 |
| **Max Steps** | 2000 |
| **Batch Size** | 8192 tokens |
| **Gradient Clip** | 1.0 |
| **Beta** | (0.9, 0.95) |

## How to Reproduce

### Prerequisites

```bash
# Install dependencies
pip install torch numpy sentencepiece datasets tqdm

# Or use the project requirements
pip install -e .
```

### Download Dataset

```bash
# Download FineWeb with 1024-token vocabulary
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

### Training

#### Single GPU (Smoke Test)

```bash
RUN_ID=baseline_v1_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
python train_gpt.py
```

#### Full Training on 8xH100

```bash
RUN_ID=baseline_v1 \
ITERATIONS=2000 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=200 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=600 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/train \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

#### Local Testing (Apple Silicon)

```bash
RUN_ID=mlx_baseline \
ITERATIONS=200 \
python train_gpt_mlx.py
```

## Expected Results

Based on challenge leaderboard:

| Metric | Target | Baseline Expectation |
|--------|--------|---------------------|
| val_bpb | < 1.50 | ~1.22 (naive baseline) |
| compressed_size | < 16MB | ~12-14 MB |
| ms/step | < 50ms | ~30-40ms on H100 |

## Files

```
records/baseline_v1/
├── README.md           # This file
├── submission.json     # Submission metadata
├── train_gpt.py        # Symlink to ../../train_gpt.py
└── logs/
    ├── run1.log        # Training log (seed 1)
    ├── run2.log        # Training log (seed 2)
    └── run3.log        # Training log (seed 3)
```

## Notes

- This is an initial baseline submission
- Actual run results will be populated after training on H100
- 3-seed validation recommended for statistical significance

## Author

- **Name:** NeuroWeave Team
- **GitHub:** @neuro-weave
- **Date:** 2026-03-25

## License

MIT
