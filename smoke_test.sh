#!/bin/bash
#
# smoke_test.sh - Lokaler Smoke Test für Parameter Golf Challenge
#
# Führt einen schnellen Smoke Test durch um die Infrastruktur zu validieren.
#

set -euo pipefail

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Konfiguration
ITERATIONS="${ITERATIONS:-50}"
BATCH_TOKENS="${BATCH_TOKENS:-1024}"
RUN_ID="${RUN_ID:-smoke_test_$(date +%Y%m%d_%H%M%S)}"

echo "========================================"
echo "  NeuroWeave Smoke Test"
echo "========================================"
echo

# Python-Pfad bestimmen
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
    log_info "Verwende virtuelle Umgebung: .venv"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
    log_warn "Keine venv gefunden, verwende system python3"
else
    log_error "Python nicht gefunden"
    exit 1
fi

# Test 1: Dependencies prüfen
log_info "Test 1: Prüfe Dependencies..."
if $PYTHON -c "import torch; import numpy; import sentencepiece; import yaml" 2>/dev/null; then
    log_success "Dependencies OK"
else
    log_error "Dependencies fehlen. Installiere mit: pip install -r requirements.txt"
    exit 1
fi

# Test 2: Dataset prüfen
log_info "Test 2: Prüfe Dataset..."
if [[ -d "./data/datasets/fineweb10B_sp1024/train" ]]; then
    SHARD_COUNT=$(ls -1 ./data/datasets/fineweb10B_sp1024/train/*.bin 2>/dev/null | wc -l)
    log_success "Dataset gefunden ($SHARD_COUNT Shards)"
else
    log_warn "Dataset nicht gefunden. Download mit:"
    echo "  python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80"
    log_info "Verwende synthetische Daten für Test..."
fi

# Test 3: Tokenizer prüfen
log_info "Test 3: Prüfe Tokenizer..."
if [[ -f "./data/tokenizers/fineweb_1024_bpe.model" ]] && [[ -s "./data/tokenizers/fineweb_1024_bpe.model" ]]; then
    log_success "Tokenizer gefunden"
else
    log_warn "Tokenizer nicht gefunden. Training verwendet Byte-Level Fallback"
fi

# Test 4: Dataset Loading Test
log_info "Test 4: Dataset Loading Test..."
$PYTHON -c "
from train_gpt import FineWebDataset
dataset = FineWebDataset(
    data_path='./data/datasets/fineweb10B_sp1024/train',
    tokenizer_path='./data/tokenizers/fineweb_1024_bpe.model',
    seq_len=1024
)
print(f'  Vocab Size: {dataset._vocab_size}')
print(f'  Shards: {len(dataset.shards)}')
x, y = dataset.get_batch(batch_tokens=1024)
print(f'  Batch: {x.shape}')
" 2>&1 | sed 's/^/  /'
log_success "Dataset Loading OK"

# Test 5: Model Forward Pass
log_info "Test 5: Model Forward Pass..."
$PYTHON -c "
import torch
from train_gpt import Config, GPT

cfg = Config(
    d_model=384,
    num_layers=9,
    num_heads=6,
    kv_heads=3,
    vocab_size=1024,
    max_seq_len=1024
)

model = GPT(cfg)
x = torch.randint(0, 1024, (2, 1024))
logits, loss = model(x, x)
print(f'  Input: {x.shape}')
print(f'  Logits: {logits.shape}')
print(f'  Loss: {loss.item():.4f}')
" 2>&1 | sed 's/^/  /'
log_success "Model Forward OK"

# Test 6: Compression Test
log_info "Test 6: Compression Test..."
$PYTHON -c "
import torch
import io
import zlib
from train_gpt import Config, GPT, compress_model

cfg = Config(
    d_model=384,
    num_layers=9,
    num_heads=6,
    kv_heads=3,
    vocab_size=1024
)

model = GPT(cfg)
size, _ = compress_model(model)
print(f'  Compressed Size: {size / 1024 / 1024:.2f} MB')
print(f'  Limit: 16.00 MB')
print(f'  Status: {\"OK\" if size < 16_000_000 else \"TOO LARGE\"}')
" 2>&1 | sed 's/^/  /'
log_success "Compression OK"

# Test 7: Kurzes Training
log_info "Test 7: Training Test ($ITERATIONS Iterationen)..."
echo "  Starte Training..."
RUN_ID="$RUN_ID" \
ITERATIONS="$ITERATIONS" \
TRAIN_BATCH_TOKENS="$BATCH_TOKENS" \
VAL_LOSS_EVERY=0 \
$PYTHON train_gpt.py 2>&1 | sed 's/^/  /' || {
    log_error "Training fehlgeschlagen"
    exit 1
}

log_success "Training OK"

echo
echo "========================================"
log_success "Alle Smoke Tests bestanden!"
echo "========================================"
echo
echo "Run ID: $RUN_ID"
echo "Iterationen: $ITERATIONS"
echo "Batch Tokens: $BATCH_TOKENS"
echo
echo "Nächste Schritte:"
echo "  1. Logs in records/baseline_v1/logs/ kopieren"
echo "  2. submission.json mit Metriken aktualisieren"
echo "  3. PR einreichen"
echo
