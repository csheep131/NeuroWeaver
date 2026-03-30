#!/bin/bash
#
# smoke_test.sh - Lokaler Smoke Test für SOTA train_gpt.py
#
# Prüft ob die SOTA train_gpt.py korrekt läuft (RTX 3050 / SDPA Fallback)
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "════════════════════════════════════════════════"
echo "  NeuroWeave SOTA Smoke Test"
echo "════════════════════════════════════════════════"
echo

# Python + venv
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
    source .venv/bin/activate
    log_info "Verwende venv: .venv"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
    log_warn "Keine venv, verwende system python3"
else
    log_error "Python nicht gefunden"
    exit 1
fi

# ── Test 1: Dependencies ──
log_info "Test 1: Dependencies..."
$PYTHON -c "
import torch, numpy, sentencepiece
print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
" 2>&1 && log_success "Dependencies OK" || { log_error "Dependencies fehlen"; exit 1; }

# ── Test 2: SOTA train_gpt.py Integrität ──
log_info "Test 2: SOTA train_gpt.py Integrität..."
LINES=$(wc -l < train_gpt.py)
if [[ $LINES -lt 1500 ]]; then
    log_error "train_gpt.py hat nur $LINES Zeilen — das ist NICHT der SOTA!"
    log_error "SOTA hat ~1920 Zeilen. Siehe SOTA_REFERENCE.md"
    exit 1
fi

$PYTHON -c "
import ast, sys
with open('train_gpt.py') as f:
    tree = ast.parse(f.read())
classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
required = ['Muon', 'GPT', 'CausalSelfAttention', 'SmearGate', 'BigramHashEmbedding', 'Block', 'MLP']
missing = [c for c in required if c not in classes]
if missing:
    print(f'  ❌ Fehlende Klassen: {missing}')
    sys.exit(1)
print(f'  ✅ Alle SOTA-Klassen vorhanden ({len(classes)} Klassen, $LINES Zeilen)')
" 2>&1 && log_success "SOTA Integrität OK" || { log_error "SOTA-Klassen fehlen!"; exit 1; }

# ── Test 3: Dataset + Tokenizer ──
log_info "Test 3: Dataset + Tokenizer..."
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

if [[ -d "$DATA_PATH" ]]; then
    TRAIN_SHARDS=$(ls -1 "$DATA_PATH"/fineweb_train_*.bin 2>/dev/null | wc -l)
    VAL_SHARDS=$(ls -1 "$DATA_PATH"/fineweb_val_*.bin 2>/dev/null | wc -l)
    log_success "Dataset: $TRAIN_SHARDS train, $VAL_SHARDS val shards"
else
    log_error "Dataset nicht gefunden: $DATA_PATH"
    echo "  Fix: python3 data/cached_challenge_fineweb.py --variant sp1024"
    exit 1
fi

if [[ -f "$TOKENIZER_PATH" ]]; then
    log_success "Tokenizer: $TOKENIZER_PATH"
else
    log_error "Tokenizer nicht gefunden: $TOKENIZER_PATH"
    exit 1
fi

# ── Test 4: SDPA Fallback ──
log_info "Test 4: Flash Attention / SDPA Fallback..."
$PYTHON -c "
try:
    from flash_attn_interface import flash_attn_func
    print('  ⚡ Flash Attention 3: verfügbar (H100-Pfad)')
except ImportError:
    print('  🔄 Flash Attention 3: NICHT verfügbar → SDPA Fallback aktiv (OK für RTX 3050)')
" 2>&1
log_success "Attention Backend OK"

# ── Test 5: Quick Training (50 Steps) ──
log_info "Test 5: Quick Training (50 Steps)..."
echo "  Starte SOTA-Training mit minimaler Config..."

RUN_ID="smoke_$(date +%Y%m%d_%H%M%S)" \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
VOCAB_SIZE=1024 \
ITERATIONS=50 \
TRAIN_BATCH_TOKENS=16384 \
TRAIN_SEQ_LEN=512 \
EVAL_SEQ_LEN=512 \
VAL_BATCH_SIZE=16384 \
VAL_LOSS_EVERY=50 \
MAX_WALLCLOCK_SECONDS=120 \
WARMDOWN_ITERS=10 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=10 \
EVAL_STRIDE=0 \
TTT_ENABLED=0 \
SWA_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tail -30 | sed 's/^/  /'

EXIT_CODE=${PIPESTATUS[0]}
if [[ $EXIT_CODE -eq 0 ]]; then
    log_success "Training OK"
else
    log_error "Training fehlgeschlagen (exit code: $EXIT_CODE)"
    exit 1
fi

echo
echo "════════════════════════════════════════════════"
log_success "Alle SOTA Smoke Tests bestanden!"
echo "════════════════════════════════════════════════"
echo
echo "Nächste Schritte:"
echo "  1. Gezielte Verbesserung in train_gpt.py machen"
echo "  2. Smoke Test erneut laufen lassen"
echo "  3. Vollständigen Run starten:"
echo "     RUN_ID=mein_experiment \\"
echo "     DATA_PATH=$DATA_PATH \\"
echo "     TOKENIZER_PATH=$TOKENIZER_PATH \\"
echo "     VOCAB_SIZE=1024 \\"
echo "     TRAIN_BATCH_TOKENS=32768 \\"
echo "     TRAIN_SEQ_LEN=1024 \\"
echo "     MAX_WALLCLOCK_SECONDS=7200 \\"
echo "     torchrun --standalone --nproc_per_node=1 train_gpt.py"
echo
