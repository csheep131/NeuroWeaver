#!/bin/bash
# Schedule all training runs for GPU execution

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo " Scheduling all training runs for GPU"
echo "========================================"

# Phase 1: Control & Baselines
echo ""
echo " Phase 1: Control & Baselines"
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run001_control.yaml" --priority 10 --gpu-memory 4096
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run002_hash.yaml" --priority 9 --gpu-memory 4096
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run003_xsa.yaml" --priority 8 --gpu-memory 4096
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run004_leakyrelu.yaml" --priority 8 --gpu-memory 4096
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run005_mixed_quant.yaml" --priority 7 --gpu-memory 4096

# Phase 2: Feature Combos
echo ""
echo " Phase 2: Feature Combinations"
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run006_film.yaml" --priority 6 --gpu-memory 4096
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run007_ttt.yaml" --priority 6 --gpu-memory 4096
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run009_gqa.yaml" --priority 6 --gpu-memory 4096
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run010_recurrence.yaml" --priority 6 --gpu-memory 4096

# Phase 3: Best Combos (lower priority, depends on earlier results)
echo ""
echo " Phase 3: Best Combinations"
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run016_best_combo_a.yaml" --priority 5 --gpu-memory 4096
python3 "$SCRIPT_DIR/training_queue.py" add "$PROJECT_ROOT/configs/runs/run017_best_combo_quantized.yaml" --priority 4 --gpu-memory 4096

echo ""
echo " All runs scheduled!"
echo ""
echo "View queue: python3 scripts/gpu_training/training_queue.py list"
echo "Run next: python3 scripts/gpu_training/training_queue.py run"
echo "Run all: python3 scripts/gpu_training/training_queue.py run --all"
echo ""
echo "Or use the watch script:"
echo " python3 scripts/gpu_training/watch_and_train.py"
