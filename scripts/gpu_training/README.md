# GPU Training Scripts

Automatisiertes GPU-Training für die Ablation Machine.

## Schnellstart

```bash
# 1. Alle Runs zur Queue hinzufügen
bash scripts/gpu_training/schedule_all_runs.sh

# 2. Queue anzeigen
python3 scripts/gpu_training/training_queue.py list

# 3. Nächsten Run starten (wartet automatisch auf GPU)
python3 scripts/gpu_training/training_queue.py run

# 4. ODER: Alle Runs automatisch ausführen
python3 scripts/gpu_training/training_queue.py run --all

# 5. ODER: Watcher starten (überwacht GPU kontinuierlich)
python3 scripts/gpu_training/watch_and_train.py
```

## Skripte

### `wait_for_gpu.py`
Wartet auf freie GPU-Ressourcen.

```bash
# Warte auf 4GB freien VRAM
python3 scripts/gpu_training/wait_for_gpu.py --memory 4096

# Führe Befehl aus wenn GPU verfügbar
python3 scripts/gpu_training/wait_for_gpu.py --memory 4096 --command "python -m runs.run --config ..."
```

### `train_with_resume.py`
Training mit Checkpointing und Auto-Resume.

```bash
# Starte oder setze fort
python3 scripts/gpu_training/train_with_resume.py --config configs/runs/run001_control.yaml

# Von vorne beginnen
python3 scripts/gpu_training/train_with_resume.py --config configs/runs/run001_control.yaml --no-resume

# Spezifisches Device
python3 scripts/gpu_training/train_with_resume.py --config configs/runs/run001_control.yaml --device cuda:0
```

**Features:**
- Automatische Checkpoints alle 1000 Steps
- Speichert besten Checkpoint separat
- Graceful Shutdown (Ctrl+C)
- Fortsetzen nach Unterbrechung

### `training_queue.py`
Verwaltet mehrere Training-Runs.

```bash
# Run hinzufügen
python3 scripts/gpu_training/training_queue.py add configs/runs/run001_control.yaml --priority 10

# Liste anzeigen
python3 scripts/gpu_training/training_queue.py list

# Nächsten Run ausführen
python3 scripts/gpu_training/training_queue.py run

# Alle Runs ausführen
python3 scripts/gpu_training/training_queue.py run --all

# Dry-run (zeigt was laufen würde)
python3 scripts/gpu_training/training_queue.py run --dry-run

# Run entfernen
python3 scripts/gpu_training/training_queue.py remove run001_control

# Abgeschlossene Runs löschen
python3 scripts/gpu_training/training_queue.py clear
```

## GPU-RAM Anforderungen

| Config | Parameter | Batch | GPU-RAM |
|--------|-----------|-------|---------|
| Standard | 38.4M | 32 | ~3.4 GB |
| GPU-Optimiert | 38.4M | 16 | ~2.5 GB |
| Mixed Precision | 38.4M | 24 | ~2.0 GB |

## Status-Überwachung

```bash
# GPU-Status
nvidia-smi

# Queue-Status
python3 scripts/gpu_training/training_queue.py list

# Aktive Trainings
ps aux | grep train_with_resume

# Logs anzeigen
tail -f results/run001_control/train_log.jsonl
```

## Troubleshooting

**GPU nicht gefunden:**
```bash
# Prüfe CUDA-Installation
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

**OOM (Out of Memory):**
- Reduziere `batch_size` in Config
- Nutze Mixed Precision (`use_amp: true`)
- Verringere `seq_len`

**Training unterbrochen:**
```bash
# Automatisch fortsetzen
python3 scripts/gpu_training/train_with_resume.py --config configs/runs/run001_control.yaml
```

## Checkpoints

Checkpoints werden gespeichert in:
```
checkpoints/{run_id}/
latest.pt # Letzter Checkpoint
best.pt # Bester Checkpoint (niedrigster Val BPB)
checkpoint_*.pt # Periodische Checkpoints
```

Checkpoint-Struktur:
```python
{
"step": 5000, # Aktueller Step
"model_state": {...}, # Modell-Gewichte
"optimizer_state": {...}, # Optimizer-State
"scheduler_state": {...}, # Scheduler-State
"metrics": {"best_val_bpb": ...} # Metriken
}
```
