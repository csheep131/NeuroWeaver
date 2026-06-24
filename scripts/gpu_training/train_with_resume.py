#!/usr/bin/env python3
"""Training mit Checkpointing und Auto-Resume.

Ermöglicht unterbrochene Trainings fortzusetzen.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from core.config import load_config
from core.registry import RunRegistry
from core.seed import set_seed
from core.logging import RunLogger
from data import create_dataloader, create_tokenizer
from train.pytorch_model import create_model
from train.optimizer_factory import create_optimizer
from train.scheduler import create_scheduler
from train.trainer import Trainer, TrainConfig
from eval.bpb_eval import BPBEvaluator


class CheckpointManager:
"""Manages training checkpoints."""

def __init__(self, run_id: str, checkpoint_dir: str = "checkpoints"):
self.run_id = run_id
self.checkpoint_dir = Path(checkpoint_dir) / run_id
self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
self.latest_checkpoint = self.checkpoint_dir / "latest.pt"
self.best_checkpoint = self.checkpoint_dir / "best.pt"

def save(self, step: int, model, optimizer, scheduler, metrics: dict, is_best: bool = False):
"""Save checkpoint."""
checkpoint = {
"step": step,
"model_state": model.state_dict(),
"optimizer_state": optimizer.state_dict(),
"scheduler_state": scheduler.state_dict() if scheduler else None,
"metrics": metrics,
"timestamp": time.time(),
}

# Save latest
torch.save(checkpoint, self.latest_checkpoint)

# Save periodic checkpoint
periodic_path = self.checkpoint_dir / f"checkpoint_{step:06d}.pt"
torch.save(checkpoint, periodic_path)

# Save best
if is_best:
torch.save(checkpoint, self.best_checkpoint)

return periodic_path

def load(self, model, optimizer=None, scheduler=None, checkpoint_path: str = None):
"""Load checkpoint. Returns step and metrics."""
path = Path(checkpoint_path) if checkpoint_path else self.latest_checkpoint

if not path.exists():
return 0, {}

print(f" Lade Checkpoint: {path}")
checkpoint = torch.load(path, map_location="cpu")

model.load_state_dict(checkpoint["model_state"])

if optimizer and checkpoint.get("optimizer_state"):
optimizer.load_state_dict(checkpoint["optimizer_state"])

if scheduler and checkpoint.get("scheduler_state"):
scheduler.load_state_dict(checkpoint["scheduler_state"])

step = checkpoint.get("step", 0)
metrics = checkpoint.get("metrics", {})

print(f" Fortsetzen ab Step {step}")
return step, metrics

def exists(self) -> bool:
"""Check if checkpoint exists."""
return self.latest_checkpoint.exists()


class GracefulExitHandler:
"""Handle graceful shutdown on SIGINT/SIGTERM."""

def __init__(self):
self.should_exit = False
signal.signal(signal.SIGINT, self._signal_handler)
signal.signal(signal.SIGTERM, self._signal_handler)

def _signal_handler(self, signum, frame):
print(f"\n Signal {signum} empfangen - Beende nach aktuellem Step...")
self.should_exit = True

def check(self) -> bool:
return self.should_exit


def train_with_checkpointing(config_path: str, resume: bool = True, device: str = None):
"""Train with checkpointing support.

Args:
config_path: Path to config file
resume: Whether to resume from checkpoint
device: Device to use (auto-detect if None)
"""
# Load config
config = load_config(config_path)
run_id = config.run_id

# Setup
logger = RunLogger(run_id)
registry = RunRegistry()
checkpoint_manager = CheckpointManager(run_id)
exit_handler = GracefulExitHandler()

# Device
if device is None:
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.log_info(f" Training Start: {run_id}")
logger.log_info(f" Device: {device}")
logger.log_info(f" Checkpoint: {checkpoint_manager.checkpoint_dir}")

# Set seed
set_seed(config.seed)

# Register run
registry.register(
run_id=run_id,
config_hash=config.config_hash,
seed=config.seed,
)
registry.start_run(run_id)

# Create tokenizer
tokenizer_cfg = config.to_dict().get("tokenizer", {})
tokenizer = create_tokenizer(
tokenizer_type=tokenizer_cfg.get("type", "byte"),
vocab_size=tokenizer_cfg.get("vocab_size", 256),
)

# Create model
model_cfg = config.to_dict()
model_cfg["model"]["vocab_size"] = tokenizer.vocab_size
model = create_model(model_cfg)
model.to(device)
num_params = model.num_parameters_millions()
logger.log_info(f" Model: {num_params:.2f}M parameters")

# Create optimizer
training_cfg = config.training
optimizer = create_optimizer(
model,
optimizer_type=str(training_cfg.get("optimizer", "adamw")),
learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
weight_decay=float(training_cfg.get("weight_decay", 0.1)),
)

# Create scheduler
num_steps = int(training_cfg.get("num_steps", 10000))
warmup_steps = int(training_cfg.get("warmup_steps", 100))
scheduler = create_scheduler(
optimizer,
scheduler_type=str(training_cfg.get("scheduler", "cosine")),
num_warmup_steps=warmup_steps,
num_training_steps=num_steps,
)

# Load checkpoint if resuming
start_step = 0
best_val_bpb = float("inf")

if resume and checkpoint_manager.exists():
start_step, metrics = checkpoint_manager.load(model, optimizer, scheduler)
best_val_bpb = metrics.get("best_val_bpb", float("inf"))

# Create data loader
data_config = {
"train_data_path": "",
"eval_data_path": "",
"seq_len": training_cfg.get("seq_len", 128),
"batch_size": training_cfg.get("batch_size", 32),
"tokenizer_type": tokenizer_cfg.get("type", "byte"),
"tokenizer_vocab_size": tokenizer.vocab_size,
"shuffle": True,
"seed": config.seed,
}
train_loader = create_dataloader(data_config, tokenizer)

# Create trainer
train_config = TrainConfig.from_dict(training_cfg)
train_config.device = device
train_config.num_steps = num_steps
trainer = Trainer(model, optimizer, train_config, logger)

# Create evaluator
bpb_evaluator = BPBEvaluator(tokenizer)

# Training loop with checkpointing
logger.log_info(f" Training: {start_step}/{num_steps} steps")

train_iter = iter(train_loader)
global_step = start_step

try:
while global_step < num_steps:
# Check for exit signal
if exit_handler.check():
logger.log_info(" Speichere Checkpoint vor Beenden...")
checkpoint_manager.save(
global_step, model, optimizer, scheduler,
{"best_val_bpb": best_val_bpb}
)
break

# Get batch
try:
batch = next(train_iter)
except StopIteration:
train_iter = iter(train_loader)
batch = next(train_iter)

# Training step
loss = trainer.train_step(batch)
if scheduler:
scheduler.step()

global_step += 1

# Log
if global_step % 10 == 0:
logger.log_step(global_step, {"loss": loss})

# Checkpoint
if global_step % 1000 == 0:
checkpoint_manager.save(
global_step, model, optimizer, scheduler,
{"best_val_bpb": best_val_bpb}
)
logger.log_info(f" Checkpoint gespeichert: Step {global_step}")

# Evaluation
if global_step % training_cfg.get("eval_every", 500) == 0:
logger.log_info(f" Evaluation bei Step {global_step}...")
model.eval()
eval_loader = create_dataloader(data_config, tokenizer)
eval_result = bpb_evaluator.compute_bpb(model, eval_loader, device=device)
val_bpb = eval_result.val_bpb

logger.log_eval(global_step, {"val_bpb": val_bpb})

# Save best
if val_bpb < best_val_bpb:
best_val_bpb = val_bpb
checkpoint_manager.save(
global_step, model, optimizer, scheduler,
{"best_val_bpb": best_val_bpb, "val_bpb": val_bpb},
is_best=True
)
logger.log_info(f" Neuer Bestwert: {val_bpb:.4f} BPB")

model.train()

# Final evaluation
logger.log_info(" Finale Evaluation...")
model.eval()
eval_loader = create_dataloader(data_config, tokenizer)
eval_result = bpb_evaluator.compute_bpb(model, eval_loader, device=device)
val_bpb = eval_result.val_bpb

# Complete run
metrics = {
"val_bpb": val_bpb,
"steps_completed": global_step,
"best_val_bpb": best_val_bpb,
}
registry.complete_run(run_id, metrics)

logger.log_info(f" Training abgeschlossen!")
logger.log_info(f" Final BPB: {val_bpb:.4f}")
logger.log_info(f" Best BPB: {best_val_bpb:.4f}")

return 0

except Exception as e:
logger.log_error(f" Training fehlgeschlagen: {e}")
# Save emergency checkpoint
checkpoint_manager.save(
global_step, model, optimizer, scheduler,
{"best_val_bpb": best_val_bpb, "error": str(e)}
)
registry.fail_run(run_id, str(e))
raise


def main():
parser = argparse.ArgumentParser(description="Training mit Checkpointing")
parser.add_argument("--config", type=str, required=True,
help="Pfad zur Config-Datei")
parser.add_argument("--no-resume", action="store_true",
help="Nicht von Checkpoint fortsetzen")
parser.add_argument("--device", type=str, default=None,
help="Device (cuda/cpu)")

args = parser.parse_args()

try:
return train_with_checkpointing(
config_path=args.config,
resume=not args.no_resume,
device=args.device,
)
except Exception as e:
print(f" Fehler: {e}")
import traceback
traceback.print_exc()
return 1


if __name__ == "__main__":
sys.exit(main())
