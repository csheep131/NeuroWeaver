#!/usr/bin/env python3
"""Training Queue - Verwaltet mehrere Trainings-Runs.

Features:
- Reihenfolge von Runs planen
- Automatisches Warten auf freie GPU
- Parallele Ausführung (wenn mehrere GPUs)
- Status-Überwachung
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class QueueItem:
    """Ein Item in der Training-Queue."""
    config_path: str
    run_id: str
    priority: int = 0
    gpu_memory_required_mb: int = 4096
    status: str = "pending"  # pending, waiting, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "QueueItem":
        return cls(**d)


class TrainingQueue:
    """Manages a queue of training runs."""
    
    QUEUE_FILE = Path("training_queue.json")
    
    def __init__(self):
        self.items: List[QueueItem] = []
        self.load()
    
    def load(self):
        """Load queue from file."""
        if self.QUEUE_FILE.exists():
            with open(self.QUEUE_FILE) as f:
                data = json.load(f)
                self.items = [QueueItem.from_dict(item) for item in data.get("items", [])]
    
    def save(self):
        """Save queue to file."""
        with open(self.QUEUE_FILE, "w") as f:
            json.dump({
                "items": [item.to_dict() for item in self.items],
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2)
    
    def add(self, config_path: str, run_id: str = None, priority: int = 0, gpu_memory_mb: int = 4096):
        """Add a run to the queue."""
        if run_id is None:
            # Extract run_id from config
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
                run_id = cfg.get("run_id", "unknown")
        
        item = QueueItem(
            config_path=config_path,
            run_id=run_id,
            priority=priority,
            gpu_memory_required_mb=gpu_memory_mb,
        )
        self.items.append(item)
        self.save()
        print(f"✅ Added to queue: {run_id} (priority: {priority})")
    
    def remove(self, run_id: str):
        """Remove a run from the queue."""
        self.items = [item for item in self.items if item.run_id != run_id]
        self.save()
    
    def list(self):
        """List all queue items."""
        if not self.items:
            print("📭 Queue is empty")
            return
        
        print("\n📋 Training Queue:")
        print("-" * 80)
        print(f"{'Status':<12} {'Priority':<10} {'Run ID':<25} {'Config':<30}")
        print("-" * 80)
        
        # Sort by priority
        sorted_items = sorted(self.items, key=lambda x: (-x.priority, x.run_id))
        
        for item in sorted_items:
            print(f"{item.status:<12} {item.priority:<10} {item.run_id:<25} {Path(item.config_path).name:<30}")
        
        print("-" * 80)
        print(f"Total: {len(self.items)} runs")
        
        # Summary by status
        by_status = {}
        for item in self.items:
            by_status[item.status] = by_status.get(item.status, 0) + 1
        
        print(f"Status: ", end="")
        for status, count in by_status.items():
            print(f"{status}={count} ", end="")
        print()
    
    def clear(self):
        """Clear all completed/failed runs."""
        self.items = [item for item in self.items if item.status in ("pending", "waiting", "running")]
        self.save()
        print("🧹 Cleared completed/failed runs")
    
    def run_next(self, dry_run: bool = False):
        """Run the next item in queue."""
        # Find next pending item
        pending = [item for item in self.items if item.status == "pending"]
        if not pending:
            print("✨ No pending runs in queue")
            return False
        
        # Sort by priority
        next_item = sorted(pending, key=lambda x: (-x.priority, x.run_id))[0]
        
        print(f"\n🚀 Next run: {next_item.run_id}")
        print(f"   Config: {next_item.config_path}")
        print(f"   Required VRAM: {next_item.gpu_memory_required_mb}MB")
        
        if dry_run:
            print("   (Dry run - not executing)")
            return True
        
        # Wait for GPU
        next_item.status = "waiting"
        self.save()
        
        print("   Waiting for GPU...")
        wait_result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "wait_for_gpu.py"),
                "--memory", str(next_item.gpu_memory_required_mb),
                "--utilization", "80",
                "--interval", "10",
            ],
            capture_output=True,
        )
        
        if wait_result.returncode != 0:
            print("❌ GPU wait failed")
            next_item.status = "failed"
            next_item.error = "GPU not available"
            self.save()
            return False
        
        # Run training
        next_item.status = "running"
        next_item.started_at = datetime.now().isoformat()
        self.save()
        
        print("   Starting training...")
        train_result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "train_with_resume.py"),
                "--config", next_item.config_path,
            ]
        )
        
        # Update status
        if train_result.returncode == 0:
            next_item.status = "completed"
            print(f"✅ {next_item.run_id} completed")
        else:
            next_item.status = "failed"
            next_item.error = f"Exit code {train_result.returncode}"
            print(f"❌ {next_item.run_id} failed")
        
        next_item.completed_at = datetime.now().isoformat()
        self.save()
        
        return True
    
    def run_all(self, dry_run: bool = False):
        """Run all pending items."""
        print(f"🎯 Running all pending runs (dry_run={dry_run})")
        
        while True:
            result = self.run_next(dry_run=dry_run)
            if not result:
                break
            
            if dry_run:
                break  # Only show first item in dry run
            
            # Small delay between runs
            time.sleep(5)
        
        print("\n🏁 Queue processing complete")
        self.list()


def main():
    parser = argparse.ArgumentParser(description="Training Queue Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add run to queue")
    add_parser.add_argument("config", type=str, help="Path to config file")
    add_parser.add_argument("--run-id", type=str, default=None, help="Run ID")
    add_parser.add_argument("--priority", type=int, default=0, help="Priority (higher first)")
    add_parser.add_argument("--gpu-memory", type=int, default=4096, help="Required GPU memory (MB)")
    
    # List command
    subparsers.add_parser("list", help="List queue")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove run from queue")
    remove_parser.add_argument("run_id", type=str, help="Run ID to remove")
    
    # Clear command
    subparsers.add_parser("clear", help="Clear completed runs")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run queue")
    run_parser.add_argument("--all", action="store_true", help="Run all pending")
    run_parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    
    args = parser.parse_args()
    
    queue = TrainingQueue()
    
    if args.command == "add":
        queue.add(args.config, args.run_id, args.priority, args.gpu_memory)
    
    elif args.command == "list":
        queue.list()
    
    elif args.command == "remove":
        queue.remove(args.run_id)
        print(f"🗑️ Removed: {args.run_id}")
    
    elif args.command == "clear":
        queue.clear()
    
    elif args.command == "run":
        if args.all:
            queue.run_all(dry_run=args.dry_run)
        else:
            queue.run_next(dry_run=args.dry_run)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
