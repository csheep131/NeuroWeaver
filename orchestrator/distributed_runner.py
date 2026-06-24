#!/usr/bin/env python3
"""
Distributed Runner für NeuroWeave Phase 5.

Verteilte Run-Ausführung mit Multi-GPU Support.

Features:
- Multi-GPU Support
- Load Balancing
- Fault Tolerance (Retry bei Failure)
- Progress Tracking
- Result Aggregation
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import RunRegistry


class WorkerStatus(Enum):
"""Status eines Workers."""

IDLE = "idle"
RUNNING = "running"
BUSY = "busy"
OFFLINE = "offline"
ERROR = "error"


class RunStatus(Enum):
"""Status eines Runs."""

PENDING = "pending"
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
RETRYING = "retrying"


@dataclass
class WorkerConfig:
"""Konfiguration für Worker."""

worker_id: str
gpu_id: int
max_concurrent_runs: int = 1
memory_limit_mb: int = 8000

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"worker_id": self.worker_id,
"gpu_id": self.gpu_id,
"max_concurrent_runs": self.max_concurrent_runs,
"memory_limit_mb": self.memory_limit_mb,
}


@dataclass
class WorkerState:
"""Zustand eines Workers."""

config: WorkerConfig
status: WorkerStatus = WorkerStatus.IDLE
current_runs: List[str] = field(default_factory=list)
total_runs_completed: int = 0
total_runs_failed: int = 0
gpu_utilization: float = 0.0
memory_used_mb: float = 0.0
last_heartbeat: str = field(default_factory=lambda: datetime.utcnow().isoformat())

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"config": self.config.to_dict(),
"status": self.status.value,
"current_runs": self.current_runs,
"total_runs_completed": self.total_runs_completed,
"total_runs_failed": self.total_runs_failed,
"gpu_utilization": self.gpu_utilization,
"memory_used_mb": self.memory_used_mb,
"last_heartbeat": self.last_heartbeat,
"available_slots": self.config.max_concurrent_runs - len(self.current_runs),
}


@dataclass
class RunTask:
"""Eine Run-Aufgabe."""

task_id: str
run_id: str
run_config: Dict[str, Any]
priority: float = 0.5
created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
started_at: Optional[str] = None
completed_at: Optional[str] = None
status: RunStatus = RunStatus.PENDING
worker_id: Optional[str] = None
retry_count: int = 0
max_retries: int = 3
error_message: Optional[str] = None
result: Optional[Dict[str, Any]] = None

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"task_id": self.task_id,
"run_id": self.run_id,
"run_config": self.run_config,
"priority": self.priority,
"created_at": self.created_at,
"started_at": self.started_at,
"completed_at": self.completed_at,
"status": self.status.value,
"worker_id": self.worker_id,
"retry_count": self.retry_count,
"max_retries": self.max_retries,
"error_message": self.error_message,
"result": self.result,
}


@dataclass
class BatchStatus:
"""Status einer Batch."""

batch_id: str
total: int = 0
running: int = 0
completed: int = 0
failed: int = 0
pending: int = 0
created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
completed_at: Optional[str] = None

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"batch_id": self.batch_id,
"total": self.total,
"running": self.running,
"completed": self.completed,
"failed": self.failed,
"pending": self.pending,
"created_at": self.created_at,
"completed_at": self.completed_at,
}


class DistributedRunner:
"""
Verteilte Run-Ausführung.

Features:
- Multi-GPU Support
- Load Balancing
- Fault Tolerance (Retry bei Failure)
- Progress Tracking
- Result Aggregation

Example:
workers = [
WorkerConfig("worker_0", gpu_id=0),
WorkerConfig("worker_1", gpu_id=1),
]

runner = DistributedRunner(workers)

# Runs einreichen
batch_id = runner.submit_runs([
{"depth": 12, "width": 512},
{"depth": 14, "width": 640},
])

# Status prüfen
status = runner.get_batch_status(batch_id)
print(f"Completed: {status['completed']}/{status['total']}")

# Worker Load
load = runner.get_worker_load()
"""

def __init__(
self,
workers: List[WorkerConfig],
results_dir: Optional[str] = None,
max_retries: int = 3,
retry_delay_seconds: float = 5.0,
) -> None:
"""
Initialisiere Distributed Runner.

Args:
workers: Liste von Worker-Konfigurationen
results_dir: Verzeichnis für Ergebnisse
max_retries: Maximale Retry-Versuche pro Run
retry_delay_seconds: Verzögerung zwischen Retries
"""
self._workers: Dict[str, WorkerState] = {}
for config in workers:
self._workers[config.worker_id] = WorkerState(config=config)

self._run_queue: List[RunTask] = []
self._running_tasks: Dict[str, RunTask] = {}
self._completed_tasks: Dict[str, RunTask] = {}
self._failed_tasks: Dict[str, RunTask] = {}

self._batches: Dict[str, BatchStatus] = {}
self._batch_tasks: Dict[str, List[str]] = {} # batch_id -> task_ids

self._results_dir = Path(results_dir) if results_dir else Path(__file__).parent.parent / "results"
self._registry = RunRegistry(results_dir=str(self._results_dir))

self._max_retries = max_retries
self._retry_delay = retry_delay_seconds

self._lock = threading.Lock()
self._is_running = False
self._scheduler_thread: Optional[threading.Thread] = None

# Callbacks
self._on_run_start: Optional[Callable[[str], None]] = None
self._on_run_complete: Optional[Callable[[str, Dict], None]] = None
self._on_run_fail: Optional[Callable[[str, str], None]] = None

@property
def workers(self) -> Dict[str, WorkerState]:
"""Alle Worker zurückgeben."""
return dict(self._workers)

@property
def queue_length(self) -> int:
"""Aktuelle Queue-Länge."""
return len(self._run_queue)

def register_callbacks(
self,
on_start: Optional[Callable[[str], None]] = None,
on_complete: Optional[Callable[[str, Dict], None]] = None,
on_fail: Optional[Callable[[str, str], None]] = None,
) -> None:
"""
Registriere Callbacks für Run-Events.

Args:
on_start: Callback bei Run-Start
on_complete: Callback bei Run-Abschluss
on_fail: Callback bei Run-Fehler
"""
self._on_run_start = on_start
self._on_run_complete = on_complete
self._on_run_fail = on_fail

def submit_runs(self, run_configs: List[Dict[str, Any]], batch_id: Optional[str] = None) -> str:
"""
Runs zur Queue hinzufügen.

Args:
run_configs: Liste von Run-Konfigurationen
batch_id: Optionale Batch-ID (wird generiert wenn None)

Returns:
batch_id
"""
if batch_id is None:
batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

# Batch erstellen
batch = BatchStatus(batch_id=batch_id, total=len(run_configs))
self._batches[batch_id] = batch

task_ids = []
with self._lock:
for i, config in enumerate(run_configs):
task_id = f"{batch_id}_task_{i}"
run_id = config.get("run_id", f"{batch_id}_run_{i}")

# Priorität berechnen (kann angepasst werden)
priority = config.get("priority", 0.5)

task = RunTask(
task_id=task_id,
run_id=run_id,
run_config=config,
priority=priority,
max_retries=self._max_retries,
)

self._run_queue.append(task)
task_ids.append(task_id)

# In Registry registrieren
parent_run_id = config.get("parent_run_id")
self._registry.register(
run_id=run_id,
config_hash=config.get("config_hash", str(hash(json.dumps(config, sort_keys=True)))),
parent_run_id=parent_run_id,
seed=config.get("seed", 42),
)

self._batch_tasks[batch_id] = task_ids

# Queue nach Priorität sortieren
self._run_queue.sort(key=lambda t: t.priority, reverse=True)

return batch_id

def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
"""
Status einer Batch.

Args:
batch_id: Batch-ID

Returns:
Dictionary mit Status-Informationen
"""
batch = self._batches.get(batch_id)
if not batch:
return {"error": f"Batch nicht gefunden: {batch_id}"}

# Status aktualisieren
task_ids = self._batch_tasks.get(batch_id, [])

running = 0
completed = 0
failed = 0
pending = 0

for task_id in task_ids:
if task_id in self._running_tasks:
running += 1
elif task_id in self._completed_tasks:
completed += 1
elif task_id in self._failed_tasks:
failed += 1
else:
# In Queue oder noch nicht gestartet
pending += 1

batch.running = running
batch.completed = completed
batch.failed = failed
batch.pending = pending

if completed + failed == batch.total:
batch.completed_at = datetime.utcnow().isoformat()

return batch.to_dict()

def get_worker_load(self) -> Dict[str, Any]:
"""
Aktuelle Worker-Auslastung.

Returns:
Dictionary mit Worker-Load-Informationen
"""
load = {}
for worker_id, worker in self._workers.items():
load[worker_id] = {
"runs": len(worker.current_runs),
"max_runs": worker.config.max_concurrent_runs,
"available_slots": worker.config.max_concurrent_runs - len(worker.current_runs),
"gpu_util": worker.gpu_utilization,
"memory_mb": worker.memory_used_mb,
"status": worker.status.value,
"total_completed": worker.total_runs_completed,
"total_failed": worker.total_runs_failed,
}
return load

def get_queue_stats(self) -> Dict[str, Any]:
"""
Queue-Statistiken.

Returns:
Dictionary mit Queue-Informationen
"""
with self._lock:
return {
"queue_length": len(self._run_queue),
"running_tasks": len(self._running_tasks),
"completed_tasks": len(self._completed_tasks),
"failed_tasks": len(self._failed_tasks),
"total_workers": len(self._workers),
"active_workers": sum(1 for w in self._workers.values() if w.status == WorkerStatus.RUNNING),
"batches": {
batch_id: batch.to_dict()
for batch_id, batch in self._batches.items()
},
}

def _get_next_task(self, worker_id: str) -> Optional[RunTask]:
"""
Hole nächste Aufgabe für Worker.

Berücksichtigt:
- Priority
- Worker-Kapazität

Args:
worker_id: Worker-ID

Returns:
Nächste RunTask oder None
"""
worker = self._workers.get(worker_id)
if not worker:
return None

# Prüfe Kapazität
if len(worker.current_runs) >= worker.config.max_concurrent_runs:
return None

# Hole höchste Priorität aus Queue
with self._lock:
if not self._run_queue:
return None

# Nimm höchste Priorität
task = self._run_queue.pop(0)
return task

def _start_task(self, task: RunTask, worker_id: str) -> None:
"""Starte Task auf Worker."""
task.status = RunStatus.RUNNING
task.started_at = datetime.utcnow().isoformat()
task.worker_id = worker_id

worker = self._workers[worker_id]
worker.current_runs.append(task.task_id)
worker.status = WorkerStatus.RUNNING

with self._lock:
self._running_tasks[task.task_id] = task

# Callback
if self._on_run_start:
self._on_run_start(task.run_id)

# Run in Registry starten
self._registry.start_run(task.run_id)

# Simuliere Run-Ausführung (in echter Implementierung: subprocess)
threading.Thread(
target=self._execute_run,
args=(task, worker_id),
daemon=True,
).start()

def _execute_run(self, task: RunTask, worker_id: str) -> None:
"""
Führe Run aus (simuliert).

In echter Implementierung würde hier das Training gestartet werden.
"""
worker = self._workers[worker_id]

try:
# Simuliere Training (1-5 Sekunden)
import random
duration = random.uniform(1.0, 5.0)
time.sleep(duration)

# Simuliere Ergebnis (80% Erfolgsrate)
success = random.random() > 0.2

if success:
# Erfolg
result = {
"val_bpb": random.uniform(1.0, 2.0),
"ms_per_step": random.uniform(100, 200),
"steps_completed": 1000,
}
self._complete_task(task, worker_id, result)
else:
# Fehler
raise RuntimeError("Simulierter Trainingsfehler")

except Exception as e:
self._fail_task(task, worker_id, str(e))

def _complete_task(self, task: RunTask, worker_id: str, result: Dict[str, Any]) -> None:
"""Task erfolgreich abschließen."""
task.status = RunStatus.COMPLETED
task.completed_at = datetime.utcnow().isoformat()
task.result = result

worker = self._workers[worker_id]
if task.task_id in worker.current_runs:
worker.current_runs.remove(task.task_id)
worker.total_runs_completed += 1

with self._lock:
del self._running_tasks[task.task_id]
self._completed_tasks[task.task_id] = task

# Run in Registry abschließen
self._registry.complete_run(task.run_id, result)

# Callback
if self._on_run_complete:
self._on_run_complete(task.run_id, result)

def _fail_task(self, task: RunTask, worker_id: str, error: str) -> None:
"""Task fehlgeschlagen."""
task.error_message = error
worker = self._workers[worker_id]

if task.task_id in worker.current_runs:
worker.current_runs.remove(task.task_id)

# Retry Logic
if task.retry_count < task.max_retries:
task.retry_count += 1
task.status = RunStatus.RETRYING
task.worker_id = None

with self._lock:
del self._running_tasks[task.task_id]
self._run_queue.insert(0, task) # Zurück an Anfang der Queue

# Retry Delay
time.sleep(self._retry_delay)
else:
# Endgültig fehlgeschlagen
task.status = RunStatus.FAILED
task.completed_at = datetime.utcnow().isoformat()
worker.total_runs_failed += 1

with self._lock:
del self._running_tasks[task.task_id]
self._failed_tasks[task.task_id] = task

# Run in Registry als fehlgeschlagen markieren
self._registry.fail_run(task.run_id, error)

# Callback
if self._on_run_fail:
self._on_run_fail(task.run_id, error)

def _scheduler_loop(self) -> None:
"""Scheduler-Hauptschleife."""
while self._is_running:
# Für jeden Worker prüfen ob Aufgabe verfügbar
for worker_id, worker in self._workers.items():
if worker.status == WorkerStatus.OFFLINE:
continue

task = self._get_next_task(worker_id)
if task:
self._start_task(task, worker_id)

# Worker-Status aktualisieren (simuliert)
self._update_worker_status()

time.sleep(0.1) # 100ms Polling

def _update_worker_status(self) -> None:
"""Aktualisiere Worker-Status (simuliert)."""
for worker in self._workers.values():
if worker.current_runs:
# Simuliere GPU-Auslastung
import random
worker.gpu_utilization = random.uniform(60, 95)
worker.memory_used_mb = random.uniform(4000, 7000)
worker.status = WorkerStatus.RUNNING
else:
worker.gpu_utilization = 0.0
worker.memory_used_mb = 0.0
worker.status = WorkerStatus.IDLE

worker.last_heartbeat = datetime.utcnow().isoformat()

def start(self) -> None:
"""Starte Distributed Runner."""
if self._is_running:
print(" Runner läuft bereits")
return

self._is_running = True
self._scheduler_thread = threading.Thread(
target=self._scheduler_loop,
daemon=True,
)
self._scheduler_thread.start()

print(f" Distributed Runner gestartet mit {len(self._workers)} Workern")

def stop(self) -> None:
"""Stoppe Distributed Runner."""
self._is_running = False
if self._scheduler_thread:
self._scheduler_thread.join(timeout=5.0)
print(" Distributed Runner gestoppt")

def auto_scale(self, target_gpu_util: float = 0.8) -> Dict[str, Any]:
"""
Automatische Skalierung.

Passt Anzahl aktiver Worker an basierend auf:
- Queue-Length
- GPU-Auslastung
- Failure-Rate

Args:
target_gpu_util: Ziel-GPU-Auslastung (0-1)

Returns:
Scaling-Entscheidung
"""
stats = self.get_queue_stats()
load = self.get_worker_load()

# Durchschnittliche GPU-Auslastung
avg_gpu_util = sum(w["gpu_util"] for w in load.values()) / len(load) if load else 0

# Queue-Druck
queue_pressure = stats["queue_length"] / max(len(load), 1)

# Failure-Rate
total_completed = sum(w["total_completed"] for w in load.values())
total_failed = sum(w["total_failed"] for w in load.values())
failure_rate = total_failed / max(total_completed + total_failed, 1)

scaling_decision = {
"current_workers": len(self._workers),
"avg_gpu_util": avg_gpu_util,
"queue_pressure": queue_pressure,
"failure_rate": failure_rate,
"action": "none",
"reason": "",
}

# Scale Up wenn Queue lang und GPU-Auslastung hoch
if queue_pressure > 2 and avg_gpu_util > target_gpu_util:
scaling_decision["action"] = "scale_up"
scaling_decision["reason"] = "Hohe Queue-Länge bei hoher GPU-Auslastung"

# Scale Down wenn Queue leer und GPU-Auslastung niedrig
elif stats["queue_length"] == 0 and avg_gpu_util < 0.3:
scaling_decision["action"] = "scale_down"
scaling_decision["reason"] = "Leere Queue bei niedriger GPU-Auslastung"

# Pause bei hoher Failure-Rate
elif failure_rate > 0.5:
scaling_decision["action"] = "pause"
scaling_decision["reason"] = f"Hohe Failure-Rate: {failure_rate:.1%}"

return scaling_decision


def cmd_distributed_runner(args: argparse.Namespace) -> int:
"""Distributed Runner Command."""
print(" Distributed Runner")
print("=" * 60)

# Worker-Konfiguration erstellen
num_workers = args.workers
workers = [
WorkerConfig(
worker_id=f"worker_{i}",
gpu_id=i,
max_concurrent_runs=args.max_concurrent,
memory_limit_mb=args.memory_limit,
)
for i in range(num_workers)
]

runner = DistributedRunner(workers)

# Beispiel-Runs einreichen
run_configs = []
for i in range(args.num_runs):
config = {
"run_id": f"dist_run_{i:03d}",
"depth": 12 + (i % 4),
"width": 512 + (i % 4) * 64,
"seed": 42 + i,
}
run_configs.append(config)

print(f"\n Submitte {len(run_configs)} Runs...")
batch_id = runner.submit_runs(run_configs)
print(f" Batch-ID: {batch_id}")

# Starte Runner
runner.start()

# Warte und zeige Status
try:
for i in range(args.duration):
time.sleep(1)

status = runner.get_batch_status(batch_id)
load = runner.get_worker_load()

print(f"\r Status: {status['completed']}/{status['total']} completed, "
f"{status['running']} running, {status['failed']} failed", end="")

# Auto-Scale prüfen
if i % 5 == 0:
scaling = runner.auto_scale()
if scaling["action"] != "none":
print(f"\n Auto-Scale: {scaling['action']} - {scaling['reason']}")

except KeyboardInterrupt:
print("\n\n Unterbrochen")

finally:
runner.stop()

# Finaler Report
print("\n\n" + "=" * 60)
print(" Final Report")
print("=" * 60)

stats = runner.get_queue_stats()
print(f"\nQueue Stats:")
print(f" Completed: {stats['completed_tasks']}")
print(f" Failed: {stats['failed_tasks']}")

print(f"\nWorker Load:")
load = runner.get_worker_load()
for worker_id, worker_load in load.items():
print(f" {worker_id}: {worker_load['total_completed']} completed, "
f"{worker_load['total_failed']} failed")

return 0


def create_parser() -> argparse.ArgumentParser:
"""Erstelle Argument Parser."""
parser = argparse.ArgumentParser(
prog="distributed-runner",
description="Distributed Run Execution",
)
parser.add_argument(
"--workers",
type=int,
default=2,
help="Anzahl Worker (default: 2)",
)
parser.add_argument(
"--max-concurrent",
type=int,
default=1,
help="Max Runs pro Worker (default: 1)",
)
parser.add_argument(
"--memory-limit",
type=int,
default=8000,
help="Memory Limit MB pro Worker (default: 8000)",
)
parser.add_argument(
"--num-runs",
type=int,
default=10,
help="Anzahl Runs zum Testen (default: 10)",
)
parser.add_argument(
"--duration",
type=int,
default=30,
help="Test-Dauer in Sekunden (default: 30)",
)
parser.set_defaults(func=cmd_distributed_runner)
return parser


def main() -> int:
"""Hauptfunktion."""
parser = create_parser()
args = parser.parse_args()
return args.func(args)


if __name__ == "__main__":
sys.exit(main())
