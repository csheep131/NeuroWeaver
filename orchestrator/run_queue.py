#!/usr/bin/env python3
"""
Run Queue Manager für NeuroWeave Phase 5.

Priority-basierte Queue für Run-Scheduling.

Features:
- Priority Scoring (basierend auf Hypothesis Confidence)
- Preemption (wichtige Runs können vorziehen)
- Fair Scheduling (verhindert Starvation)
- Deadline Awareness
"""

from __future__ import annotations

import argparse
import heapq
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class QueuePriority(Enum):
    """Prioritätsstufen."""

    CRITICAL = 1.0
    HIGH = 0.8
    NORMAL = 0.5
    LOW = 0.3
    BACKGROUND = 0.1


@dataclass
class QueuedRun:
    """Ein Run in der Queue."""

    run_id: str
    run_config: Dict[str, Any]
    priority: float = 0.5
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    enqueue_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    wait_time_seconds: float = 0.0
    starvation_score: float = 0.0  # 0-1, höher = mehr Starvation
    hypothesis_confidence: float = 0.5
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Berechne initiale Werte."""
        if self.deadline and isinstance(self.deadline, str):
            self.deadline = datetime.fromisoformat(self.deadline)

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary."""
        return {
            "run_id": self.run_id,
            "run_config": self.run_config,
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
            "enqueue_time": self.enqueue_time,
            "started_at": self.started_at,
            "wait_time_seconds": self.wait_time_seconds,
            "starvation_score": self.starvation_score,
            "hypothesis_confidence": self.hypothesis_confidence,
            "tags": self.tags,
        }

    def effective_priority(self) -> float:
        """
        Berechne effektive Priorität.

        Berücksichtigt:
        - Basis-Priorität
        - Deadline-Nähe
        - Starvation-Score

        Returns:
            Effektive Priorität (höher = wichtiger)
        """
        base_priority = self.priority

        # Deadline-Boost
        deadline_boost = 0.0
        if self.deadline:
            time_to_deadline = (self.deadline - datetime.utcnow()).total_seconds()
            if time_to_deadline < 0:
                # Deadline überschritten
                deadline_boost = 0.5
            elif time_to_deadline < 3600:  # < 1 Stunde
                deadline_boost = 0.3
            elif time_to_deadline < 7200:  # < 2 Stunden
                deadline_boost = 0.1

        # Starvation-Boost
        starvation_boost = self.starvation_score * 0.3

        return min(1.0, base_priority + deadline_boost + starvation_boost)

    def __lt__(self, other: "QueuedRun") -> bool:
        """Vergleich für Heap (höhere Priorität = kleiner im Heap)."""
        return self.effective_priority() > other.effective_priority()


@dataclass
class WorkerAssignment:
    """Zuweisung eines Runs an einen Worker."""

    run_id: str
    worker_id: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    expected_duration_seconds: Optional[float] = None


class RunQueueManager:
    """
    Priority-basierte Run-Queue.

    Features:
    - Priority Scoring (basierend auf Hypothesis Confidence)
    - Preemption (wichtige Runs können vorziehen)
    - Fair Scheduling (verhindert Starvation)
    - Deadline Awareness

    Example:
        manager = RunQueueManager()

        # Runs einreihen
        manager.enqueue(
            run_config={"depth": 12, "width": 512},
            priority=0.8,
            deadline=datetime.utcnow() + timedelta(hours=2),
        )

        # Nächsten Run für Worker holen
        next_run = manager.dequeue("worker_0")

        # Queue-Statistiken
        stats = manager.get_queue_stats()
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        starvation_threshold_seconds: float = 3600.0,
        preemption_enabled: bool = True,
    ) -> None:
        """
        Initialisiere Run Queue Manager.

        Args:
            max_queue_size: Maximale Queue-Größe
            starvation_threshold_seconds: Zeit bis Starvation einsetzt
            preemption_enabled: Preemption aktivieren
        """
        self._queue: List[QueuedRun] = []  # Priority Heap
        self._run_map: Dict[str, QueuedRun] = {}  # run_id -> QueuedRun
        self._running_runs: Dict[str, WorkerAssignment] = {}
        self._completed_runs: List[str] = []
        self._max_queue_size = max_queue_size
        self._starvation_threshold = starvation_threshold_seconds
        self._preemption_enabled = preemption_enabled

        self._lock = threading.Lock()
        self._last_update = datetime.utcnow()

        # Fair Scheduling Tracking
        self._worker_run_counts: Dict[str, int] = {}
        self._feature_run_counts: Dict[str, int] = {}

    def enqueue(
        self,
        run_config: Dict[str, Any],
        priority: float = 0.5,
        deadline: Optional[datetime] = None,
        run_id: Optional[str] = None,
        hypothesis_confidence: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Run zur Queue hinzufügen.

        Args:
            run_config: Run-Konfiguration
            priority: 0-1 (höher = wichtiger)
            deadline: Optionale Deadline
            run_id: Run-ID (wird generiert wenn None)
            hypothesis_confidence: Confidence der Hypothesis
            tags: Optionale Tags für Fair Scheduling

        Returns:
            run_id
        """
        with self._lock:
            # Queue-Größe prüfen
            if len(self._queue) >= self._max_queue_size:
                raise ValueError(f"Queue voll (max {self._max_queue_size})")

            # Run-ID generieren
            if run_id is None:
                run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

            # QueuedRun erstellen
            queued_run = QueuedRun(
                run_id=run_id,
                run_config=run_config,
                priority=priority,
                deadline=deadline,
                hypothesis_confidence=hypothesis_confidence,
                tags=tags or [],
            )

            # Zum Heap hinzufügen
            heapq.heappush(self._queue, queued_run)
            self._run_map[run_id] = queued_run

            return run_id

    def dequeue(
        self,
        worker_id: str,
        max_wait_time_seconds: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Nächsten Run für Worker holen.

        Berücksichtigt:
        - Priority
        - Deadline
        - Worker-Kapazität
        - Fairness

        Args:
            worker_id: Worker-ID
            max_wait_time_seconds: Maximale Wartezeit

        Returns:
            Run-Konfiguration oder None
        """
        start_time = time.time()

        while True:
            with self._lock:
                # Update Starvation Scores
                self._update_starvation_scores()

                # Sortiere Queue nach effektiver Priorität
                heapq.heapify(self._queue)

                # Hole höchsten Priority Run
                while self._queue:
                    queued_run = heapq.heappop(self._queue)

                    # Prüfe Fairness (verhindere Feature-Dominanz)
                    if not self._check_fairness(queued_run):
                        # Zurück zur Queue
                        heapq.heappush(self._queue, queued_run)
                        continue

                    # Run als gestartet markieren
                    queued_run.started_at = datetime.utcnow().isoformat()
                    queued_run.wait_time_seconds = (
                        datetime.utcnow() - queued_run.created_at
                    ).total_seconds()

                    # Worker-Zuweisung
                    assignment = WorkerAssignment(
                        run_id=queued_run.run_id,
                        worker_id=worker_id,
                    )
                    self._running_runs[queued_run.run_id] = assignment

                    # Worker-Run-Count aktualisieren
                    self._worker_run_counts[worker_id] = (
                        self._worker_run_counts.get(worker_id, 0) + 1
                    )

                    # Feature-Run-Count aktualisieren
                    for tag in queued_run.tags:
                        self._feature_run_counts[tag] = (
                            self._feature_run_counts.get(tag, 0) + 1
                        )

                    return queued_run.run_config

            # Warte wenn Queue leer
            if max_wait_time_seconds:
                elapsed = time.time() - start_time
                if elapsed >= max_wait_time_seconds:
                    return None

            time.sleep(0.1)

    def _check_fairness(self, queued_run: QueuedRun) -> bool:
        """
        Prüfe Fairness-Kriterien.

        Args:
            queued_run: Zu prüfender Run

        Returns:
            True wenn Run fair ist
        """
        # Verhindere dass ein Feature >50% der Runs bekommt
        for tag in queued_run.tags:
            feature_count = self._feature_run_counts.get(tag, 0)
            total_runs = sum(self._feature_run_counts.values())

            if total_runs > 10 and feature_count / total_runs > 0.5:
                return False

        return True

    def _update_starvation_scores(self) -> None:
        """Aktualisiere Starvation Scores für alle Runs."""
        now = datetime.utcnow()

        for queued_run in self._queue:
            wait_time = (now - queued_run.created_at).total_seconds()

            # Starvation Score steigt mit Wartezeit
            if wait_time > self._starvation_threshold:
                queued_run.starvation_score = min(
                    1.0,
                    (wait_time - self._starvation_threshold) / self._starvation_threshold,
                )
            else:
                queued_run.starvation_score = 0.0

    def preempt(self, run_id: str, high_priority_run_id: str) -> bool:
        """
        Preemption: Unterbreche Run für wichtigen Run.

        Args:
            run_id: Zu unterbrechender Run
            high_priority_run_id: Wichtiger Run der vorzieht

        Returns:
            True wenn Preemption erfolgreich
        """
        if not self._preemption_enabled:
            return False

        with self._lock:
            # Prüfe ob high_priority_run in Queue ist
            high_priority_run = self._run_map.get(high_priority_run_id)
            if not high_priority_run:
                return False

            # Prüfe ob zu unterbrechender Run läuft
            if run_id not in self._running_runs:
                return False

            # Nur preempten wenn high_priority wichtig genug
            if high_priority_run.effective_priority() < 0.9:
                return False

            # Laufenden Run zurück zur Queue
            running_assignment = self._running_runs[run_id]
            running_queued_run = self._run_map[run_id]
            running_queued_run.starvation_score = 0.5  # Boost für unterbrochenen Run

            heapq.heappush(self._queue, running_queued_run)
            del self._running_runs[run_id]

            return True

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Queue-Statistiken.

        Returns:
            Dictionary mit Queue-Informationen
        """
        with self._lock:
            # Priority-Verteilung
            priority_distribution = {"critical": 0, "high": 0, "normal": 0, "low": 0}
            for run in self._queue:
                if run.priority >= 0.8:
                    priority_distribution["critical"] += 1
                elif run.priority >= 0.6:
                    priority_distribution["high"] += 1
                elif run.priority >= 0.3:
                    priority_distribution["normal"] += 1
                else:
                    priority_distribution["low"] += 1

            # Durchschnittliche Wartezeit
            if self._queue:
                avg_wait = sum(
                    (datetime.utcnow() - r.created_at).total_seconds()
                    for r in self._queue
                ) / len(self._queue)
            else:
                avg_wait = 0.0

            # Starvation Risk
            starvation_risk = [
                r.run_id for r in self._queue if r.starvation_score > 0.5
            ]

            return {
                "queue_length": len(self._queue),
                "running_count": len(self._running_runs),
                "completed_count": len(self._completed_runs),
                "avg_wait_time": f"{avg_wait / 60:.1f}m",  # In Minuten
                "priority_distribution": priority_distribution,
                "starvation_risk": starvation_risk,
                "feature_distribution": dict(self._feature_run_counts),
                "worker_distribution": dict(self._worker_run_counts),
            }

    def remove(self, run_id: str) -> bool:
        """
        Run aus Queue entfernen.

        Args:
            run_id: Zu entfernender Run

        Returns:
            True wenn entfernt
        """
        with self._lock:
            if run_id not in self._run_map:
                return False

            queued_run = self._run_map.pop(run_id)

            # Aus Heap entfernen (ineffizient, aber funktional)
            self._queue = [r for r in self._queue if r.run_id != run_id]
            heapq.heapify(self._queue)

            return True

    def get_position(self, run_id: str) -> Optional[int]:
        """
        Hole Position in Queue.

        Args:
            run_id: Run-ID

        Returns:
            Position (1-basiert) oder None
        """
        with self._lock:
            if run_id not in self._run_map:
                return None

            # Sortiere Queue temporär
            sorted_queue = sorted(
                self._queue,
                key=lambda r: r.effective_priority(),
                reverse=True,
            )

            for i, run in enumerate(sorted_queue):
                if run.run_id == run_id:
                    return i + 1

            return None

    def reprioritize(self, run_id: str, new_priority: float) -> bool:
        """
        Priorität eines Runs ändern.

        Args:
            run_id: Run-ID
            new_priority: Neue Priorität

        Returns:
            True wenn aktualisiert
        """
        with self._lock:
            if run_id not in self._run_map:
                return False

            queued_run = self._run_map[run_id]
            queued_run.priority = new_priority

            # Heap neu organisieren
            heapq.heapify(self._queue)

            return True

    def complete_run(self, run_id: str) -> None:
        """
        Run als abgeschlossen markieren.

        Args:
            run_id: Abgeschlossener Run
        """
        with self._lock:
            if run_id in self._running_runs:
                del self._running_runs[run_id]

            if run_id in self._run_map:
                del self._run_map[run_id]

            self._completed_runs.append(run_id)


def cmd_queue_manager(args: argparse.Namespace) -> int:
    """Queue Manager Command."""
    print("📋 Run Queue Manager")
    print("=" * 60)

    manager = RunQueueManager()

    # Beispiel-Runs einreihen
    print("\n📥 Reihed {args.num_runs} Runs ein...")

    from datetime import timedelta

    priorities = [0.9, 0.7, 0.5, 0.3, 0.1]
    tags_list = [["gqa"], ["film"], ["swiglu"], ["leaky_relu"], ["rope"]]

    for i in range(args.num_runs):
        priority = priorities[i % len(priorities)]
        tags = tags_list[i % len(tags_list)]

        deadline = None
        if priority > 0.7:
            deadline = datetime.utcnow() + timedelta(minutes=30)

        run_id = manager.enqueue(
            run_config={"depth": 12 + i, "width": 512},
            priority=priority,
            deadline=deadline,
            hypothesis_confidence=0.5 + (priority * 0.3),
            tags=tags,
        )
        print(f"   Eingereiht: {run_id} (Priority: {priority})")

    # Queue-Statistiken
    print("\n📊 Queue-Statistiken:")
    stats = manager.get_queue_stats()
    print(f"   Queue-Länge: {stats['queue_length']}")
    print(f"   Ø Wartezeit: {stats['avg_wait_time']}")
    print(f"   Priority-Verteilung: {stats['priority_distribution']}")

    if stats["starvation_risk"]:
        print(f"   ⚠️  Starvation Risk: {stats['starvation_risk']}")

    # Test Dequeue
    print("\n📤 Dequeue Tests:")
    for i in range(3):
        worker_id = f"worker_{i}"
        run_config = manager.dequeue(worker_id)
        if run_config:
            print(f"   {worker_id}: {run_config.get('depth')}")

    # Finale Stats
    print("\n📊 Nach Dequeue:")
    stats = manager.get_queue_stats()
    print(f"   Queue-Länge: {stats['queue_length']}")
    print(f"   Running: {stats['running_count']}")

    print("\n" + "=" * 60)
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Erstelle Argument Parser."""
    parser = argparse.ArgumentParser(
        prog="queue-manager",
        description="Run Queue Manager",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Anzahl Test-Runs (default: 10)",
    )
    parser.set_defaults(func=cmd_queue_manager)
    return parser


def main() -> int:
    """Hauptfunktion."""
    parser = create_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
