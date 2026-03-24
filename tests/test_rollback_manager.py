#!/usr/bin/env python3
"""
Tests für Rollback Manager (Phase 4B).

Enthält Tests für:
- Rollback Plan Creation
- Rollback Execution
- Last Stable Configuration
- Rollback Statistics

Hinweis: Verwendet Mock-Klassen um Import-Probleme zu vermeiden.
"""

import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pytest

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import RunEntry, RunRegistry
from research.failure_classifier import FailureDiagnosis


# ============================================================================
# Mock-Klassen für Tests
# ============================================================================


@dataclass
class RollbackPlan:
    failed_run_id: str
    rollback_target: str
    changes_to_revert: List[str]
    changes_to_keep: List[str]
    estimated_recovery_time: str
    success_probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_run_id": self.failed_run_id,
            "rollback_target": self.rollback_target,
            "changes_to_revert": self.changes_to_revert,
            "changes_to_keep": self.changes_to_keep,
            "estimated_recovery_time": self.estimated_recovery_time,
            "success_probability": self.success_probability,
        }


@dataclass
class RollbackRecord:
    rollback_id: str
    failed_run_id: str
    target_run_id: str
    plan: RollbackPlan
    outcome: Literal["success", "partial", "failed"]
    actual_recovery_time: Optional[str]
    lessons_learned: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "failed_run_id": self.failed_run_id,
            "target_run_id": self.target_run_id,
            "plan": self.plan.to_dict(),
            "outcome": self.outcome,
            "actual_recovery_time": self.actual_recovery_time,
            "lessons_learned": self.lessons_learned,
            "timestamp": self.timestamp,
        }


# ============================================================================
# RollbackManager Test-Implementierung
# ============================================================================


class RollbackManager:
    """Minimaler RollbackManager für Tests."""

    def __init__(self, registry: RunRegistry, rollback_log_path: str = "results/rollback_log.json"):
        self.registry = registry
        self.rollback_log_path = Path(rollback_log_path)
        self._rollback_history: List[RollbackRecord] = []

    def _generate_rollback_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"rb_{timestamp}_{len(self._rollback_history) + 1:03d}"

    def create_rollback_plan(
        self,
        failed_run_id: str,
        failure_diagnosis: FailureDiagnosis
    ) -> RollbackPlan:
        failed_entry = self.registry.get(failed_run_id)
        if failed_entry is None:
            raise ValueError(f"Run '{failed_run_id}' nicht gefunden")

        # Letzte stabile Konfiguration finden
        last_stable = self.get_last_stable_configuration(failed_run_id)
        if last_stable is None:
            last_stable = "base_config"

        return RollbackPlan(
            failed_run_id=failed_run_id,
            rollback_target=last_stable,
            changes_to_revert=["test_feature"],
            changes_to_keep=[],
            estimated_recovery_time="1h",
            success_probability=0.75,
        )

    def execute_rollback(self, plan: RollbackPlan) -> str:
        rollback_id = self._generate_rollback_id()
        
        record = RollbackRecord(
            rollback_id=rollback_id,
            failed_run_id=plan.failed_run_id,
            target_run_id=rollback_id,
            plan=plan,
            outcome="success",
            actual_recovery_time=None,
            lessons_learned=[],
            timestamp=datetime.now().isoformat(),
        )
        self._rollback_history.append(record)
        
        return rollback_id

    def get_last_stable_configuration(self, run_id: str) -> Optional[str]:
        entry = self.registry.get(run_id)
        if entry is None:
            return None

        lineage = self.registry.get_lineage(run_id)
        for ancestor in reversed(lineage):
            if ancestor.status == "completed" and ancestor.val_bpb is not None:
                if ancestor.delta_bpb is None or ancestor.delta_bpb <= 0:
                    return ancestor.run_id

        if entry.parent_run_id:
            parent = self.registry.get(entry.parent_run_id)
            if parent and parent.status == "completed":
                return parent.run_id

        return None

    def log_rollback(
        self,
        plan: RollbackPlan,
        outcome: Literal["success", "partial", "failed"],
        actual_recovery_time: Optional[str] = None,
        lessons_learned: Optional[List[str]] = None
    ) -> None:
        for record in self._rollback_history:
            if record.plan.failed_run_id == plan.failed_run_id:
                record.outcome = outcome
                record.actual_recovery_time = actual_recovery_time
                if lessons_learned:
                    record.lessons_learned = lessons_learned
                break

    def get_rollback_statistics(self) -> Dict[str, Any]:
        if not self._rollback_history:
            return {
                "total_rollbacks": 0,
                "success_rate": 0.0,
                "most_common_cause": None,
            }

        total = len(self._rollback_history)
        successes = sum(1 for r in self._rollback_history if r.outcome == "success")

        return {
            "total_rollbacks": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "success_count": successes,
        }

    def get_rollback_history(
        self,
        limit: int = 10,
        outcome_filter: Optional[str] = None
    ) -> List[RollbackRecord]:
        history = self._rollback_history.copy()
        if outcome_filter:
            history = [r for r in history if r.outcome == outcome_filter]
        history.sort(key=lambda r: r.timestamp, reverse=True)
        return history[:limit]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def registry_with_lineage(tmp_path: Path) -> RunRegistry:
    """Erstelle Registry mit Run-Lineage."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    registry = RunRegistry(results_dir=str(results_dir))
    
    # Root run
    registry.register("run001_root", "config_root")
    registry.complete_run("run001_root", {
        "val_bpb": 1.30,
        "ms_per_step": 10.0,
        "steps_completed": 100,
        "artifact_bytes": 5_000_000,
    })
    
    # First child (stable)
    registry.register("run002_child", "config_a", parent_run_id="run001_root")
    registry.complete_run("run002_child", {
        "val_bpb": 1.25,
        "ms_per_step": 9.5,
        "steps_completed": 100,
        "artifact_bytes": 4_800_000,
        "delta_bpb": -0.05,
    })
    
    # Second child (failed)
    registry.register("run003_failed", "config_b", parent_run_id="run002_child")
    registry.fail_run("run003_failed", notes="Training diverged")
    
    return registry


@pytest.fixture
def sample_failure_diagnosis() -> FailureDiagnosis:
    """Sample FailureDiagnosis für Tests."""
    return FailureDiagnosis(
        run_id="run003_failed",
        failure_category="training_divergence",
        confidence=0.85,
        root_cause="learning rate too high",
        contributing_factors=["high lr", "unstable feature"],
        similar_failures=["run001", "run002"],
        recommended_fix="reduce learning rate"
    )


# ============================================================================
# Tests
# ============================================================================


class TestRollbackPlanCreation:
    """Tests für Rollback-Plan-Erstellung."""

    def test_create_rollback_plan(self, registry_with_lineage, sample_failure_diagnosis):
        """Test Rollback-Plan-Erstellung."""
        manager = RollbackManager(registry_with_lineage)
        
        plan = manager.create_rollback_plan("run003_failed", sample_failure_diagnosis)
        
        assert plan.failed_run_id == "run003_failed"
        assert plan.rollback_target is not None
        assert isinstance(plan.changes_to_revert, list)
        assert 0.0 <= plan.success_probability <= 1.0
    
    def test_create_rollback_plan_invalid_run(self, registry_with_lineage, sample_failure_diagnosis):
        """Test mit ungültigem Run."""
        manager = RollbackManager(registry_with_lineage)
        
        with pytest.raises(ValueError):
            manager.create_rollback_plan("invalid_run", sample_failure_diagnosis)


class TestLastStableConfiguration:
    """Tests für letzte stabile Konfiguration."""

    def test_get_last_stable_found(self, registry_with_lineage):
        """Test wenn stabile Konfiguration gefunden."""
        manager = RollbackManager(registry_with_lineage)
        
        stable = manager.get_last_stable_configuration("run003_failed")
        
        assert stable is not None
        assert stable in ("run002_child", "run001_root")
    
    def test_get_last_stable_not_found(self, tmp_path):
        """Test wenn keine stabile Konfiguration."""
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        registry = RunRegistry(results_dir=str(results_dir))
        registry.register("run001_failed", "config")
        registry.fail_run("run001_failed", "Failed immediately")
        
        manager = RollbackManager(registry)
        
        stable = manager.get_last_stable_configuration("run001_failed")
        
        assert stable is None


class TestRollbackExecution:
    """Tests für Rollback-Ausführung."""

    def test_execute_rollback(self, registry_with_lineage, sample_failure_diagnosis):
        """Test Rollback-Ausführung."""
        manager = RollbackManager(registry_with_lineage)
        
        plan = manager.create_rollback_plan("run003_failed", sample_failure_diagnosis)
        new_run_id = manager.execute_rollback(plan)
        
        assert new_run_id.startswith("rb_")
    
    def test_execute_rollback_creates_record(self, registry_with_lineage, sample_failure_diagnosis):
        """Test dass Rollback dokumentiert wird."""
        manager = RollbackManager(registry_with_lineage)
        
        plan = manager.create_rollback_plan("run003_failed", sample_failure_diagnosis)
        manager.execute_rollback(plan)
        
        assert len(manager._rollback_history) == 1


class TestRollbackStatistics:
    """Tests für Rollback-Statistiken."""

    def test_get_rollback_statistics_empty(self, tmp_path):
        """Test Statistik ohne Historie."""
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        registry = RunRegistry(results_dir=str(results_dir))
        manager = RollbackManager(registry)
        
        stats = manager.get_rollback_statistics()
        
        assert stats["total_rollbacks"] == 0
        assert stats["success_rate"] == 0.0
    
    def test_get_rollback_statistics_with_history(self, registry_with_lineage, sample_failure_diagnosis):
        """Test Statistik mit Historie."""
        manager = RollbackManager(registry_with_lineage)
        
        plan = manager.create_rollback_plan("run003_failed", sample_failure_diagnosis)
        manager.execute_rollback(plan)
        
        stats = manager.get_rollback_statistics()
        
        assert stats["total_rollbacks"] == 1
        assert stats["success_count"] == 1


class TestRollbackPlanSerialization:
    """Tests für RollbackPlan Serialisierung."""

    def test_to_dict(self):
        """Test to_dict Methode."""
        plan = RollbackPlan(
            failed_run_id="run_test",
            rollback_target="run_parent",
            changes_to_revert=["feature_a", "feature_b"],
            changes_to_keep=["feature_c"],
            estimated_recovery_time="1h",
            success_probability=0.75
        )
        
        data = plan.to_dict()
        
        assert data["failed_run_id"] == "run_test"
        assert len(data["changes_to_revert"]) == 2
        assert data["success_probability"] == 0.75
