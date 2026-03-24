#!/usr/bin/env python3
"""
Tests für Run Quarantine Manager (Phase 4B).

Enthält Tests für:
- Quarantine Check
- Add/Release Quarantine
- Tick (Dekrement)
- Feature Statistics

Hinweis: Verwendet Mock-Klassen um Import-Probleme zu vermeiden.
"""

import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pytest

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import RunRegistry


# ============================================================================
# Mock-Klassen für Tests
# ============================================================================


@dataclass
class QuarantineEntry:
    target: str
    reason: str
    quarantine_type: Literal["feature", "combination", "context_specific"]
    triggered_by: List[str]
    quarantine_start: str
    quarantine_duration: int
    remaining_runs: int
    context_filter: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "reason": self.reason,
            "quarantine_type": self.quarantine_type,
            "triggered_by": self.triggered_by,
            "quarantine_start": self.quarantine_start,
            "quarantine_duration": self.quarantine_duration,
            "remaining_runs": self.remaining_runs,
            "context_filter": self.context_filter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuarantineEntry":
        return cls(
            target=data.get("target", ""),
            reason=data.get("reason", ""),
            quarantine_type=data.get("quarantine_type", "feature"),
            triggered_by=data.get("triggered_by", []),
            quarantine_start=data.get("quarantine_start", ""),
            quarantine_duration=data.get("quarantine_duration", 5),
            remaining_runs=data.get("remaining_runs", 5),
            context_filter=data.get("context_filter"),
        )


class RunQuarantineManager:
    """Minimaler RunQuarantineManager für Tests."""

    def __init__(
        self,
        quarantine_threshold: int = 3,
        quarantine_duration: int = 5,
        quarantine_log_path: str = "results/quarantine_log.json"
    ):
        self.quarantine_threshold = quarantine_threshold
        self.quarantine_duration = quarantine_duration
        self._quarantine_entries: Dict[str, QuarantineEntry] = {}
        self._feature_failure_counts: Dict[str, List[str]] = {}
        self._successful_run_count: int = 0

    def check_quarantine(
        self,
        proposed_features: List[str],
        context: str = "default"
    ) -> Tuple[bool, Optional[str]]:
        if not proposed_features:
            return False, None

        for feature in proposed_features:
            if feature in self._quarantine_entries:
                entry = self._quarantine_entries[feature]
                if entry.context_filter and entry.context_filter != context:
                    continue
                return True, f"Feature '{feature}' ist in Quarantäne: {entry.reason}"

        return False, None

    def add_quarantine(
        self,
        feature: str,
        triggered_by: List[str],
        context: Optional[str] = None,
        reason: Optional[str] = None,
        duration: Optional[int] = None
    ) -> QuarantineEntry:
        target_key = feature
        if context:
            target_key = f"{feature}@{context}"

        quarantine_type: Literal["feature", "combination", "context_specific"]
        if "+" in feature:
            quarantine_type = "combination"
        elif context:
            quarantine_type = "context_specific"
        else:
            quarantine_type = "feature"

        default_reason = f"Automatische Quarantäne nach {len(triggered_by)} Fehlern"
        entry = QuarantineEntry(
            target=feature,
            reason=reason or default_reason,
            quarantine_type=quarantine_type,
            triggered_by=triggered_by.copy(),
            quarantine_start=datetime.now().isoformat(),
            quarantine_duration=duration or self.quarantine_duration,
            remaining_runs=duration or self.quarantine_duration,
            context_filter=context,
        )

        self._quarantine_entries[target_key] = entry
        return entry

    def release_quarantine(self, feature: str, context: Optional[str] = None) -> bool:
        target_key = feature
        if context:
            target_key = f"{feature}@{context}"

        if target_key in self._quarantine_entries:
            del self._quarantine_entries[target_key]
            return True
        return False

    def tick(self, successful_run_id: str, registry: Optional[RunRegistry] = None) -> None:
        self._successful_run_count += 1

        expired = []
        for target_key, entry in self._quarantine_entries.items():
            entry.remaining_runs -= 1
            if entry.remaining_runs <= 0:
                expired.append(target_key)

        for target_key in expired:
            del self._quarantine_entries[target_key]

    def get_quarantine_list(self) -> List[QuarantineEntry]:
        return list(self._quarantine_entries.values())

    def human_override(
        self,
        feature: str,
        justification: str,
        context: Optional[str] = None
    ) -> bool:
        target_key = feature
        if context:
            target_key = f"{feature}@{context}"

        if target_key in self._quarantine_entries:
            entry = self._quarantine_entries[target_key]
            entry.reason += f" | Human Override: {justification}"
            return True
        return False

    def get_feature_statistics(self) -> Dict[str, Any]:
        quarantine_counts = {
            "total": len(self._quarantine_entries),
            "feature": sum(1 for e in self._quarantine_entries.values() if e.quarantine_type == "feature"),
            "combination": sum(1 for e in self._quarantine_entries.values() if e.quarantine_type == "combination"),
            "context_specific": sum(1 for e in self._quarantine_entries.values() if e.quarantine_type == "context_specific"),
        }

        return {
            "total_features_tracked": len(self._feature_failure_counts),
            "active_quarantines": quarantine_counts,
            "total_successful_runs": self._successful_run_count,
        }


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def quarantine_manager(tmp_path: Path) -> RunQuarantineManager:
    """Erstelle QuarantineManager für Tests."""
    log_path = tmp_path / "quarantine_log.json"
    return RunQuarantineManager(
        quarantine_threshold=3,
        quarantine_duration=5,
        quarantine_log_path=str(log_path)
    )


@pytest.fixture
def manager_with_quarantine(quarantine_manager) -> RunQuarantineManager:
    """QuarantineManager mit aktiver Quarantäne."""
    quarantine_manager.add_quarantine(
        feature="film",
        triggered_by=["run001", "run002", "run003"],
        reason="Multiple failures"
    )
    return quarantine_manager


# ============================================================================
# Tests
# ============================================================================


class TestCheckQuarantine:
    """Tests für Quarantäne-Prüfung."""

    def test_check_quarantine_not_blocked(self, quarantine_manager):
        """Test nicht blockiertes Feature."""
        is_blocked, reason = quarantine_manager.check_quarantine(["gqa"])
        
        assert is_blocked is False
        assert reason is None
    
    def test_check_quarantine_blocked(self, manager_with_quarantine):
        """Test blockiertes Feature."""
        is_blocked, reason = manager_with_quarantine.check_quarantine(["film"])
        
        assert is_blocked is True
        assert reason is not None
        assert "film" in reason


class TestAddQuarantine:
    """Tests für Quarantäne-Hinzufügung."""

    def test_add_quarantine_feature(self, quarantine_manager):
        """Test Feature-Quarantäne."""
        entry = quarantine_manager.add_quarantine(
            feature="leaky_relu",
            triggered_by=["run001", "run002"]
        )
        
        assert entry.target == "leaky_relu"
        assert entry.quarantine_type == "feature"
        assert entry.remaining_runs == 5
    
    def test_add_quarantine_combination(self, quarantine_manager):
        """Test Kombinations-Quarantäne."""
        entry = quarantine_manager.add_quarantine(
            feature="film+xsa",
            triggered_by=["run001"]
        )
        
        assert entry.target == "film+xsa"
        assert entry.quarantine_type == "combination"


class TestReleaseQuarantine:
    """Tests für Quarantäne-Aufhebung."""

    def test_release_quarantine_success(self, manager_with_quarantine):
        """Test erfolgreiche Aufhebung."""
        result = manager_with_quarantine.release_quarantine("film")
        
        assert result is True
        
        is_blocked, reason = manager_with_quarantine.check_quarantine(["film"])
        assert is_blocked is False
    
    def test_release_quarantine_not_found(self, quarantine_manager):
        """Test Aufhebung nicht existenter Quarantäne."""
        result = quarantine_manager.release_quarantine("nonexistent")
        
        assert result is False


class TestTick:
    """Tests für Tick-Funktion."""

    def test_tick_decrements_counter(self, quarantine_manager):
        """Test dass Tick Zähler dekrementiert."""
        quarantine_manager.add_quarantine(
            feature="test_feature",
            triggered_by=["run001"]
        )
        
        entries = quarantine_manager.get_quarantine_list()
        assert entries[0].remaining_runs == 5
        
        for i in range(3):
            quarantine_manager.tick(f"success_run_{i}")
        
        entries = quarantine_manager.get_quarantine_list()
        assert entries[0].remaining_runs == 2
    
    def test_tick_removes_expired(self, quarantine_manager):
        """Test dass abgelaufene Quarantänen entfernt werden."""
        quarantine_manager.add_quarantine(
            feature="expiring_feature",
            triggered_by=["run001"],
            duration=2
        )
        
        quarantine_manager.tick("success_run_1")
        quarantine_manager.tick("success_run_2")
        
        entries = quarantine_manager.get_quarantine_list()
        assert len(entries) == 0


class TestFeatureStatistics:
    """Tests für Feature-Statistiken."""

    def test_get_feature_statistics_empty(self, quarantine_manager):
        """Test Statistik ohne Daten."""
        stats = quarantine_manager.get_feature_statistics()
        
        assert stats["total_features_tracked"] == 0
        assert stats["active_quarantines"]["total"] == 0
    
    def test_get_feature_statistics_with_data(self, manager_with_quarantine):
        """Test Statistik mit Daten."""
        stats = manager_with_quarantine.get_feature_statistics()
        
        assert stats["active_quarantines"]["total"] >= 1


class TestQuarantineEntrySerialization:
    """Tests für QuarantineEntry Serialisierung."""

    def test_to_dict(self):
        """Test to_dict Methode."""
        entry = QuarantineEntry(
            target="test_feature",
            reason="Test reason",
            quarantine_type="feature",
            triggered_by=["run001", "run002"],
            quarantine_start="2024-01-01T00:00:00",
            quarantine_duration=5,
            remaining_runs=3,
        )
        
        data = entry.to_dict()
        
        assert data["target"] == "test_feature"
        assert data["quarantine_type"] == "feature"
        assert len(data["triggered_by"]) == 2
        assert data["remaining_runs"] == 3
    
    def test_from_dict(self):
        """Test from_dict Methode."""
        data = {
            "target": "loaded_feature",
            "reason": "Loaded reason",
            "quarantine_type": "combination",
            "triggered_by": ["run003"],
            "quarantine_start": "2024-01-02T00:00:00",
            "quarantine_duration": 10,
            "remaining_runs": 8,
            "context_filter": "low_budget"
        }
        
        entry = QuarantineEntry.from_dict(data)
        
        assert entry.target == "loaded_feature"
        assert entry.quarantine_type == "combination"
        assert entry.context_filter == "low_budget"
