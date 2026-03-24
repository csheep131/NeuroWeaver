#!/usr/bin/env python3
"""
Tests für Alerting System.

Phase 4C: Guardrail System & Integration
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Füge Parent-Directory zum Path hinzu für direkte Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.alerting import (
    AlertSeverity,
    Alert,
    AlertManager,
    create_alert_manager,
)


class TestAlertSeverity:
    """Tests für AlertSeverity Enum."""

    def test_severity_values(self):
        """Severity-Werte prüfen."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlert:
    """Tests für Alert-Klasse."""

    def test_alert_creation(self):
        """Alert erstellen."""
        alert = Alert(
            alert_id="alert-123",
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            source="test_source",
            timestamp=datetime.utcnow(),
            requires_action=True,
        )

        assert alert.alert_id == "alert-123"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.title == "Test Alert"
        assert alert.requires_action is True
        assert alert.acknowledged is False
        assert alert.resolved is False

    def test_alert_to_dict(self):
        """Alert zu Dictionary konvertieren."""
        alert = Alert(
            alert_id="alert-456",
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test",
            source="test",
            timestamp=datetime.utcnow(),
            requires_action=False,
            metadata={"key": "value"},
        )

        result_dict = alert.to_dict()

        assert result_dict["alert_id"] == "alert-456"
        assert result_dict["severity"] == "warning"
        assert result_dict["metadata"]["key"] == "value"

    def test_alert_is_active(self):
        """Prüfen ob Alert aktiv."""
        alert = Alert(
            alert_id="alert-active",
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            source="test",
            timestamp=datetime.utcnow(),
            requires_action=False,
        )

        assert alert.is_active() is True

    def test_alert_not_active_resolved(self):
        """Aufgelöster Alert ist nicht aktiv."""
        alert = Alert(
            alert_id="alert-resolved",
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            source="test",
            timestamp=datetime.utcnow(),
            requires_action=False,
            resolved=True,
        )

        assert alert.is_active() is False


class TestAlertManager:
    """Tests für AlertManager-Klasse."""

    @pytest.fixture
    def manager(self):
        """Manager Fixture."""
        return create_alert_manager()

    def test_manager_initialization(self, manager):
        """Manager Initialisierung."""
        assert len(manager.alerts) == 0
        assert "console" in manager.notification_channels

    def test_create_alert(self, manager):
        """Alert erstellen."""
        alert = manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
            source="test_source",
        )

        assert alert.alert_id is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.acknowledged is False

        alerts = manager.alerts
        assert len(alerts) == 1

    def test_create_alert_with_metadata(self, manager):
        """Alert mit Metadaten erstellen."""
        alert = manager.create_alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            source="test",
            metadata={"custom_key": "custom_value"},
        )

        assert alert.metadata["custom_key"] == "custom_value"

    def test_create_high_severity_alert(self, manager):
        """High-Severity Alert erstellt Notification."""
        alert = manager.create_alert(
            severity=AlertSeverity.HIGH,
            title="High Severity",
            message="Test",
            source="test",
            requires_action=True,
        )

        assert alert.severity == AlertSeverity.HIGH
        assert alert.requires_action is True

    def test_get_unacknowledged_alerts(self, manager):
        """Nicht bestätigte Alerts abrufen."""
        manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="Unacknowledged",
            message="Test",
            source="test",
        )

        unacknowledged = manager.get_unacknowledged_alerts()

        assert len(unacknowledged) == 1
        assert unacknowledged[0].title == "Unacknowledged"

    def test_get_unacknowledged_alerts_filtered(self, manager):
        """Nicht bestätigte Alerts nach Severity gefiltert."""
        manager.create_alert(
            severity=AlertSeverity.INFO,
            title="Info Alert",
            message="Test",
            source="test",
        )
        manager.create_alert(
            severity=AlertSeverity.HIGH,
            title="High Alert",
            message="Test",
            source="test",
        )

        high_alerts = manager.get_unacknowledged_alerts(AlertSeverity.HIGH)

        assert len(high_alerts) == 1
        assert high_alerts[0].severity == AlertSeverity.HIGH

    def test_acknowledge_alert(self, manager):
        """Alert bestätigen."""
        alert = manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="To Acknowledge",
            message="Test",
            source="test",
        )

        success = manager.acknowledge_alert(alert.alert_id, "test_user")

        assert success is True

        updated = manager.alerts[0]
        assert updated.acknowledged is True
        assert updated.acknowledged_by == "test_user"

    def test_acknowledge_nonexistent_alert(self, manager):
        """Nicht-existenten Alert bestätigen."""
        success = manager.acknowledge_alert("nonexistent-id", "user")

        assert success is False

    def test_acknowledge_already_acknowledged_alert(self, manager):
        """Bereits bestätigten Alert erneut bestätigen."""
        alert = manager.create_alert(
            severity=AlertSeverity.INFO,
            title="Already Acked",
            message="Test",
            source="test",
        )
        manager.acknowledge_alert(alert.alert_id, "user1")

        success = manager.acknowledge_alert(alert.alert_id, "user2")

        assert success is False

    def test_resolve_alert(self, manager):
        """Alert auflösen."""
        alert = manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="To Resolve",
            message="Test",
            source="test",
        )

        success = manager.resolve_alert(alert.alert_id, "test_user")

        assert success is True

        updated = manager.alerts[0]
        assert updated.resolved is True
        assert updated.resolved_by == "test_user"

    def test_resolve_nonexistent_alert(self, manager):
        """Nicht-existenten Alert auflösen."""
        success = manager.resolve_alert("nonexistent-id", "user")

        assert success is False

    def test_resolve_already_resolved_alert(self, manager):
        """Bereits aufgelösten Alert erneut auflösen."""
        alert = manager.create_alert(
            severity=AlertSeverity.INFO,
            title="Already Resolved",
            message="Test",
            source="test",
        )
        manager.resolve_alert(alert.alert_id, "user1")

        success = manager.resolve_alert(alert.alert_id, "user2")

        assert success is False

    def test_get_alert_summary(self, manager):
        """Alert-Zusammenfassung."""
        manager.create_alert(
            severity=AlertSeverity.INFO,
            title="Info",
            message="Test",
            source="test",
        )
        manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="Warning",
            message="Test",
            source="test",
        )
        manager.create_alert(
            severity=AlertSeverity.HIGH,
            title="High",
            message="Test",
            source="test",
        )

        summary = manager.get_alert_summary(hours=24)

        assert summary["total"] == 3
        assert "by_severity" in summary
        assert summary["by_severity"]["info"] == 1
        assert summary["by_severity"]["warning"] == 1
        assert summary["by_severity"]["high"] == 1

    def test_get_alert_summary_empty(self, manager):
        """Zusammenfassung wenn keine Alerts."""
        summary = manager.get_alert_summary(hours=24)

        assert summary["total"] == 0
        assert summary["pending"] == 0

    def test_add_notification_channel(self, manager):
        """Notification-Channel hinzufügen."""
        manager.add_notification_channel("slack")

        assert "slack" in manager.notification_channels

    def test_add_duplicate_notification_channel(self, manager):
        """Duplizierten Channel hinzufügen."""
        manager.add_notification_channel("email")
        manager.add_notification_channel("email")

        count = manager.notification_channels.count("email")
        assert count == 1

    def test_remove_notification_channel(self, manager):
        """Notification-Channel entfernen."""
        manager.add_notification_channel("slack")
        manager.remove_notification_channel("slack")

        assert "slack" not in manager.notification_channels

    def test_get_alerts_by_source(self, manager):
        """Alerts nach Quelle filtern."""
        manager.create_alert(
            severity=AlertSeverity.INFO,
            title="From Source A",
            message="Test",
            source="source_a",
        )
        manager.create_alert(
            severity=AlertSeverity.INFO,
            title="From Source B",
            message="Test",
            source="source_b",
        )

        source_a_alerts = manager.get_alerts_by_source("source_a")

        assert len(source_a_alerts) == 1
        assert source_a_alerts[0].source == "source_a"

    def test_get_active_alerts(self, manager):
        """Aktive Alerts abrufen."""
        manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="Active",
            message="Test",
            source="test",
        )
        alert_resolved = manager.create_alert(
            severity=AlertSeverity.INFO,
            title="Resolved",
            message="Test",
            source="test",
        )
        manager.resolve_alert(alert_resolved.alert_id, "user")

        active = manager.get_active_alerts()

        assert len(active) == 1
        assert active[0].title == "Active"

    def test_clear_resolved(self, manager):
        """Aufgelöste Alerts bereinigen."""
        alert1 = manager.create_alert(
            severity=AlertSeverity.INFO,
            title="To Keep",
            message="Test",
            source="test",
        )
        alert2 = manager.create_alert(
            severity=AlertSeverity.INFO,
            title="To Remove",
            message="Test",
            source="test",
        )
        manager.resolve_alert(alert2.alert_id, "user")

        cleared = manager.clear_resolved()

        assert cleared == 1
        assert len(manager.alerts) == 1

    def test_create_guardrail_violation_alert(self, manager):
        """Guardrail Violation Alert erstellen."""
        alert = manager.create_guardrail_violation_alert(
            guardrail_name="Budget Limit",
            violation_message="Budget exceeded",
            is_hard_limit=True,
        )

        assert alert.severity == AlertSeverity.HIGH
        assert alert.source == "guardrail_violation"
        assert alert.requires_action is True
        assert alert.metadata["guardrail_name"] == "Budget Limit"

    def test_create_guardrail_violation_soft_limit(self, manager):
        """Soft-Limit Guardrail Violation Alert."""
        alert = manager.create_guardrail_violation_alert(
            guardrail_name="Exploration Ratio",
            violation_message="Exploration too high",
            is_hard_limit=False,
        )

        assert alert.severity == AlertSeverity.WARNING
        assert alert.requires_action is False

    def test_create_anomaly_alert(self, manager):
        """Anomaly Alert erstellen."""
        alert = manager.create_anomaly_alert(
            anomaly_type="Performance Degradation",
            details="Performance dropped by 30%",
            confidence=0.95,
        )

        assert alert.severity == AlertSeverity.HIGH
        assert alert.source == "anomaly_detector"
        assert alert.metadata["confidence"] == 0.95

    def test_create_anomaly_alert_low_confidence(self, manager):
        """Anomaly Alert mit niedriger Confidence."""
        alert = manager.create_anomaly_alert(
            anomaly_type="Minor Anomaly",
            details="Small deviation",
            confidence=0.6,
        )

        assert alert.severity == AlertSeverity.WARNING

    def test_create_approval_required_alert(self, manager):
        """Approval Required Alert erstellen."""
        alert = manager.create_approval_required_alert(
            action_type="submit_bundle",
            request_id="req-123",
            risk_level="high",
        )

        assert alert.severity == AlertSeverity.HIGH
        assert alert.source == "approval_required"
        assert alert.requires_action is True

    def test_create_approval_required_alert_low_risk(self, manager):
        """Approval Required Alert mit niedrigem Risk."""
        alert = manager.create_approval_required_alert(
            action_type="propose_runs",
            request_id="req-456",
            risk_level="low",
        )

        assert alert.severity == AlertSeverity.INFO

    def test_create_system_health_alert(self, manager):
        """System Health Alert erstellen."""
        alert = manager.create_system_health_alert(
            health_issue="High OOM Rate",
            metric_value=0.25,
            threshold=0.1,
        )

        assert alert.severity == AlertSeverity.HIGH
        assert alert.source == "system_health"
        assert alert.requires_action is True

    def test_create_system_health_alert_critical(self, manager):
        """Kritischer System Health Alert."""
        alert = manager.create_system_health_alert(
            health_issue="Critical Memory Issue",
            metric_value=0.5,
            threshold=0.1,
        )

        assert alert.severity == AlertSeverity.CRITICAL


class TestCreateAlertManager:
    """Tests für create_alert_manager Funktion."""

    def test_create_manager(self):
        """Manager erstellen."""
        manager = create_alert_manager()

        assert isinstance(manager, AlertManager)
        assert "console" in manager.notification_channels
