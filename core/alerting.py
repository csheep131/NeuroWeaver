#!/usr/bin/env python3
"""
Alerting & Notification System für NeuroWeave.

Wann muss Human eingreifen?

Phase 4C: Guardrail System & Integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import uuid


class AlertSeverity(Enum):
"""Alert-Schweregrade."""

INFO = "info"
WARNING = "warning"
HIGH = "high"
CRITICAL = "critical"


@dataclass
class Alert:
"""Ein Alert."""

alert_id: str
severity: AlertSeverity
title: str
message: str
source: str # "guardrail_violation", "anomaly_detector", ...
timestamp: datetime
requires_action: bool
acknowledged: bool = False
acknowledged_by: Optional[str] = None
acknowledged_at: Optional[datetime] = None
resolved: bool = False
resolved_at: Optional[datetime] = None
resolved_by: Optional[str] = None
metadata: Dict = field(default_factory=dict)

def to_dict(self) -> Dict:
"""Konvertiere zu Dictionary."""
return {
"alert_id": self.alert_id,
"severity": self.severity.value,
"title": self.title,
"message": self.message,
"source": self.source,
"timestamp": self.timestamp.isoformat(),
"requires_action": self.requires_action,
"acknowledged": self.acknowledged,
"acknowledged_by": self.acknowledged_by,
"acknowledged_at": (
self.acknowledged_at.isoformat() if self.acknowledged_at else None
),
"resolved": self.resolved,
"resolved_at": (
self.resolved_at.isoformat() if self.resolved_at else None
),
"resolved_by": self.resolved_by,
"metadata": dict(self.metadata),
}

def is_active(self) -> bool:
"""Prüfen ob Alert noch aktiv ist."""
return not self.resolved


class AlertManager:
"""
Alerting-System für Human-Notification.

Alert-Typen:
1. Guardrail Violation (Hard-Limit überschritten)
2. Anomaly Detected (kritische Instabilität)
3. Approval Required (Human-Entscheidung nötig)
4. System Health (OOM-Rate zu hoch, etc.)
"""

def __init__(self) -> None:
"""Initialisiere AlertManager."""
self._alerts: Dict[str, Alert] = {}
self._notification_channels: List[str] = ["console"]
self._alert_history_limit = 1000

@property
def alerts(self) -> List[Alert]:
"""Alle Alerts zurückgeben."""
return list(self._alerts.values())

@property
def notification_channels(self) -> List[str]:
"""Konfigurierte Notification-Channels."""
return list(self._notification_channels)

def create_alert(
self,
severity: AlertSeverity,
title: str,
message: str,
source: str,
requires_action: bool = False,
metadata: Optional[Dict] = None,
) -> Alert:
"""
Alert erstellen.

Args:
severity: Schweregrad des Alerts
title: Kurzer Titel
message: Detaillierte Nachricht
source: Quelle des Alerts
requires_action: Ob Human-Aktion erforderlich ist
metadata: Zusätzliche Metadaten

Returns:
Der erstellte Alert
"""
alert_id = str(uuid.uuid4())
timestamp = datetime.utcnow()

alert = Alert(
alert_id=alert_id,
severity=severity,
title=title,
message=message,
source=source,
timestamp=timestamp,
requires_action=requires_action,
acknowledged=False,
metadata=metadata or {},
)

self._alerts[alert_id] = alert

# Sende Notifications für High/Critical Alerts
if severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
self.send_notifications(alert)

# History-Limit einhalten
self._enforce_history_limit()

return alert

def _enforce_history_limit(self) -> None:
"""Älteste Alerts entfernen wenn Limit erreicht."""
while len(self._alerts) > self._alert_history_limit:
oldest_id = min(
self._alerts.keys(),
key=lambda x: self._alerts[x].timestamp,
)
del self._alerts[oldest_id]

def get_unacknowledged_alerts(
self, severity: Optional[AlertSeverity] = None
) -> List[Alert]:
"""
Nicht bestätigte Alerts.

Args:
severity: Optionaler Filter nach Schweregrad

Returns:
Liste der nicht bestätigten Alerts
"""
alerts = []
for alert in self._alerts.values():
if not alert.acknowledged and not alert.resolved:
if severity is None or alert.severity == severity:
alerts.append(alert)

# Sortiere nach Schweregrad (kritisch zuerst) und Zeit
severity_order = {
AlertSeverity.CRITICAL: 0,
AlertSeverity.HIGH: 1,
AlertSeverity.WARNING: 2,
AlertSeverity.INFO: 3,
}
alerts.sort(
key=lambda x: (severity_order.get(x.severity, 4), x.timestamp),
reverse=False,
)

return alerts

def acknowledge_alert(
self, alert_id: str, acknowledged_by: str
) -> bool:
"""
Alert bestätigen.

Args:
alert_id: ID des Alerts
acknowledged_by: Name/Rolle des Bestätigers

Returns:
True wenn erfolgreich bestätigt
"""
alert = self._alerts.get(alert_id)
if alert is None:
return False

if alert.acknowledged or alert.resolved:
return False

alert.acknowledged = True
alert.acknowledged_by = acknowledged_by
alert.acknowledged_at = datetime.utcnow()

return True

def resolve_alert(
self, alert_id: str, resolved_by: str
) -> bool:
"""
Alert auflösen.

Args:
alert_id: ID des Alerts
resolved_by: Name/Rolle des Auflösers

Returns:
True wenn erfolgreich aufgelöst
"""
alert = self._alerts.get(alert_id)
if alert is None:
return False

if alert.resolved:
return False

alert.resolved = True
alert.resolved_by = resolved_by
alert.resolved_at = datetime.utcnow()

return True

def send_notifications(self, alert: Alert) -> int:
"""
Notifications versenden.

Channels:
- Console (immer)
- Email (bei High/Critical)
- Slack (konfigurierbar)

Args:
alert: Der Alert für den Notifications gesendet werden

Returns:
Anzahl gesendeter Notifications
"""
sent_count = 0

# Console-Notification (immer)
self._send_console_notification(alert)
sent_count += 1

# Email-Notification bei High/Critical
if alert.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
if "email" in self._notification_channels:
self._send_email_notification(alert)
sent_count += 1

# Slack-Notification wenn konfiguriert
if "slack" in self._notification_channels:
self._send_slack_notification(alert)
sent_count += 1

return sent_count

def _send_console_notification(self, alert: Alert) -> None:
"""
Console-Notification senden.

Args:
alert: Der Alert
"""
severity_icons = {
AlertSeverity.INFO: "",
AlertSeverity.WARNING: "",
AlertSeverity.HIGH: "",
AlertSeverity.CRITICAL: "",
}

icon = severity_icons.get(alert.severity, "•")
timestamp_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")

print(
f"{icon} [{timestamp_str}] {alert.title}\n"
f" {alert.message}\n"
f" Quelle: {alert.source} | "
f"Action required: {'Ja' if alert.requires_action else 'Nein'}"
)

def _send_email_notification(self, alert: Alert) -> None:
"""
Email-Notification senden (Placeholder).

Args:
alert: Der Alert
"""
# Placeholder für tatsächliche Email-Sendung
# In Produktion: SMTP-Server, Templates, etc.
pass

def _send_slack_notification(self, alert: Alert) -> None:
"""
Slack-Notification senden (Placeholder).

Args:
alert: Der Alert
"""
# Placeholder für tatsächliche Slack-Sendung
# In Produktion: Webhook, Channels, etc.
pass

def get_alert_summary(self, hours: int = 24) -> Dict:
"""
Alert-Zusammenfassung.

Args:
hours: Zeitraum in Stunden (default: 24)

Returns:
Dictionary mit Zusammenfassung
"""
now = datetime.utcnow()
cutoff = now - timedelta(hours=hours)

recent_alerts = [
a for a in self._alerts.values()
if a.timestamp >= cutoff
]

total = len(recent_alerts)

# Nach Schweregrad gruppieren
by_severity: Dict[str, int] = {}
for alert in recent_alerts:
severity_key = alert.severity.value
by_severity[severity_key] = by_severity.get(severity_key, 0) + 1

# acknowledged vs pending
acknowledged = sum(1 for a in recent_alerts if a.acknowledged)
pending = sum(
1 for a in recent_alerts
if not a.acknowledged and not a.resolved
)
resolved = sum(1 for a in recent_alerts if a.resolved)

# Action-required Alerts
action_required = sum(
1 for a in recent_alerts
if a.requires_action and not a.resolved
)

return {
"total": total,
"by_severity": by_severity,
"acknowledged": acknowledged,
"pending": pending,
"resolved": resolved,
"action_required": action_required,
"time_range_hours": hours,
}

def add_notification_channel(self, channel: str) -> None:
"""
Notification-Channel hinzufügen.

Args:
channel: Channel-Name ("email", "slack", etc.)
"""
if channel not in self._notification_channels:
self._notification_channels.append(channel)

def remove_notification_channel(self, channel: str) -> None:
"""
Notification-Channel entfernen.

Args:
channel: Channel-Name
"""
if channel in self._notification_channels:
self._notification_channels.remove(channel)

def get_alerts_by_source(self, source: str) -> List[Alert]:
"""
Alerts nach Quelle filtern.

Args:
source: Quelle zum Filtern

Returns:
Liste der Alerts von dieser Quelle
"""
return [a for a in self._alerts.values() if a.source == source]

def get_active_alerts(self) -> List[Alert]:
"""
Alle aktiven (nicht aufgelösten) Alerts.

Returns:
Liste der aktiven Alerts
"""
return [a for a in self._alerts.values() if a.is_active()]

def clear_resolved(self) -> int:
"""
Aufgelöste Alerts bereinigen.

Returns:
Anzahl bereinigter Alerts
"""
resolved_ids = [
aid for aid, alert in self._alerts.items()
if alert.resolved
]

for alert_id in resolved_ids:
del self._alerts[alert_id]

return len(resolved_ids)

def create_guardrail_violation_alert(
self,
guardrail_name: str,
violation_message: str,
is_hard_limit: bool,
) -> Alert:
"""
Helper: Guardrail Violation Alert erstellen.

Args:
guardrail_name: Name der verletzten Guardrail
violation_message: Details der Verletzung
is_hard_limit: Ob es eine Hard-Limit Verletzung ist

Returns:
Der erstellte Alert
"""
severity = (
AlertSeverity.HIGH if is_hard_limit else AlertSeverity.WARNING
)

return self.create_alert(
severity=severity,
title=f"Guardrail Violation: {guardrail_name}",
message=violation_message,
source="guardrail_violation",
requires_action=is_hard_limit,
metadata={
"guardrail_name": guardrail_name,
"is_hard_limit": is_hard_limit,
},
)

def create_anomaly_alert(
self,
anomaly_type: str,
details: str,
confidence: float,
) -> Alert:
"""
Helper: Anomaly Detected Alert erstellen.

Args:
anomaly_type: Typ der Anomalie
details: Details der Anomalie
confidence: Confidence der Erkennung

Returns:
Der erstellte Alert
"""
severity = AlertSeverity.HIGH if confidence > 0.9 else AlertSeverity.WARNING

return self.create_alert(
severity=severity,
title=f"Anomaly Detected: {anomaly_type}",
message=details,
source="anomaly_detector",
requires_action=True,
metadata={
"anomaly_type": anomaly_type,
"confidence": confidence,
},
)

def create_approval_required_alert(
self,
action_type: str,
request_id: str,
risk_level: str,
) -> Alert:
"""
Helper: Approval Required Alert erstellen.

Args:
action_type: Typ der Aktion
request_id: ID der Approval-Anfrage
risk_level: Risk-Level der Aktion

Returns:
Der erstellte Alert
"""
severity_map = {
"low": AlertSeverity.INFO,
"medium": AlertSeverity.WARNING,
"high": AlertSeverity.HIGH,
}
severity = severity_map.get(risk_level, AlertSeverity.WARNING)

return self.create_alert(
severity=severity,
title=f"Approval Required: {action_type}",
message=f"Human-Freigabe erforderlich für '{action_type}' (Risk: {risk_level})",
source="approval_required",
requires_action=True,
metadata={
"action_type": action_type,
"request_id": request_id,
"risk_level": risk_level,
},
)

def create_system_health_alert(
self,
health_issue: str,
metric_value: float,
threshold: float,
) -> Alert:
"""
Helper: System Health Alert erstellen.

Args:
health_issue: Beschreibung des Problems
metric_value: Aktueller Metrik-Wert
threshold: Threshold der verletzt wurde

Returns:
Der erstellte Alert
"""
severity = AlertSeverity.CRITICAL if metric_value > threshold * 2 else AlertSeverity.HIGH

return self.create_alert(
severity=severity,
title=f"System Health: {health_issue}",
message=f"{health_issue}: Wert {metric_value:.2f} (Threshold: {threshold:.2f})",
source="system_health",
requires_action=True,
metadata={
"health_issue": health_issue,
"metric_value": metric_value,
"threshold": threshold,
},
)


def create_alert_manager() -> AlertManager:
"""
Factory-Funktion für AlertManager.

Returns:
Konfigurierter AlertManager
"""
return AlertManager()
