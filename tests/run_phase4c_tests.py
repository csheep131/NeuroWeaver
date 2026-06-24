#!/usr/bin/env python3
"""
Test Runner für Phase 4C Module.

Führt Tests ohne pytest durch um Import-Probleme zu vermeiden.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta
from importlib.util import spec_from_file_location, module_from_spec

PROJECT_ROOT = Path(__file__).parent.parent


def load_module(module_name: str, file_path: Path):
"""Lade Modul direkt von Pfad."""
# Registriere Modul zuerst in sys.modules
module = type(sys)(module_name)
module.__file__ = str(file_path)
module.__package__ = module_name.rsplit('.', 1)[0] if '.' in module_name else ''
sys.modules[module_name] = module

# Führe Code aus
with open(file_path, 'r', encoding='utf-8') as f:
code = f.read()
exec(code, module.__dict__)
return module


def run_tests():
"""Führe alle Tests aus."""
print("=" * 60)
print("Phase 4C Module Tests")
print("=" * 60)

# Füge PROJECT_ROOT zum sys.path für Imports
sys.path.insert(0, str(PROJECT_ROOT))

# Lade Module in Abhängigkeits-Reihenfolge
print("\n Lade Module...")

# 1. guardrails (keine internen Abhängigkeiten)
guardrails = load_module("orchestrator.guardrails", PROJECT_ROOT / "orchestrator" / "guardrails.py")
print(" guardrails")

# 2. autonomy_orchestrator (hängt von guardrails ab)
autonomy_orchestrator = load_module(
"orchestrator.autonomy_orchestrator",
PROJECT_ROOT / "orchestrator" / "autonomy_orchestrator.py"
)
print(" autonomy_orchestrator")

# 3. approval_interface (hängt von autonomy_orchestrator ab)
approval_interface = load_module(
"orchestrator.approval_interface",
PROJECT_ROOT / "orchestrator" / "approval_interface.py"
)
print(" approval_interface")

# 4. alerting (keine internen Abhängigkeiten)
alerting = load_module("core.alerting", PROJECT_ROOT / "core" / "alerting.py")
print(" alerting")

# 5. override_learner (keine internen Abhängigkeiten)
override_learner = load_module(
"research.override_learner",
PROJECT_ROOT / "research" / "override_learner.py"
)
print(" override_learner")

print(" Alle Module geladen")

# Test-Zähler
passed = 0
failed = 0

# ===== Guardrails Tests =====
print("\n" + "=" * 60)
print(" Guardrail Tests")
print("=" * 60)

# Test 1: Guardrail check passes
try:
g = guardrails.Guardrail(
guardrail_type=guardrails.GuardrailType.BUDGET,
name="Test",
description="Test",
threshold=100.0,
is_hard_limit=True,
action_on_violation="block",
)
ok, msg = g.check(50.0)
assert ok is True
print(" Guardrail check passes")
passed += 1
except Exception as e:
print(f" Guardrail check passes: {e}")
failed += 1

# Test 2: Guardrail check violation
try:
ok, msg = g.check(150.0)
assert ok is False
assert msg is not None
print(" Guardrail check violation")
passed += 1
except Exception as e:
print(f" Guardrail check violation: {e}")
failed += 1

# Test 3: create_default_guardrails
try:
config = guardrails.create_default_guardrails()
assert len(config.guardrails) == 5
assert config.level == guardrails.AutonomyLevel.SUPERVISED
print(" create_default_guardrails")
passed += 1
except Exception as e:
print(f" create_default_guardrails: {e}")
failed += 1

# Test 4: GuardrailManager check_action
try:
mgr = guardrails.GuardrailManager(config)
ok, violations = mgr.check_action("propose_runs", {
"budget_used": 45.0,
"confidence": 0.78,
"exploration_ratio": 0.3,
"safety_score": 0.9,
})
assert ok is True, f"Expected True but got {ok}, violations: {violations}"
print(" GuardrailManager check_action allowed")
passed += 1
except Exception as e:
print(f" GuardrailManager check_action: {e}")
failed += 1

# Test 5: GuardrailManager blocked (niedrige Confidence)
try:
ok, violations = mgr.check_action("propose_runs", {
"confidence": 0.4, # Unter Minimum von 0.6
"budget_used": 45.0,
"exploration_ratio": 0.3,
"safety_score": 0.9,
})
assert ok is False, f"Expected False but got {ok}"
print(" GuardrailManager check_action blocked")
passed += 1
except Exception as e:
print(f" GuardrailManager blocked: {e}")
failed += 1

# ===== Autonomy Orchestrator Tests =====
print("\n" + "=" * 60)
print(" Autonomy Orchestrator Tests")
print("=" * 60)

import asyncio

# Test 6: create_autonomy_orchestrator
try:
orch = autonomy_orchestrator.create_autonomy_orchestrator()
assert orch is not None
print(" create_autonomy_orchestrator")
passed += 1
except Exception as e:
print(f" create_autonomy_orchestrator: {e}")
failed += 1

# Test 7: propose_action
async def test_propose():
orch = autonomy_orchestrator.create_autonomy_orchestrator()
action = await orch.propose_action("propose_runs", {"confidence": 0.75})
assert action.action_type == "propose_runs"
return True

try:
result = asyncio.run(test_propose())
print(" propose_action")
passed += 1
except Exception as e:
print(f" propose_action: {e}")
failed += 1

# Test 8: get_statistics
try:
orch = autonomy_orchestrator.create_autonomy_orchestrator()
stats = orch.get_statistics()
assert "total_actions" in stats
print(" get_statistics")
passed += 1
except Exception as e:
print(f" get_statistics: {e}")
failed += 1

# ===== Approval Interface Tests =====
print("\n" + "=" * 60)
print(" Approval Interface Tests")
print("=" * 60)

# Test 9: create_approval_interface
try:
orch = autonomy_orchestrator.create_autonomy_orchestrator()
iface = approval_interface.create_approval_interface(orch)
assert iface is not None
print(" create_approval_interface")
passed += 1
except Exception as e:
print(f" create_approval_interface: {e}")
failed += 1

# Test 10: get_approval_statistics
try:
stats = iface.get_approval_statistics()
assert "pending" in stats
print(" get_approval_statistics")
passed += 1
except Exception as e:
print(f" get_approval_statistics: {e}")
failed += 1

# ===== Alerting Tests =====
print("\n" + "=" * 60)
print(" Alerting Tests")
print("=" * 60)

# Test 11: create_alert_manager
try:
alert_mgr = alerting.create_alert_manager()
assert alert_mgr is not None
print(" create_alert_manager")
passed += 1
except Exception as e:
print(f" create_alert_manager: {e}")
failed += 1

# Test 12: create_alert
try:
alert = alert_mgr.create_alert(
severity=alerting.AlertSeverity.WARNING,
title="Test Alert",
message="Test message",
source="test",
)
assert alert.alert_id is not None
print(" create_alert")
passed += 1
except Exception as e:
print(f" create_alert: {e}")
failed += 1

# Test 13: get_alert_summary
try:
summary = alert_mgr.get_alert_summary(hours=24)
assert "total" in summary
print(" get_alert_summary")
passed += 1
except Exception as e:
print(f" get_alert_summary: {e}")
failed += 1

# ===== Override Learner Tests =====
print("\n" + "=" * 60)
print(" Override Learner Tests")
print("=" * 60)

# Test 14: create_override_learner
try:
learner = override_learner.create_override_learner()
assert learner is not None
print(" create_override_learner")
passed += 1
except Exception as e:
print(f" create_override_learner: {e}")
failed += 1

# Test 15: log_override
try:
event = learner.log_override(
original_action="test_action",
original_decision="execute",
human_decision="block",
context={},
)
assert event.override_id is not None
print(" log_override")
passed += 1
except Exception as e:
print(f" log_override: {e}")
failed += 1

# Test 16: analyze_override_patterns
try:
patterns = learner.analyze_override_patterns()
assert "total_overrides" in patterns
print(" analyze_override_patterns")
passed += 1
except Exception as e:
print(f" analyze_override_patterns: {e}")
failed += 1

# Test 17: calibrate_confidence
try:
result = learner.calibrate_confidence(0.8, 0.75)
assert "calibration_factor" in result
print(" calibrate_confidence")
passed += 1
except Exception as e:
print(f" calibrate_confidence: {e}")
failed += 1

# ===== Summary =====
print("\n" + "=" * 60)
print(" Test Summary")
print("=" * 60)
print(f" Passed: {passed}")
print(f" Failed: {failed}")
print(f" Success Rate: {passed/(passed+failed)*100:.1f}%")

return failed == 0


if __name__ == "__main__":
success = run_tests()
sys.exit(0 if success else 1)
