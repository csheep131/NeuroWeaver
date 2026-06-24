#!/usr/bin/env python3
"""
Tests für Phase 4A Module.

Enthält Tests für:
- Surrogate Scorer
- Hypothesis Generator
- Pareto Tracker
- Adaptive Kill Thresholds
"""

import json
import pickle
import tempfile
from pathlib import Path
from typing import List

import pytest

from core.meta_features import RunMetaFeatures
from research.surrogate_scorer import SurrogateScorer
from research.hypothesis_generator import HypothesisGenerator, RunHypothesis
from research.pareto_tracker import ParetoTracker, ParetoPoint
from research.adaptive_kill_thresholds import (
AdaptiveKillThresholdManager,
KillThresholds,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_meta_features() -> List[RunMetaFeatures]:
"""Erstelle Sample Meta-Features für Tests."""
return [
RunMetaFeatures(
run_id="run_001",
features_active=["gqa", "film"],
parent_run_id=None,
lineage_depth=0,
siblings_count=2,
budget_class="medium",
sequence_length="local",
quantization_type="none",
step_time_ms=10.5,
memory_usage_mb=1024.0,
training_stability=0.85,
delta_bpb_vs_parent=-0.015,
efficiency_gain_percent=5.2,
model_size_change_percent=0.0,
quant_gap=0.0,
),
RunMetaFeatures(
run_id="run_002",
features_active=["gqa", "leaky_relu"],
parent_run_id="run_001",
lineage_depth=1,
siblings_count=1,
budget_class="medium",
sequence_length="local",
quantization_type="int6",
step_time_ms=9.8,
memory_usage_mb=950.0,
training_stability=0.78,
delta_bpb_vs_parent=-0.022,
efficiency_gain_percent=8.5,
model_size_change_percent=-15.0,
quant_gap=0.008,
),
RunMetaFeatures(
run_id="run_003",
features_active=["mixed_quant", "swiglu"],
parent_run_id="run_001",
lineage_depth=1,
siblings_count=1,
budget_class="high",
sequence_length="remote",
quantization_type="mixed",
step_time_ms=12.3,
memory_usage_mb=1200.0,
training_stability=0.92,
delta_bpb_vs_parent=0.005,
efficiency_gain_percent=-2.1,
model_size_change_percent=-25.0,
quant_gap=0.018,
),
RunMetaFeatures(
run_id="run_004",
features_active=["gqa", "film", "leaky_relu"],
parent_run_id="run_002",
lineage_depth=2,
siblings_count=0,
budget_class="low",
sequence_length="local",
quantization_type="none",
step_time_ms=11.0,
memory_usage_mb=1100.0,
training_stability=0.65,
delta_bpb_vs_parent=-0.008,
efficiency_gain_percent=3.2,
model_size_change_percent=5.0,
quant_gap=0.0,
),
RunMetaFeatures(
run_id="run_005",
features_active=["rope", "flash_attn"],
parent_run_id=None,
lineage_depth=0,
siblings_count=3,
budget_class="high",
sequence_length="remote",
quantization_type="gptq_lite",
step_time_ms=8.5,
memory_usage_mb=800.0,
training_stability=0.88,
delta_bpb_vs_parent=-0.030,
efficiency_gain_percent=12.0,
model_size_change_percent=-10.0,
quant_gap=0.012,
),
RunMetaFeatures(
run_id="run_006",
features_active=["gqa"],
parent_run_id="run_005",
lineage_depth=1,
siblings_count=2,
budget_class="medium",
sequence_length="local",
quantization_type="none",
step_time_ms=10.0,
memory_usage_mb=1000.0,
training_stability=0.80,
delta_bpb_vs_parent=-0.012,
efficiency_gain_percent=4.5,
model_size_change_percent=0.0,
quant_gap=0.0,
),
]


@pytest.fixture
def trained_scorer(sample_meta_features: List[RunMetaFeatures]) -> SurrogateScorer:
"""Erstelle einen trainierten Surrogate Scorer."""
scorer = SurrogateScorer(model_type="random_forest")

# Filtere Features mit vollständigen Targets
train_features = [
f for f in sample_meta_features
if f.delta_bpb_vs_parent is not None and f.efficiency_gain_percent is not None
]

targets = {
"delta_bpb": [f.delta_bpb_vs_parent for f in train_features],
"efficiency_gain": [f.efficiency_gain_percent for f in train_features],
}

scorer.train(train_features, targets)
return scorer


# ============================================================================
# Surrogate Scorer Tests
# ============================================================================


class TestSurrogateScorer:
"""Tests für SurrogateScorer."""

def test_init_default(self):
"""Test Initialisierung mit Default-Werten."""
scorer = SurrogateScorer()
assert scorer.model_type == "random_forest"
assert scorer.bpb_model is None
assert not scorer._is_trained

def test_init_gradient_boosting(self):
"""Test Initialisierung mit Gradient Boosting."""
scorer = SurrogateScorer(model_type="gradient_boosting")
assert scorer.model_type == "gradient_boosting"

def test_init_invalid_model_type(self):
"""Test Initialisierung mit ungültigem Modell-Typ."""
with pytest.raises(ValueError, match="Ungültiger model_type"):
SurrogateScorer(model_type="invalid_model")

def test_train_success(self, sample_meta_features: List[RunMetaFeatures]):
"""Test erfolgreiches Training."""
scorer = SurrogateScorer()

train_features = [
f for f in sample_meta_features
if f.delta_bpb_vs_parent is not None and f.efficiency_gain_percent is not None
]

targets = {
"delta_bpb": [f.delta_bpb_vs_parent for f in train_features],
"efficiency_gain": [f.efficiency_gain_percent for f in train_features],
}

metrics = scorer.train(train_features, targets)

assert scorer._is_trained
assert scorer.bpb_model is not None
assert scorer.efficiency_model is not None
assert "bpb_cv_rmse" in metrics
assert "efficiency_cv_rmse" in metrics

def test_train_empty_features(self):
"""Test Training mit leeren Features."""
scorer = SurrogateScorer()

with pytest.raises(ValueError, match="Keine Features"):
scorer.train([], {"delta_bpb": [], "efficiency_gain": []})

def test_train_missing_targets(self, sample_meta_features: List[RunMetaFeatures]):
"""Test Training mit fehlenden Targets."""
scorer = SurrogateScorer()

with pytest.raises(ValueError, match="Targets müssen"):
scorer.train(sample_meta_features, {"delta_bpb": []})

def test_train_length_mismatch(self, sample_meta_features: List[RunMetaFeatures]):
"""Test Training mit Längen-Ungleichheit."""
scorer = SurrogateScorer()

with pytest.raises(ValueError, match="müssen gleiche Länge haben"):
scorer.train(
sample_meta_features,
{"delta_bpb": [0.1, 0.2], "efficiency_gain": [0.1, 0.2, 0.3]},
)

def test_predict_success(self, trained_scorer: SurrogateScorer, sample_meta_features: List[RunMetaFeatures]):
"""Test erfolgreiche Vorhersage."""
features = sample_meta_features[0]

predicted_bpb, predicted_eff, confidence = trained_scorer.predict(features)

assert isinstance(predicted_bpb, float)
assert isinstance(predicted_eff, float)
assert isinstance(confidence, float)
assert 0.0 <= confidence <= 1.0

def test_predict_not_trained(self, sample_meta_features: List[RunMetaFeatures]):
"""Test Vorhersage ohne Training."""
scorer = SurrogateScorer()

with pytest.raises(RuntimeError, match="Modell muss zuerst trainiert werden"):
scorer.predict(sample_meta_features[0])

def test_get_feature_importance(self, trained_scorer: SurrogateScorer):
"""Test Feature-Importance Extraktion."""
importance = trained_scorer.get_feature_importance()

assert isinstance(importance, dict)
assert len(importance) > 0
assert all(isinstance(v, float) for v in importance.values())

def test_save_and_load(self, trained_scorer: SurrogateScorer):
"""Test Speichern und Laden des Modells."""
with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
temp_path = f.name

try:
# Speichern
trained_scorer.save(temp_path)

# Neues Modell erstellen und laden
loaded_scorer = SurrogateScorer()
loaded_scorer.load(temp_path)

assert loaded_scorer._is_trained
assert loaded_scorer.model_type == trained_scorer.model_type

# Vorhersage-Vergleich
test_features = RunMetaFeatures(
run_id="test",
features_active=["gqa"],
lineage_depth=1,
siblings_count=2,
budget_class="medium",
sequence_length="local",
quantization_type="none",
)

pred_orig = trained_scorer.predict(test_features)
pred_loaded = loaded_scorer.predict(test_features)

assert abs(pred_orig[0] - pred_loaded[0]) < 1e-6
assert abs(pred_orig[1] - pred_loaded[1]) < 1e-6

finally:
Path(temp_path).unlink(missing_ok=True)

def test_load_file_not_found(self):
"""Test Laden einer nicht existierenden Datei."""
scorer = SurrogateScorer()

with pytest.raises(FileNotFoundError):
scorer.load("/nonexistent/path/model.pkl")

def test_to_json(self, trained_scorer: SurrogateScorer):
"""Test JSON-Export."""
json_str = trained_scorer.to_json()

assert isinstance(json_str, str)
data = json.loads(json_str)

assert "model_type" in data
assert "feature_importance" in data
assert "n_features" in data


# ============================================================================
# Hypothesis Generator Tests
# ============================================================================


class TestHypothesisGenerator:
"""Tests für HypothesisGenerator."""

def test_init(self, trained_scorer: SurrogateScorer, sample_meta_features: List[RunMetaFeatures]):
"""Test Initialisierung."""
generator = HypothesisGenerator(trained_scorer, sample_meta_features)

assert generator.scorer is trained_scorer
assert len(generator.meta_features) == len(sample_meta_features)
assert isinstance(generator.feature_success_rates, dict)

def test_generate_exploitation_hypotheses(
self,
trained_scorer: SurrogateScorer,
sample_meta_features: List[RunMetaFeatures],
):
"""Test Generierung von Exploitation-Hypothesen."""
generator = HypothesisGenerator(trained_scorer, sample_meta_features)
hypotheses = generator.generate_exploitation_hypotheses(top_k=3)

assert isinstance(hypotheses, list)
assert len(hypotheses) <= 3

for h in hypotheses:
assert isinstance(h, RunHypothesis)
assert h.hypothesis_type == "exploitation"
assert h.risk_level in ["low", "medium", "high"]

def test_generate_exploration_hypotheses(
self,
trained_scorer: SurrogateScorer,
sample_meta_features: List[RunMetaFeatures],
):
"""Test Generierung von Exploration-Hypothesen."""
generator = HypothesisGenerator(trained_scorer, sample_meta_features)
hypotheses = generator.generate_exploration_hypotheses(top_k=3)

assert isinstance(hypotheses, list)

for h in hypotheses:
assert h.hypothesis_type == "exploration"

def test_generate_pattern_based_hypotheses(
self,
trained_scorer: SurrogateScorer,
sample_meta_features: List[RunMetaFeatures],
):
"""Test Generierung von Pattern-based Hypothesen."""
generator = HypothesisGenerator(trained_scorer, sample_meta_features)
hypotheses = generator.generate_pattern_based_hypotheses(top_k=3)

assert isinstance(hypotheses, list)

for h in hypotheses:
assert h.hypothesis_type == "pattern_based"

def test_generate_all(
self,
trained_scorer: SurrogateScorer,
sample_meta_features: List[RunMetaFeatures],
):
"""Test Generierung aller Hypothesen."""
generator = HypothesisGenerator(trained_scorer, sample_meta_features)
hypotheses = generator.generate_all()

assert isinstance(hypotheses, list)

# Nach Confidence sortiert
for i in range(len(hypotheses) - 1):
assert hypotheses[i].confidence >= hypotheses[i + 1].confidence

def test_get_feature_success_rates(
self,
trained_scorer: SurrogateScorer,
sample_meta_features: List[RunMetaFeatures],
):
"""Test Feature-Erfolgsraten."""
generator = HypothesisGenerator(trained_scorer, sample_meta_features)
rates = generator.get_feature_success_rates()

assert isinstance(rates, dict)
assert all(isinstance(v, float) for v in rates.values())
assert all(0.0 <= v <= 1.0 for v in rates.values())

def test_get_diminishing_returns_features(
self,
trained_scorer: SurrogateScorer,
sample_meta_features: List[RunMetaFeatures],
):
"""Test Identifikation von Features mit sinkendem Grenznutzen."""
generator = HypothesisGenerator(trained_scorer, sample_meta_features)
diminishing = generator.get_diminishing_returns_features()

assert isinstance(diminishing, list)

def test_run_hypothesis_to_dict(self):
"""Test RunHypothesis Dictionary-Konvertierung."""
hypothesis = RunHypothesis(
features_proposed=["gqa", "film"],
predicted_delta_bpb=-0.015,
confidence=0.85,
hypothesis_type="exploitation",
reasoning="Test reasoning",
similar_successful_runs=["run_001", "run_002"],
risk_level="low",
)

data = hypothesis.to_dict()

assert data["features_proposed"] == ["gqa", "film"]
assert data["predicted_delta_bpb"] == -0.015
assert data["confidence"] == 0.85
assert data["hypothesis_type"] == "exploitation"


# ============================================================================
# Pareto Tracker Tests
# ============================================================================


class TestParetoTracker:
"""Tests für ParetoTracker."""

def test_init(self):
"""Test Initialisierung."""
tracker = ParetoTracker()
assert tracker.points == []
assert tracker.frontier_history == []

def test_add_run(self):
"""Test Hinzufügen eines Runs."""
tracker = ParetoTracker()
point = tracker.add_run(
run_id="test_run",
delta_bpb=-0.015,
efficiency_gain=5.2,
size_change=0.0,
)

assert len(tracker.points) == 1
assert point.run_id == "test_run"
assert point.delta_bpb == -0.015

def test_compute_pareto_frontier(self):
"""Test Pareto-Frontier Berechnung."""
tracker = ParetoTracker()

# Punkte hinzufügen
tracker.add_run("run_1", delta_bpb=-0.02, efficiency_gain=10.0, size_change=0.0)
tracker.add_run("run_2", delta_bpb=-0.01, efficiency_gain=5.0, size_change=-5.0)
tracker.add_run("run_3", delta_bpb=0.01, efficiency_gain=3.0, size_change=-10.0)

frontier = tracker.compute_pareto_frontier()

assert isinstance(frontier, list)
assert len(frontier) > 0

# Alle Punkte in Frontier sollten als optimal markiert sein
for point in frontier:
assert point.is_pareto_optimal

def test_get_dominated_points(self):
"""Test dominierte Punkte."""
tracker = ParetoTracker()

# Dominierenden Punkt hinzufügen
tracker.add_run("dominant", delta_bpb=-0.03, efficiency_gain=15.0, size_change=-5.0)
# Dominierten Punkt hinzufügen
tracker.add_run("dominated", delta_bpb=-0.01, efficiency_gain=5.0, size_change=0.0)

tracker.compute_pareto_frontier()
dominated = tracker.get_dominated_points()

dominated_ids = [p.run_id for p in dominated]
assert "dominated" in dominated_ids

def test_compute_frontier_volume(self):
"""Test Volumen-Berechnung."""
tracker = ParetoTracker()

# Leere Frontier
assert tracker.compute_frontier_volume() == 0.0

# Mit Punkten
tracker.add_run("run_1", delta_bpb=-0.02, efficiency_gain=10.0, size_change=0.0)
tracker.add_run("run_2", delta_bpb=-0.01, efficiency_gain=5.0, size_change=-5.0)

volume = tracker.compute_frontier_volume()
assert volume > 0.0

def test_get_frontier_expansion(self):
"""Test Frontier-Expansion."""
tracker = ParetoTracker()

# Keine History
assert tracker.get_frontier_expansion() == 0.0

# Snapshot erstellen
tracker.add_run("run_1", delta_bpb=-0.02, efficiency_gain=10.0, size_change=0.0)
tracker.snapshot()

# Mehr Punkte
tracker.add_run("run_2", delta_bpb=-0.03, efficiency_gain=12.0, size_change=-2.0)

expansion = tracker.get_frontier_expansion()
assert isinstance(expansion, float)

def test_identify_gaps(self):
"""Test Lücken-Identifikation."""
tracker = ParetoTracker()

tracker.add_run("run_1", delta_bpb=-0.02, efficiency_gain=10.0, size_change=0.0)
tracker.add_run("run_2", delta_bpb=-0.01, efficiency_gain=5.0, size_change=-5.0)

gaps = tracker.identify_gaps(num_gaps=2)

assert isinstance(gaps, list)

for gap in gaps:
assert "target_bpb" in gap
assert "target_efficiency" in gap
assert "reason" in gap

def test_get_statistics(self):
"""Test Statistik-Berechnung."""
tracker = ParetoTracker()

tracker.add_run("run_1", delta_bpb=-0.02, efficiency_gain=10.0, size_change=0.0)
tracker.add_run("run_2", delta_bpb=-0.01, efficiency_gain=5.0, size_change=-5.0)

stats = tracker.get_statistics()

assert "total_points" in stats
assert "pareto_optimal_count" in stats
assert "frontier_volume" in stats

def test_to_json(self):
"""Test JSON-Export."""
tracker = ParetoTracker()
tracker.add_run("run_1", delta_bpb=-0.02, efficiency_gain=10.0, size_change=0.0)

json_str = tracker.to_json()

assert isinstance(json_str, str)
data = json.loads(json_str)

assert "points" in data
assert "frontier" in data
assert "statistics" in data

def test_load_from_runs(self):
"""Test Laden von Runs aus Dictionary-Liste."""
tracker = ParetoTracker()

runs = [
{"run_id": "run_1", "delta_bpb": -0.02, "efficiency_gain": 10.0, "size_change": 0.0},
{"run_id": "run_2", "delta_bpb": -0.01, "efficiency_gain": 5.0, "size_change": -5.0},
]

count = tracker.load_from_runs(runs)

assert count == 2
assert len(tracker.points) == 2

def test_pareto_point_dominates(self):
"""Test ParetoPoint Dominanz-Logik."""
p1 = ParetoPoint(
run_id="p1",
delta_bpb=-0.03,
efficiency_gain=15.0,
size_change=-5.0,
is_pareto_optimal=True,
)
p2 = ParetoPoint(
run_id="p2",
delta_bpb=-0.01,
efficiency_gain=5.0,
size_change=0.0,
is_pareto_optimal=False,
)

assert p1.dominates(p2)
assert not p2.dominates(p1)


# ============================================================================
# Adaptive Kill Thresholds Tests
# ============================================================================


class TestAdaptiveKillThresholds:
"""Tests für AdaptiveKillThresholdManager."""

def test_init(self):
"""Test Initialisierung."""
manager = AdaptiveKillThresholdManager()

assert "low_budget" in manager.thresholds
assert "medium_budget" in manager.thresholds
assert "high_budget" in manager.thresholds

def test_get_thresholds(self):
"""Test Thresholds für Budget-Klasse."""
manager = AdaptiveKillThresholdManager()

thresholds = manager.get_thresholds("medium_budget")

assert isinstance(thresholds, KillThresholds)
assert thresholds.context == "medium_budget"

def test_get_thresholds_invalid(self):
"""Test Thresholds für ungültige Budget-Klasse."""
manager = AdaptiveKillThresholdManager()

with pytest.raises(ValueError, match="Ungültige Budget-Klasse"):
manager.get_thresholds("invalid_budget")

def test_should_kill_run_no_violations(self):
"""Test Run ohne Verletzungen."""
manager = AdaptiveKillThresholdManager()

features = RunMetaFeatures(
run_id="test_run",
features_active=["gqa"],
lineage_depth=1,
siblings_count=2,
budget_class="medium", # Korrekte Budget-Klasse
sequence_length="local",
quantization_type="none",
step_time_ms=10.0,
memory_usage_mb=1000.0,
training_stability=0.85,
delta_bpb_vs_parent=-0.015,
efficiency_gain_percent=5.0,
model_size_change_percent=0.0,
quant_gap=0.005,
)

should_kill, reason = manager.should_kill_run(features)

assert should_kill is False
assert "Alle Kill-Kriterien erfüllt" in reason

def test_should_kill_run_with_violations(self):
"""Test Run mit Verletzungen."""
manager = AdaptiveKillThresholdManager()

features = RunMetaFeatures(
run_id="test_run",
features_active=["gqa"],
lineage_depth=1,
siblings_count=2,
budget_class="low", # Korrekte Budget-Klasse (strengere Thresholds)
sequence_length="local",
quantization_type="none",
step_time_ms=10.0,
memory_usage_mb=1000.0,
training_stability=0.40, # Unter dem Limit (0.7)
delta_bpb_vs_parent=0.05, # Über dem Limit (0.02)
efficiency_gain_percent=5.0,
model_size_change_percent=0.0,
quant_gap=0.005,
)

should_kill, reason = manager.should_kill_run(features)

assert should_kill is True
assert "Kill-Kriterien verletzt" in reason

def test_adapt_thresholds_high_success(self):
"""Test Threshold-Anpassung bei hoher Erfolgsrate."""
manager = AdaptiveKillThresholdManager()

original_threshold = manager.thresholds["medium_budget"].max_delta_bpb

# Hohe Erfolgsrate (> 80%)
manager.adapt_thresholds(recent_success_rate=0.85)

new_threshold = manager.thresholds["medium_budget"].max_delta_bpb

# Thresholds sollten strenger geworden sein
assert new_threshold < original_threshold

def test_adapt_thresholds_low_success(self):
"""Test Threshold-Anpassung bei niedriger Erfolgsrate."""
manager = AdaptiveKillThresholdManager()

original_threshold = manager.thresholds["medium_budget"].max_delta_bpb

# Niedrige Erfolgsrate (< 30%)
manager.adapt_thresholds(recent_success_rate=0.20)

new_threshold = manager.thresholds["medium_budget"].max_delta_bpb

# Thresholds sollten lockerer geworden sein
assert new_threshold > original_threshold

def test_get_feature_specific_thresholds(self):
"""Test Feature-spezifische Thresholds."""
manager = AdaptiveKillThresholdManager()

# Quant-Feature
quant_thresholds = manager.get_feature_specific_thresholds("mixed_quant")
assert quant_thresholds.max_quant_gap > 0.02 # Mehr Toleranz

# Speed-Feature
speed_thresholds = manager.get_feature_specific_thresholds("flash_attn")
assert speed_thresholds.max_step_time_increase <= 10.0 # Streng bei Speed

def test_set_feature_threshold(self):
"""Test Setzen von Feature-spezifischen Thresholds."""
manager = AdaptiveKillThresholdManager()

custom_thresholds = KillThresholds(
context="custom",
max_delta_bpb=0.15,
min_efficiency_gain=-15.0,
max_quant_gap=0.08,
max_step_time_increase=50.0,
max_memory_increase=60.0,
min_training_stability=0.4,
)

manager.set_feature_threshold("custom_feature", custom_thresholds)

retrieved = manager.get_feature_specific_thresholds("custom_feature")
assert retrieved.max_delta_bpb == 0.15

def test_get_kill_statistics(self):
"""Test Kill-Statistiken."""
manager = AdaptiveKillThresholdManager()

# Einige Entscheidungen simulieren
features_good = RunMetaFeatures(
run_id="good_run",
features_active=["gqa"],
lineage_depth=1,
siblings_count=2,
budget_class="medium", # Korrekte Budget-Klasse
sequence_length="local",
quantization_type="none",
training_stability=0.85,
delta_bpb_vs_parent=-0.015,
efficiency_gain_percent=5.0,
)

manager.should_kill_run(features_good)

stats = manager.get_kill_statistics()

assert "total_decisions" in stats
assert "kills" in stats
assert "kill_rate" in stats

def test_reset(self):
"""Test Zurücksetzen des Managers."""
manager = AdaptiveKillThresholdManager()

# Änderungen vornehmen
manager.adapt_thresholds(0.20)

# Zurücksetzen
manager.reset()

# Sollte wieder sein
assert len(manager.recent_decisions) == 0
assert len(manager.success_rate_history) == 0

def test_kill_thresholds_to_dict(self):
"""Test KillThresholds Dictionary-Konvertierung."""
thresholds = KillThresholds(
context="test",
max_delta_bpb=0.05,
min_efficiency_gain=-5.0,
max_quant_gap=0.02,
max_step_time_increase=20.0,
max_memory_increase=30.0,
min_training_stability=0.6,
)

data = thresholds.to_dict()

assert data["context"] == "test"
assert data["max_delta_bpb"] == 0.05
assert data["min_efficiency_gain"] == -5.0

def test_kill_thresholds_is_stricter_than(self):
"""Test Vergleich von Thresholds."""
strict = KillThresholds(
context="strict",
max_delta_bpb=0.01,
min_efficiency_gain=-1.0,
max_quant_gap=0.005,
max_step_time_increase=5.0,
max_memory_increase=10.0,
min_training_stability=0.8,
)

loose = KillThresholds(
context="loose",
max_delta_bpb=0.10,
min_efficiency_gain=-10.0,
max_quant_gap=0.05,
max_step_time_increase=50.0,
max_memory_increase=100.0,
min_training_stability=0.3,
)

assert strict.is_stricter_than(loose)
assert not loose.is_stricter_than(strict)


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhase4AIntegration:
"""Integrationstests für Phase 4A Module."""

def test_full_workflow(self, sample_meta_features: List[RunMetaFeatures]):
"""Test kompletter Workflow von Training bis Empfehlung."""
# 1. Surrogate Scorer trainieren
scorer = SurrogateScorer(model_type="random_forest")

train_features = [
f for f in sample_meta_features
if f.delta_bpb_vs_parent is not None and f.efficiency_gain_percent is not None
]

targets = {
"delta_bpb": [f.delta_bpb_vs_parent for f in train_features],
"efficiency_gain": [f.efficiency_gain_percent for f in train_features],
}

scorer.train(train_features, targets)

# 2. Hypothesen generieren
generator = HypothesisGenerator(scorer, sample_meta_features)
hypotheses = generator.generate_all()

assert len(hypotheses) > 0

# 3. Pareto-Frontier analysieren
tracker = ParetoTracker()

for f in sample_meta_features:
if f.delta_bpb_vs_parent is not None:
tracker.add_run(
run_id=f.run_id,
delta_bpb=f.delta_bpb_vs_parent,
efficiency_gain=f.efficiency_gain_percent or 0.0,
size_change=f.model_size_change_percent or 0.0,
)

frontier = tracker.get_frontier_points()
assert len(frontier) > 0

# 4. Kill-Thresholds evaluieren
kill_manager = AdaptiveKillThresholdManager()

for f in sample_meta_features[:3]:
should_kill, reason = kill_manager.should_kill_run(f)
assert isinstance(should_kill, bool)

# 5. Feature-Importance extrahieren
importance = scorer.get_feature_importance()
assert len(importance) > 0

# 6. Feature-Erfolgsraten
success_rates = generator.get_feature_success_rates()
assert len(success_rates) > 0


if __name__ == "__main__":
pytest.main([__file__, "-v"])
