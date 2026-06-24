#!/usr/bin/env python3
"""
Meta-Feature Dashboard für NeuroWeave.

Dieses Dashboard bietet Einblicke in die extrahierten Meta-Features
aller Runs und hilft bei der Analyse von Feature-Performance und
Co-occurrence Mustern.

Phase 3 Commands:
python -m orchestrator.meta_dashboard summary # Übersicht aller Features
python -m orchestrator.meta_dashboard co-occurrence # Feature Co-occurrence Matrix
python -m orchestrator.meta_dashboard feature-stats # Statistiken pro Feature
python -m orchestrator.meta_dashboard lineage # Lineage-Analyse
python -m orchestrator.meta_dashboard budget # Budget-Klassen Analyse
python -m orchestrator.meta_dashboard quant # Quantisierungs-Analyse

Phase 4A Commands (neu):
python -m orchestrator.meta_dashboard predictions # Surrogate Scorer Vorhersagen
python -m orchestrator.meta_dashboard hypotheses # Run-Vorschläge generieren
python -m orchestrator.meta_dashboard pareto # Pareto-Frontier anzeigen
python -m orchestrator.meta_dashboard recommendations # Top-Empfehlungen

Phase 4B Commands (neu):
python -m orchestrator.meta_dashboard anomalies # Anomalien anzeigen
python -m orchestrator.meta_dashboard failures # Fehler-Analyse
python -m orchestrator.meta_dashboard quarantine # Quarantäne-Liste
python -m orchestrator.meta_dashboard drift # Drift-Reports
python -m orchestrator.meta_dashboard recovery # Recovery-Empfehlungen

Phase 4C Commands (neu):
python -m orchestrator.meta_dashboard guardrails # Guardrail-Status
python -m orchestrator.meta_dashboard approvals # Ausstehende Freigaben
python -m orchestrator.meta_dashboard alerts # Alert-Historie
python -m orchestrator.meta_dashboard autonomy-stats # Autonomie-Statistiken
python -m orchestrator.meta_dashboard overrides # Override-Analyse

Beispiele:
python -m orchestrator.meta_dashboard summary --top 10
python -m orchestrator.meta_dashboard feature-stats --min-count 3
python -m orchestrator.meta_dashboard co-occurrence --limit 20
python -m orchestrator.meta_dashboard predictions
python -m orchestrator.meta_dashboard hypotheses --top 10
python -m orchestrator.meta_dashboard pareto --plot
python -m orchestrator.meta_dashboard recommendations --top 5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.meta_features import MetaFeatureExtractor, RunMetaFeatures
from core.registry import RunRegistry
from research.surrogate_scorer import SurrogateScorer
from research.hypothesis_generator import HypothesisGenerator, RunHypothesis
from research.pareto_tracker import ParetoTracker, ParetoPoint
from research.adaptive_kill_thresholds import AdaptiveKillThresholdManager
from research.anomaly_detector import AnomalyDetector, AnomalyReport
from research.failure_classifier import FailureClassifier, FailureDiagnosis
from research.drift_monitor import DriftMonitor, DriftReport
from orchestrator.rollback_manager import RollbackManager, RollbackPlan
from orchestrator.run_quarantine import RunQuarantineManager, QuarantineEntry
from orchestrator.guardrails import create_default_guardrails, GuardrailManager, AutonomyLevel
from orchestrator.autonomy_orchestrator import create_autonomy_orchestrator, AutonomyOrchestrator
from orchestrator.approval_interface import create_approval_interface, ApprovalInterface
from core.alerting import create_alert_manager, AlertManager, AlertSeverity
from research.override_learner import create_override_learner, OverrideLearner

# Phase 4 Evaluation Imports
from research.ab_testing import ABTestFramework, ABTestConfig
from research.success_metrics import SuccessMetricsTracker
from research.refinement_engine import RefinementEngine


def load_features_from_registry(registry: RunRegistry, extractor: MetaFeatureExtractor) -> List[RunMetaFeatures]:
"""Lade Meta-Features direkt aus dem Registry."""
run_ids = [run.run_id for run in registry.list_runs()]
return extractor.extract_batch(run_ids, registry)


def load_features_from_json(path: str) -> Optional[List[RunMetaFeatures]]:
"""Lade Meta-Features aus JSON-Datei."""
json_path = Path(path)
if not json_path.exists():
return None

with open(json_path, "r", encoding="utf-8") as f:
data = json.load(f)

return [RunMetaFeatures.from_dict(item) for item in data]


def get_features(registry: RunRegistry, extractor: MetaFeatureExtractor, json_path: Optional[str] = None) -> List[RunMetaFeatures]:
"""
Lade Meta-Features aus JSON oder Registry.

Args:
registry: RunRegistry Instanz
extractor: MetaFeatureExtractor Instanz
json_path: Optionaler Pfad zu JSON-Datei

Returns:
Liste von RunMetaFeatures Objekten
"""
if json_path:
features = load_features_from_json(json_path)
if features:
return features
print(f"Warnung: JSON-Datei '{json_path}' nicht gefunden, verwende Registry...")

return load_features_from_registry(registry, extractor)


def print_summary(features: List[RunMetaFeatures], top_n: int = 10) -> None:
"""
Drucke Zusammenfassung der Meta-Features.

Args:
features: Liste von RunMetaFeatures
top_n: Anzahl der Top-Features für Feature-Liste
"""
if not features:
print("Keine Features vorhanden.")
return

print("\n" + "=" * 70)
print("META-FEATURE DASHBOARD - ZUSAMMENFASSUNG")
print("=" * 70)

# Grundlegende Statistiken
print(f"\n Gesamtübersicht:")
print(f" Total Runs: {len(features)}")

# Unique Features
all_features = set()
for f in features:
all_features.update(f.features_active)
print(f" Unique Features: {len(all_features)}")

# Top Features nach Häufigkeit
feature_counts = defaultdict(int)
for f in features:
for feat in f.features_active:
feature_counts[feat] += 1

if feature_counts:
sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])[:top_n]
print(f"\n Top {top_n} Features nach Häufigkeit:")
for feat, count in sorted_features:
pct = 100 * count / len(features)
print(f" • {feat}: {count} ({pct:.1f}%)")

# Budget Class Distribution
budget_dist = defaultdict(int)
for f in features:
budget_dist[f.budget_class] += 1

print(f"\n Budget-Klassen:")
for budget_class in ["low", "medium", "high"]:
count = budget_dist.get(budget_class, 0)
pct = 100 * count / len(features) if features else 0
bar = "" * int(pct / 5)
print(f" {budget_class.upper():8}: {count:3} ({pct:5.1f}%) {bar}")

# Quantization Distribution
quant_dist = defaultdict(int)
for f in features:
quant_dist[f.quantization_type] += 1

print(f"\n Quantisierungs-Typen:")
for quant_type in ["none", "int6", "int5", "mixed", "gptq_lite"]:
count = quant_dist.get(quant_type, 0)
pct = 100 * count / len(features) if features else 0
bar = "" * int(pct / 5)
print(f" {quant_type:8}: {count:3} ({pct:5.1f}%) {bar}")

# Sequence Length Distribution
seq_dist = defaultdict(int)
for f in features:
seq_dist[f.sequence_length] += 1

print(f"\n Sequenzlängen:")
for seq_len in ["local", "remote"]:
count = seq_dist.get(seq_len, 0)
pct = 100 * count / len(features) if features else 0
bar = "" * int(pct / 5)
print(f" {seq_len:8}: {count:3} ({pct:5.1f}%) {bar}")

# Lineage Statistics
depths = [f.lineage_depth for f in features]
siblings = [f.siblings_count for f in features]

if depths:
print(f"\n Lineage-Statistiken:")
print(f" Depth: min={min(depths):2}, max={max(depths):2}, avg={sum(depths) / len(depths):5.2f}")
print(f" Siblings: min={min(siblings):2}, max={max(siblings):2}, avg={sum(siblings) / len(siblings):5.2f}")

# Performance Statistics
completed_with_parent = [f for f in features if f.delta_bpb_vs_parent != 0.0 or f.parent_run_id is not None]

if completed_with_parent:
deltas = [f.delta_bpb_vs_parent for f in completed_with_parent if f.delta_bpb_vs_parent != 0.0]
if deltas:
print(f"\n Performance (ΔBPB vs Parent):")
print(f" Count: {len(deltas)} Runs mit Parent-Vergleich")
print(f" min: {min(deltas):+.4f} (beste Verbesserung)")
print(f" max: {max(deltas):+.4f} (schlechteste Veränderung)")
print(f" avg: {sum(deltas) / len(deltas):+.4f}")

# Zähle Verbesserungen vs Verschlechterungen
improvements = sum(1 for d in deltas if d < 0)
degradations = sum(1 for d in deltas if d > 0)
print(f" Verbesserungen: {improvements} ({100 * improvements / len(deltas):.1f}%)")
print(f" Verschlechterungen: {degradations} ({100 * degradations / len(deltas):.1f}%)")

# Effizienz-Statistiken
efficiency_gains = [f.efficiency_gain_percent for f in features if f.efficiency_gain_percent != 0.0]
if efficiency_gains:
print(f"\n Effizienz-Gewinn (%):")
print(f" min: {min(efficiency_gains):+.1f}%, max: {max(efficiency_gains):+.1f}%, avg: {sum(efficiency_gains) / len(efficiency_gains):+.1f}%")

print("\n" + "=" * 70)


def print_co_occurrence(features: List[RunMetaFeatures], limit: int = 20, min_count: int = 1) -> None:
"""
Drucke Feature Co-occurrence Matrix.

Args:
features: Liste von RunMetaFeatures
limit: Maximale Anzahl anzuzeigender Paare
min_count: Minimale Co-occurrence für Anzeige
"""
if not features:
print("Keine Features vorhanden.")
return

extractor = MetaFeatureExtractor()
co_occ = extractor.compute_co_occurrence(features)

# Filtere nach min_count
filtered = {k: v for k, v in co_occ.items() if v >= min_count}

if not filtered:
print(f"Keine Co-occurrences mit min_count={min_count} gefunden.")
return

print("\n" + "=" * 70)
print(f"FEATURE CO-OCCURRENCE (Top {limit})")
print("=" * 70)

# Sortiere nach Count
sorted_co_occ = sorted(filtered.items(), key=lambda x: -x[1])[:limit]

print(f"\n{'Feature 1':<25} {'Feature 2':<25} {'Count':>8}")
print("-" * 60)

for (f1, f2), count in sorted_co_occ:
print(f"{f1:<25} {f2:<25} {count:>8}")

# Heatmap-ähnliche Darstellung für häufigste Features
all_features = set()
for f in features:
all_features.update(f.features_active)

if len(all_features) <= 15:
print("\n" + "=" * 70)
print("CO-OCCURRENCE MATRIX (Heatmap)")
print("=" * 70)

sorted_features = sorted(all_features)

# Header
header = f"{'':<20}"
for f in sorted_features[:12]: # Max 12 für Lesbarkeit
header += f"{f[:8]:>8}"
print(header)
print("-" * len(header))

# Rows
for f1 in sorted_features[:12]:
row = f"{f1:<20}"
for f2 in sorted_features[:12]:
if f1 == f2:
row += f"{'-':>8}"
else:
count = co_occ.get((min(f1, f2), max(f1, f2)), 0)
if count > 0:
row += f"{count:>8}"
else:
row += f"{'·':>8}"
print(row)

print("=" * 70)


def print_feature_stats(features: List[RunMetaFeatures], min_count: int = 2, sort_by: str = "count") -> None:
"""
Drucke Statistiken pro Feature.

Args:
features: Liste von RunMetaFeatures
min_count: Minimale Anzahl von Runs für Statistik
sort_by: Sortierreihenfolge ("count", "avg_delta", "success_rate")
"""
if not features:
print("Keine Features vorhanden.")
return

# Sammle Outcomes pro Feature
feature_outcomes: Dict[str, List[float]] = defaultdict(list)
feature_efficiency: Dict[str, List[float]] = defaultdict(list)
feature_stability: Dict[str, List[float]] = defaultdict(list)

for f in features:
for feat in f.features_active:
if f.delta_bpb_vs_parent != 0.0:
feature_outcomes[feat].append(f.delta_bpb_vs_parent)
if f.efficiency_gain_percent != 0.0:
feature_efficiency[feat].append(f.efficiency_gain_percent)
feature_stability[feat].append(f.training_stability)

# Berechne Statistiken
stats = []
for feat in set(feature_outcomes.keys()) | set(feature_efficiency.keys()):
outcomes = feature_outcomes.get(feat, [])
efficiency = feature_efficiency.get(feat, [])
stability = feature_stability.get(feat, [])

count = len(outcomes) if outcomes else len(stability)

if count < min_count:
continue

avg_delta = sum(outcomes) / len(outcomes) if outcomes else 0.0
avg_efficiency = sum(efficiency) / len(efficiency) if efficiency else 0.0
avg_stability = sum(stability) / len(stability) if stability else 0.0

# Erfolgsquote (ΔBPB < 0 = Verbesserung)
success_rate = sum(1 for d in outcomes if d < 0) / len(outcomes) if outcomes else 0.0

stats.append({
"feature": feat,
"count": count,
"avg_delta": avg_delta,
"avg_efficiency": avg_efficiency,
"avg_stability": avg_stability,
"success_rate": success_rate,
})

if not stats:
print(f"Keine Features mit min_count={min_count} gefunden.")
return

# Sortiere
if sort_by == "count":
stats.sort(key=lambda x: -x["count"])
elif sort_by == "avg_delta":
stats.sort(key=lambda x: x["avg_delta"]) # Niedriger = besser
elif sort_by == "success_rate":
stats.sort(key=lambda x: -x["success_rate"])

print("\n" + "=" * 80)
print(f"FEATURE STATISTIKEN (min_count={min_count}, sortiert nach {sort_by})")
print("=" * 80)

print(f"\n{'Feature':<25} {'Count':>6} {'Avg ΔBPB':>10} {'Avg Eff%':>10} {'Stability':>10} {'Success%':>10}")
print("-" * 80)

for s in stats:
delta_str = f"{s['avg_delta']:+.4f}"
eff_str = f"{s['avg_efficiency']:+.1f}"
stab_str = f"{s['avg_stability']:.2f}"
succ_str = f"{100 * s['success_rate']:.1f}%"

# Farbliche Markierung (simuliert mit Symbolen)
delta_symbol = "↓" if s["avg_delta"] < 0 else "↑" if s["avg_delta"] > 0 else "•"

print(f"{s['feature']:<25} {s['count']:>6} {delta_str:>10} {eff_str:>10} {stab_str:>10} {succ_str:>10} {delta_symbol}")

print("=" * 80)


def print_lineage_analysis(features: List[RunMetaFeatures]) -> None:
"""
Drucke Lineage-Analyse.

Args:
features: Liste von RunMetaFeatures
"""
if not features:
print("Keine Features vorhanden.")
return

# Gruppiere nach Parent
parent_groups: Dict[Optional[str], List[RunMetaFeatures]] = defaultdict(list)
for f in features:
parent_groups[f.parent_run_id].append(f)

# Finde Root-Runs (ohne Parent)
root_runs = parent_groups.get(None, [])

print("\n" + "=" * 70)
print("LINEAGE ANALYSE")
print("=" * 70)

print(f"\n Übersicht:")
print(f" Root Runs (ohne Parent): {len(root_runs)}")
print(f" Runs mit Parent: {len(features) - len(root_runs)}")

# Lineage-Tiefe Verteilung
depth_dist = defaultdict(int)
for f in features:
depth_dist[f.lineage_depth] += 1

print(f"\n Lineage-Tiefe Verteilung:")
for depth in sorted(depth_dist.keys()):
count = depth_dist[depth]
bar = "" * count
print(f" Depth {depth}: {count:3} {bar}")

# Größte Familien
family_sizes = [(parent_id, len(runs)) for parent_id, runs in parent_groups.items() if parent_id is not None]
family_sizes.sort(key=lambda x: -x[1])

if family_sizes:
print(f"\n Top 5 Parent-Familien (nach Children-Anzahl):")
for parent_id, size in family_sizes[:5]:
print(f" {parent_id}: {size} Children")

# Siblings-Statistiken
siblings_counts = [f.siblings_count for f in features if f.siblings_count > 0]
if siblings_counts:
print(f"\n Siblings-Statistiken:")
print(f" Runs mit Siblings: {len(siblings_counts)}")
print(f" Max Siblings: {max(siblings_counts)}")
print(f" Avg Siblings: {sum(siblings_counts) / len(siblings_counts):.2f}")

print("=" * 70)


def print_budget_analysis(features: List[RunMetaFeatures]) -> None:
"""
Drucke Budget-Klassen Analyse.

Args:
features: Liste von RunMetaFeatures
"""
if not features:
print("Keine Features vorhanden.")
return

# Gruppiere nach Budget-Klasse
budget_groups: Dict[str, List[RunMetaFeatures]] = defaultdict(list)
for f in features:
budget_groups[f.budget_class].append(f)

print("\n" + "=" * 70)
print("BUDGET-KLASSEN ANALYSE")
print("=" * 70)

for budget_class in ["low", "medium", "high"]:
runs = budget_groups.get(budget_class, [])

if not runs:
continue

print(f"\n Budget-Klasse: {budget_class.upper()}")
print(f" Runs: {len(runs)}")

# Feature-Verteilung
feature_counts = defaultdict(int)
for f in runs:
for feat in f.features_active:
feature_counts[feat] += 1

if feature_counts:
print(f" Top Features:")
sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])[:5]
for feat, count in sorted_features:
pct = 100 * count / len(runs)
print(f" • {feat}: {count} ({pct:.1f}%)")

# Performance
deltas = [f.delta_bpb_vs_parent for f in runs if f.delta_bpb_vs_parent != 0.0]
if deltas:
avg_delta = sum(deltas) / len(deltas)
improvements = sum(1 for d in deltas if d < 0)
print(f" Performance:")
print(f" Avg ΔBPB: {avg_delta:+.4f}")
print(f" Verbesserungen: {improvements}/{len(deltas)} ({100 * improvements / len(deltas):.1f}%)")

print("=" * 70)


def print_quant_analysis(features: List[RunMetaFeatures]) -> None:
"""
Drucke Quantisierungs-Analyse.

Args:
features: Liste von RunMetaFeatures
"""
if not features:
print("Keine Features vorhanden.")
return

# Gruppiere nach Quantisierungs-Typ
quant_groups: Dict[str, List[RunMetaFeatures]] = defaultdict(list)
for f in features:
quant_groups[f.quantization_type].append(f)

print("\n" + "=" * 70)
print("QUANTISIERUNGS ANALYSE")
print("=" * 70)

for quant_type in ["none", "int6", "int5", "mixed", "gptq_lite"]:
runs = quant_groups.get(quant_type, [])

if not runs:
continue

print(f"\n Quantisierung: {quant_type.upper()}")
print(f" Runs: {len(runs)}")

# Quant-Gap Statistiken
gaps = [f.quant_gap for f in runs if f.quant_gap != 0.0]
if gaps:
print(f" Quant-Gap (BPB Degradation):")
print(f" min: {min(gaps):.4f}")
print(f" max: {max(gaps):.4f}")
print(f" avg: {sum(gaps) / len(gaps):.4f}")

# Feature-Verteilung
feature_counts = defaultdict(int)
for f in runs:
for feat in f.features_active:
feature_counts[feat] += 1

if feature_counts:
print(f" Häufige Features:")
sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])[:5]
for feat, count in sorted_features:
pct = 100 * count / len(runs)
print(f" • {feat}: {count} ({pct:.1f}%)")

print("=" * 70)


def print_predictions(features: List[RunMetaFeatures], top_n: int = 10) -> None:
"""
Drucke Surrogate Scorer Vorhersagen.

Args:
features: Liste von RunMetaFeatures
top_n: Anzahl der Top-Vorhersagen
"""
if not features:
print("Keine Features vorhanden.")
return

print("\n" + "=" * 70)
print("SURROGATE SCORER VORHERSAGEN")
print("=" * 70)

# Surrogate Scorer initialisieren und trainieren
scorer = SurrogateScorer(model_type="random_forest")

# Trainingsdaten vorbereiten
targets = {
"delta_bpb": [f.delta_bpb_vs_parent for f in features if f.delta_bpb_vs_parent is not None],
"efficiency_gain": [f.efficiency_gain_percent for f in features if f.efficiency_gain_percent is not None],
}

# Nur Features mit Targets verwenden
train_features = [
f for f in features
if f.delta_bpb_vs_parent is not None and f.efficiency_gain_percent is not None
]

if len(train_features) < 5:
print(f" Nicht genügend Trainingsdaten ({len(train_features)} < 5)")
print(" Führe mehr Runs durch um den Surrogate Scorer zu trainieren.")
return

# Modell trainieren
try:
metrics = scorer.train(train_features, {
"delta_bpb": targets["delta_bpb"][:len(train_features)],
"efficiency_gain": targets["efficiency_gain"][:len(train_features)],
})

print(f"\n Modell-Statistiken:")
print(f" Modell-Typ: Random Forest")
print(f" Trainings-Runs: {len(train_features)}")
print(f" BPB CV-RMSE: {metrics.get('bpb_cv_rmse', 0):.4f}")
print(f" Efficiency CV-RMSE: {metrics.get('efficiency_cv_rmse', 0):.4f}")

# Feature-Importance anzeigen
importance = scorer.get_feature_importance()
sorted_importance = sorted(importance.items(), key=lambda x: -x[1])[:10]

print(f"\n Top 10 Feature-Importancen:")
for feat, imp in sorted_importance:
bar = "" * int(imp * 20)
print(f" {feat:<30} {imp:.4f} {bar}")

# Vorhersagen für alle Runs
print(f"\n Vorhersagen (Top {top_n}):")
print(f"{'Run ID':<40} {'Pred ΔBPB':>12} {'Pred Eff%':>12} {'Confidence':>12}")
print("-" * 78)

predictions = []
for f in features:
try:
pred_bpb, pred_eff, conf = scorer.predict(f)
predictions.append((f.run_id, pred_bpb, pred_eff, conf))
except Exception:
continue

# Nach Confidence sortieren
predictions.sort(key=lambda x: -x[3])

for run_id, pred_bpb, pred_eff, conf in predictions[:top_n]:
bpb_str = f"{pred_bpb:+.4f}"
eff_str = f"{pred_eff:+.1f}"
conf_str = f"{conf:.1%}"
symbol = "" if pred_bpb < 0 else ""
print(f"{run_id:<40} {bpb_str:>12} {eff_str:>12} {conf_str:>12} {symbol}")

except Exception as e:
print(f" Fehler beim Trainieren: {e}")

print("=" * 70)


def print_hypotheses(features: List[RunMetaFeatures], top_n: int = 10) -> None:
"""
Drucke generierte Run-Hypothesen.

Args:
features: Liste von RunMetaFeatures
top_n: Anzahl der Top-Hypothesen
"""
if not features:
print("Keine Features vorhanden.")
return

print("\n" + "=" * 80)
print("HYPOTHESIS GENERATOR - RUN VORSCHLÄGE")
print("=" * 80)

# Surrogate Scorer trainieren
scorer = SurrogateScorer(model_type="random_forest")

train_features = [
f for f in features
if f.delta_bpb_vs_parent is not None and f.efficiency_gain_percent is not None
]

if len(train_features) < 5:
print(f" Nicht genügend Trainingsdaten ({len(train_features)} < 5)")
return

targets = {
"delta_bpb": [f.delta_bpb_vs_parent for f in train_features],
"efficiency_gain": [f.efficiency_gain_percent for f in train_features],
}

try:
scorer.train(train_features, targets)
except Exception as e:
print(f" Fehler beim Trainieren: {e}")
return

# Hypothesis Generator initialisieren
generator = HypothesisGenerator(scorer, features)

# Alle Hypothesen generieren
hypotheses = generator.generate_all()

if not hypotheses:
print(" Keine Hypothesen generiert.")
return

print(f"\n Übersicht:")
print(f" Total Hypothesen: {len(hypotheses)}")

exploitation_count = sum(1 for h in hypotheses if h.hypothesis_type == "exploitation")
exploration_count = sum(1 for h in hypotheses if h.hypothesis_type == "exploration")
pattern_count = sum(1 for h in hypotheses if h.hypothesis_type == "pattern_based")

print(f" Exploitation: {exploitation_count}")
print(f" Exploration: {exploration_count}")
print(f" Pattern-based: {pattern_count}")

# Feature-Erfolgsraten
success_rates = generator.get_feature_success_rates()
if success_rates:
print(f"\n Top Feature-Erfolgsraten:")
sorted_rates = sorted(success_rates.items(), key=lambda x: -x[1])[:5]
for feat, rate in sorted_rates:
count = generator.feature_counts.get(feat, 0)
print(f" {feat:<25} {rate:.1%} ({count} Runs)")

# Top-Hypothesen anzeigen
print(f"\n Top {top_n} Hypothesen:")
print("-" * 80)

for i, h in enumerate(hypotheses[:top_n], 1):
risk_symbol = {"low": "🟢", "medium": "🟡", "high": ""}.get(h.risk_level, "")
type_symbol = {
"exploitation": "",
"exploration": "",
"pattern_based": ""
}.get(h.hypothesis_type, "•")

print(f"\n{i:2}. {type_symbol} {', '.join(h.features_proposed)}")
print(f" Vorhergesagtes ΔBPB: {h.predicted_delta_bpb:+.4f}")
print(f" Confidence: {h.confidence:.1%}")
print(f" Risiko: {h.risk_level} {risk_symbol}")
print(f" Begründung: {h.reasoning}")

if h.similar_successful_runs:
similar_str = ", ".join(h.similar_successful_runs[:3])
print(f" Ähnliche Runs: {similar_str}")

print("\n" + "=" * 80)


def print_pareto_frontier(features: List[RunMetaFeatures], plot: bool = False) -> None:
"""
Drucke Pareto-Frontier Analyse.

Args:
features: Liste von RunMetaFeatures
plot: Ob Plot erstellt werden soll
"""
if not features:
print("Keine Features vorhanden.")
return

print("\n" + "=" * 70)
print("PARETO FRONTIER ANALYSE")
print("=" * 70)

# Pareto Tracker initialisieren
tracker = ParetoTracker()

# Runs hinzufügen (nur mit vollständigen Daten)
for f in features:
if f.delta_bpb_vs_parent is not None and f.efficiency_gain_percent is not None:
size_change = f.model_size_change_percent or 0.0
tracker.add_run(
run_id=f.run_id,
delta_bpb=f.delta_bpb_vs_parent,
efficiency_gain=f.efficiency_gain_percent,
size_change=size_change,
)

# Frontier berechnen
frontier = tracker.get_frontier_points()
dominated = tracker.get_dominated_points()

print(f"\n Übersicht:")
print(f" Total Runs: {len(tracker.points)}")
print(f" Pareto-optimal: {len(frontier)}")
print(f" Dominiert: {len(dominated)}")

# Statistik
stats = tracker.get_statistics()
print(f"\n Frontier-Statistiken:")
print(f" Volumen: {stats.get('frontier_volume', 0):.2f}")
print(f" Bestes ΔBPB: {stats.get('best_delta_bpb', 0):+.4f}")
print(f" Beste Effizienz: {stats.get('best_efficiency_gain', 0):+.1f}%")
print(f" Frontier Expansion: {stats.get('frontier_expansion', 0):+.1%}")

# Pareto-optimale Runs anzeigen
if frontier:
print(f"\n Pareto-optimale Runs:")
print(f"{'Run ID':<40} {'ΔBPB':>10} {'Eff%':>10} {'Size%':>10}")
print("-" * 72)

# Sortiert nach ΔBPB
sorted_frontier = sorted(frontier, key=lambda p: p.delta_bpb)
for p in sorted_frontier:
bpb_str = f"{p.delta_bpb:+.4f}"
eff_str = f"{p.efficiency_gain:+.1f}"
size_str = f"{p.size_change:+.1f}"
print(f"{p.run_id:<40} {bpb_str:>10} {eff_str:>10} {size_str:>10}")

# Lücken identifizieren
gaps = tracker.identify_gaps(num_gaps=3)
if gaps:
print(f"\n Identifizierte Lücken:")
for i, gap in enumerate(gaps, 1):
print(f" {i}. Target ΔBPB: {gap['target_bpb']:+.4f}, "
f"Effizienz: {gap['target_efficiency']:+.1f}%")
print(f" {gap['reason'][:80]}...")

# Plot erstellen
if plot:
try:
output_path = tracker.plot_frontier(output_path="results/pareto_frontier.png")
print(f"\n Plot erstellt: {output_path}")
except ImportError as e:
print(f"\n Plotting nicht verfügbar: {e}")
except Exception as e:
print(f"\n Fehler beim Plotten: {e}")

print("=" * 70)


def print_recommendations(features: List[RunMetaFeatures], top_n: int = 5) -> None:
"""
Drucke Top-Empfehlungen für nächste Runs.

Args:
features: Liste von RunMetaFeatures
top_n: Anzahl der Top-Empfehlungen
"""
if not features:
print("Keine Features vorhanden.")
return

print("\n" + "=" * 80)
print("TOP EMPFEHLUNGEN FÜR NÄCHSTE RUNS")
print("=" * 80)

# Surrogate Scorer trainieren
scorer = SurrogateScorer(model_type="gradient_boosting")

train_features = [
f for f in features
if f.delta_bpb_vs_parent is not None and f.efficiency_gain_percent is not None
]

if len(train_features) < 5:
print(f" Nicht genügend Trainingsdaten ({len(train_features)} < 5)")
return

targets = {
"delta_bpb": [f.delta_bpb_vs_parent for f in train_features],
"efficiency_gain": [f.efficiency_gain_percent for f in train_features],
}

try:
scorer.train(train_features, targets)
except Exception as e:
print(f" Fehler beim Trainieren: {e}")
return

# Hypothesis Generator
generator = HypothesisGenerator(scorer, features)
hypotheses = generator.generate_all()

# Adaptive Kill Thresholds
kill_manager = AdaptiveKillThresholdManager()
kill_stats = kill_manager.get_kill_statistics()

print(f"\n Analyse-Basis:")
print(f" Trainings-Runs: {len(train_features)}")
print(f" Generierte Hypothesen: {len(hypotheses)}")
print(f" Kill-Rate (recent): {kill_stats.get('kill_rate', 0):.1%}")

# Top-Empfehlungen
if hypotheses:
print(f"\n Top {top_n} Empfehlungen:")
print("-" * 80)

for i, h in enumerate(hypotheses[:top_n], 1):
priority = "" if i <= 2 else "" if i <= 5 else "•"
risk_symbol = {"low": "🟢", "medium": "🟡", "high": ""}.get(h.risk_level, "")

print(f"\n{i}. {priority} {', '.join(h.features_proposed)}")
print(f" Typ: {h.hypothesis_type} | Risiko: {h.risk_level} {risk_symbol}")
print(f" Erwartetes ΔBPB: {h.predicted_delta_bpb:+.4f} (Confidence: {h.confidence:.1%})")
print(f" Begründung: {h.reasoning[:100]}...")

# Diminishing Returns Features
diminishing = generator.get_diminishing_returns_features()
if diminishing:
print(f"\n Features mit sinkendem Grenznutzen:")
for feat in diminishing[:5]:
rate = generator.feature_success_rates.get(feat, 0)
count = generator.feature_counts.get(feat, 0)
print(f" • {feat}: {rate:.1%} Erfolgsrate in {count} Runs")

# Kill-Threshold Empfehlungen
print(f"\n Kill-Threshold Empfehlungen:")
for budget_class in ["low_budget", "medium_budget", "high_budget"]:
thresh = kill_manager.get_thresholds(budget_class)
print(f" {budget_class}:")
print(f" Max ΔBPB: {thresh.max_delta_bpb:+.2%}")
print(f" Min Effizienz: {thresh.min_efficiency_gain:+.1f}%")

print("\n" + "=" * 80)


# =============================================================================
# Phase 4B Commands
# =============================================================================

def print_anomalies(registry: RunRegistry, top_n: int = 10) -> None:
"""
Drucke Anomalie-Reports.

Args:
registry: RunRegistry
top_n: Anzahl der Top-Anomalien
"""
print("=" * 80)
print(" ANOMALIE-ERKENNUNG")
print("=" * 80)

detector = AnomalyDetector(significance_level=0.05)
all_reports: List[AnomalyReport] = []

# Alle Runs durchgehen
for run in registry.list_runs():
reports = detector.run_all_checks(run.run_id, registry)
all_reports.extend(reports)

if not all_reports:
print("\n Keine Anomalien erkannt.")
print("\n" + "=" * 80)
return

# Nach Schweregrad sortieren
severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
all_reports.sort(key=lambda r: severity_order.get(r.severity, 4))

# Statistik
stats = detector.get_summary_statistics(all_reports)
print(f"\n Zusammenfassung:")
print(f" Gesamt: {stats['total_anomalies']} Anomalien")
print(f" Kritisch: {stats.get('critical_count', 0)}")
print(f" Hoch: {stats.get('high_count', 0)}")
print(f" Nach Typ: {stats['by_type']}")

# Top-Anomalien
print(f"\n Top {min(top_n, len(all_reports))} Anomalien:")
print("-" * 80)

for i, report in enumerate(all_reports[:top_n], 1):
severity_icon = {"critical": "", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(report.severity, "")
print(f"\n{i}. {severity_icon} [{report.severity.upper()}] {report.anomaly_type}")
print(f" Run: {report.run_id}")
print(f" Beschreibung: {report.description}")

# Statistische Evidenz
if report.statistical_evidence:
print(f" Evidenz: ", end="")
evidence_parts = []
for key, value in list(report.statistical_evidence.items())[:3]:
if isinstance(value, float):
evidence_parts.append(f"{key}={value:.3f}")
else:
evidence_parts.append(f"{key}={value}")
print(", ".join(evidence_parts))

print(f" Empfehlung: {report.recommended_action[:80]}...")

print("\n" + "=" * 80)


def print_failures(registry: RunRegistry, top_n: int = 10) -> None:
"""
Drucke Fehler-Analyse.

Args:
registry: RunRegistry
top_n: Anzahl der Top-Fehler
"""
print("=" * 80)
print(" FEHLER-ANALYSE")
print("=" * 80)

classifier = FailureClassifier()

# Fehler-Statistiken
stats = classifier.get_failure_statistics(registry)
print(f"\n Fehler-Statistiken:")
print(f" Gesamt-Runs: {stats['total_runs']}")
print(f" Fehlgeschlagen: {stats['failed_runs']}")
print(f" Fehlerrate: {stats['failure_rate']:.1%}")
print(f" Nach Kategorie: {stats['by_category']}")
print(f" Häufigster Fehler: {stats['most_common_failure']}")

# Top-Fehler klassifizieren
failed_runs = [r for r in registry.list_runs() if r.status in ("failed", "killed")]

if not failed_runs:
print("\n Keine fehlgeschlagenen Runs.")
print("\n" + "=" * 80)
return

print(f"\n Top {min(top_n, len(failed_runs))} Fehler-Diagnosen:")
print("-" * 80)

diagnoses: List[FailureDiagnosis] = []
for run in failed_runs[:top_n]:
diagnosis = classifier.classify(run.run_id, registry)
if diagnosis:
diagnoses.append(diagnosis)

if not diagnoses:
print("\n Keine Diagnosen verfügbar.")
print("\n" + "=" * 80)
return

for i, diag in enumerate(diagnoses, 1):
confidence_icon = "" if diag.confidence < 0.5 else "🟡" if diag.confidence < 0.75 else "🟢"
print(f"\n{i}. {confidence_icon} {diag.failure_category.upper()} (Confidence: {diag.confidence:.1%})")
print(f" Run: {diag.run_id}")
print(f" Root Cause: {diag.root_cause}")

if diag.contributing_factors:
print(f" Faktoren: {', '.join(diag.contributing_factors)}")

if diag.similar_failures:
print(f" Ähnliche Fehler: {', '.join(diag.similar_failures[:3])}")

print(f" Empfohlener Fix: {diag.recommended_fix}")

print("\n" + "=" * 80)


def print_quarantine(registry: RunRegistry) -> None:
"""
Drucke Quarantäne-Liste.

Args:
registry: RunRegistry
"""
print("=" * 80)
print(" QUARANTÄNE-LISTE")
print("=" * 80)

manager = RunQuarantineManager()

# Statistik
stats = manager.get_feature_statistics()
print(f"\n Quarantäne-Statistik:")
print(f" Getrackte Features: {stats['total_features_tracked']}")
print(f" Aktive Quarantänen: {stats['active_quarantines']['total']}")
print(f" - Features: {stats['active_quarantines']['feature']}")
print(f" - Kombinationen: {stats['active_quarantines']['combination']}")
print(f" - Kontext-spezifisch: {stats['active_quarantines']['context_specific']}")
print(f" Erfolgreiche Runs: {stats['total_successful_runs']}")

# Aktive Quarantänen
entries = manager.get_quarantine_list()

if not entries:
print("\n Keine aktiven Quarantänen.")
print("\n" + "=" * 80)
return

print(f"\n Aktive Quarantänen ({len(entries)}):")
print("-" * 80)

for i, entry in enumerate(entries, 1):
type_icon = {"feature": "", "combination": "", "context_specific": ""}.get(entry.quarantine_type, "")
print(f"\n{i}. {type_icon} {entry.target}")
print(f" Typ: {entry.quarantine_type}")
print(f" Grund: {entry.reason}")
print(f" Ausgelöst durch: {', '.join(entry.triggered_by[:3])}")
print(f" Verbleibende Runs: {entry.remaining_runs} / {entry.quarantine_duration}")

if entry.context_filter:
print(f" Kontext: {entry.context_filter}")

print("\n" + "=" * 80)


def print_drift(registry: RunRegistry, top_n: int = 10) -> None:
"""
Drucke Drift-Reports.

Args:
registry: RunRegistry
top_n: Anzahl der Top-Drifts
"""
print("=" * 80)
print(" DRIFT-MONITORING")
print("=" * 80)

monitor = DriftMonitor(window_size=20, threshold=0.05)

# Environment Drift prüfen
env_report = monitor.detect_environment_drift(registry)
if env_report:
print(f"\n ENVIRONMENT DRIFT erkannt:")
print(f" Schweregrad: {env_report.severity}")
print(f" Empfehlung: {env_report.recommended_action}")

# Zusammenfassung
summary = monitor.get_drift_summary()
print(f"\n Drift-Zusammenfassung:")
print(f" Gesamt-Alerts: {summary['total_alerts']}")
print(f" Nach Typ: {summary['by_type']}")
print(f" Nach Schweregrad: {summary['by_severity']}")

# Aktive Alerts
alerts = monitor.get_drift_alerts(limit=top_n)

if not alerts:
print("\n Keine Drift-Alerts.")
print("\n" + "=" * 80)
return

print(f"\n Top {len(alerts)} Drift-Alerts:")
print("-" * 80)

for i, report in enumerate(alerts, 1):
severity_icon = {"high": "", "medium": "🟡", "low": "🟢"}.get(report.severity, "")
print(f"\n{i}. {severity_icon} [{report.severity.upper()}] {report.drift_type}")
print(f" Betroffene Features: {', '.join(report.affected_features) if report.affected_features else 'Alle'}")
print(f" Drift-Magnitude: {report.drift_magnitude:+.1%}")
print(f" Signifikanz: p={report.statistical_significance:.3f}")
print(f" Empfohlung: {report.recommended_action[:80]}...")

print("\n" + "=" * 80)


def print_recovery(registry: RunRegistry, top_n: int = 10) -> None:
"""
Drucke Recovery-Empfehlungen.

Args:
registry: RunRegistry
top_n: Anzahl der Top-Empfehlungen
"""
print("=" * 80)
print(" RECOVERY-EMPFEHLUNGEN")
print("=" * 80)

manager = RollbackManager(registry)
classifier = FailureClassifier()

# Rollback-Statistiken
stats = manager.get_rollback_statistics()
print(f"\n Rollback-Statistik:")
print(f" Gesamt-Rollbacks: {stats['total_rollbacks']}")
print(f" Erfolgsrate: {stats['success_rate']:.1%}")
print(f" Erfolgreich: {stats['success_count']}")
print(f" Partiell: {stats['partial_count']}")
print(f" Fehlgeschlagen: {stats['failed_count']}")
print(f" Häufigste Ursache: {stats['most_common_cause']}")

# Fehlgeschlagene Runs für Recovery
failed_runs = [r for r in registry.list_runs() if r.status in ("failed", "killed")]

if not failed_runs:
print("\n Keine fehlgeschlagenen Runs für Recovery.")
print("\n" + "=" * 80)
return

print(f"\n Recovery-Optionen für {min(top_n, len(failed_runs))} fehlgeschlagene Runs:")
print("-" * 80)

for i, run in enumerate(failed_runs[:top_n], 1):
diagnosis = classifier.classify(run.run_id, registry)

if diagnosis:
print(f"\n{i}. {run.run_id} - {diagnosis.failure_category.upper()}")
print(f" Confidence: {diagnosis.confidence:.1%}")
print(f" Root Cause: {diagnosis.root_cause}")
print(f" Empfohlener Fix: {diagnosis.recommended_fix}")

# Rollback-Empfehlung
last_stable = manager.get_last_stable_configuration(run.run_id)
if last_stable:
print(f" Rollback-Ziel: {last_stable}")
else:
print(f" Rollback-Ziel: Kein stabiler Vorfahre gefunden")
else:
print(f"\n{i}. {run.run_id} - Status: {run.status}")
if run.notes:
print(f" Notizen: {run.notes[:80]}...")

# Rollback-Historie
history = manager.get_rollback_history(limit=5)
if history:
print(f"\n Letzte Rollbacks:")
for record in history:
outcome_icon = {"success": "", "partial": "", "failed": ""}.get(record.outcome, "")
print(f" {outcome_icon} {record.rollback_id}: {record.failed_run_id} → {record.target_run_id}")

print("\n" + "=" * 80)


# =============================================================================
# Phase 4C Commands - Guardrail System & Integration
# =============================================================================

def print_guardrails() -> None:
"""Guardrail-Status anzeigen."""
print("=" * 80)
print(" GUARDRAIL STATUS")
print("=" * 80)

# Erstelle default Guardrails
config = create_default_guardrails()
manager = GuardrailManager(config)

print(f"\n Autonomie-Level: {config.level.value.upper()}")
print(f" Anzahl Guardrails: {len(config.guardrails)}")

print(f"\n {len(config.guardrails)} Guardrails aktiv:")

for guardrail in config.guardrails:
icon = "" if guardrail.is_hard_limit else ""
print(f"\n {icon} {guardrail.name}")
print(f" Typ: {guardrail.guardrail_type.value}")
print(f" Threshold: {guardrail.threshold:.2f}")
print(f" Hard Limit: {'Ja' if guardrail.is_hard_limit else 'Nein'}")
print(f" Action on Violation: {guardrail.action_on_violation}")
print(f" Beschreibung: {guardrail.description}")

print(f"\n Erlaubte Aktionen: {', '.join(config.allowed_actions)}")
print(f" Benötigt Approval für: {', '.join(config.requires_approval)}")

print("\n" + "=" * 80)


def print_approvals() -> None:
"""Ausstehende Freigaben anzeigen."""
print("=" * 80)
print(" AUSSTEHENDE FREIGABEN")
print("=" * 80)

# Erstelle Orchestrator und Interface
orchestrator = create_autonomy_orchestrator()
interface = create_approval_interface(orchestrator)

# Simuliere einige Pending Approvals für Demo
pending = interface.get_pending_approvals()

if not pending:
print("\n Keine ausstehenden Freigaben.")
print(" Alle Aktionen sind genehmigt oder abgelaufen.")
else:
print(f"\n {len(pending)} ausstehende Freigabe(n):")

for i, request in enumerate(pending, 1):
risk_icon = {"low": "🟢", "medium": "🟡", "high": ""}.get(
request.risk_level, ""
)
print(f"\n {i}. {request.action_type}")
print(f" {risk_icon} Risk: {request.risk_level}")
print(f" Confidence: {request.confidence:.1%}")
print(f" Erstellt: {request.created_at.strftime('%Y-%m-%d %H:%M')}")
print(f" Läuft ab: {request.expires_at.strftime('%Y-%m-%d %H:%M')}")

# Zeige Statistiken
stats = interface.get_approval_statistics()
print(f"\n Statistiken:")
print(f" Pending: {stats['pending']}")
print(f" Heute genehmigt: {stats['approved_today']}")
print(f" Heute abgelehnt: {stats['rejected_today']}")
print(f" Ø Genehmigungszeit: {stats['avg_approval_time']}")

print("\n" + "=" * 80)


def print_alerts(hours: int = 24) -> None:
"""Alert-Historie anzeigen."""
print("=" * 80)
print(" ALERT-HISTORIE")
print("=" * 80)

# Erstelle AlertManager
manager = create_alert_manager()

# Zeige Summary
summary = manager.get_alert_summary(hours=hours)

print(f"\n Zusammenfassung (letzte {hours}h):")
print(f" Total Alerts: {summary['total']}")
print(f" Davon bestätigt: {summary['acknowledged']}")
print(f" Davon ausstehend: {summary['pending']}")
print(f" Davon aufgelöst: {summary['resolved']}")
print(f" Action required: {summary['action_required']}")

print(f"\n Nach Schweregrad:")
for severity, count in summary.get("by_severity", {}).items():
icon = {"info": "", "warning": "", "high": "", "critical": ""}.get(
severity, "•"
)
print(f" {icon} {severity.upper()}: {count}")

# Zeige aktive Alerts
active_alerts = manager.get_active_alerts()
if active_alerts:
print(f"\n Aktive Alerts ({len(active_alerts)}):")
for alert in active_alerts[:10]:
sev_icon = {
AlertSeverity.INFO: "",
AlertSeverity.WARNING: "",
AlertSeverity.HIGH: "",
AlertSeverity.CRITICAL: "",
}.get(alert.severity, "•")
print(f" {sev_icon} [{alert.timestamp.strftime('%H:%M')}] {alert.title}")
print(f" {alert.message[:80]}...")
else:
print("\n Keine aktiven Alerts.")

print("\n" + "=" * 80)


def print_autonomy_stats() -> None:
"""Autonomie-Statistiken anzeigen."""
print("=" * 80)
print(" AUTONOMIE-STATISTIKEN")
print("=" * 80)

# Erstelle Orchestrator
orchestrator = create_autonomy_orchestrator()

stats = orchestrator.get_statistics()

total = stats["total_actions"]
if total == 0:
print("\n Noch keine Aktionen durchgeführt.")
print(" Das System ist bereit für autonome Operationen.")
else:
print(f"\n Gesamt-Statistiken:")
print(f" Total Actions: {total}")
print(f" Auto-ausgeführt: {stats['auto_executed']} ({stats['success_rate']:.0%})")
print(f" Human-genehmigt: {stats['human_approved']}")
print(f" Blockiert: {stats['blocked_by_guardrails']}")

# Erfolgsrate visualisieren
success_rate = stats["success_rate"]
bar_length = 30
filled = int(bar_length * success_rate)
bar = "" * filled + "" * (bar_length - filled)
print(f"\n Erfolgsrate: [{bar}] {success_rate:.0%}")

# Zeige Guardrail-Status
config = create_default_guardrails()
manager = GuardrailManager(config)
guardrail_status = manager.get_guardrail_status()

print(f"\n Guardrail-Konfiguration:")
print(f" Autonomie-Level: {guardrail_status['autonomy_level'].upper()}")
print(f" Total Guardrails: {guardrail_status['total_guardrails']}")

print("\n" + "=" * 80)


def print_overrides(hours: int = 24) -> None:
"""Override-Analyse anzeigen."""
print("=" * 80)
print(" OVERRIDE-ANALYSE")
print("=" * 80)

# Erstelle OverrideLearner
learner = create_override_learner()

# Zeige Statistiken
stats = learner.get_override_statistics(hours=hours)

print(f"\n Zusammenfassung (letzte {hours}h):")
print(f" Total Overrides: {stats['total_overrides']}")

if stats['total_overrides'] == 0:
print("\n Keine Overrides in diesem Zeitraum.")
print(" Das System trifft autonome Entscheidungen ohne Human-Eingriffe.")
else:
print(f"\n Nach Entscheidung:")
for decision, count in stats.get("by_decision", {}).items():
icon = {"approve": "", "reject": "", "modify": ""}.get(decision, "•")
print(f" {icon} {decision}: {count}")

print(f"\n Nach Action-Type:")
for action_type, count in stats.get("by_action_type", {}).items():
print(f" • {action_type}: {count}")

print(f"\n Ø Confidence vor Override: {stats.get('avg_confidence_before', 0):.1%}")

# Zeige Muster-Analyse
patterns = learner.analyze_override_patterns()
if patterns["most_overridden_actions"]:
print(f"\n Häufigste Overrides:")
for item in patterns["most_overridden_actions"][:5]:
print(f" • {item['action']}: {item['count']}x")

# Zeige Threshold-Vorschläge
suggestions = learner.suggest_threshold_adjustments()
if suggestions:
print(f"\n Threshold-Anpassungsvorschläge:")
for sug in suggestions[:3]:
print(f" • {sug['suggestion']}")
print(f" → {sug['recommendation']}")

print("\n" + "=" * 80)


# ============================================================================
# Phase 4 Evaluation Commands (Woche 9-10)
# ============================================================================


def print_success_metrics(registry: RunRegistry) -> None:
"""
Phase 4 Success Metrics anzeigen.

Zeigt alle 5 Success Metrics mit Status an.
"""
print("\n" + "=" * 80)
print("PHASE 4 SUCCESS METRICS")
print("=" * 80)

tracker = SuccessMetricsTracker(registry)
metrics = tracker.get_all_metrics()

targets_met = sum(1 for m in metrics.values() if m.target_met)
total_targets = len(metrics)

print(f"\n **{targets_met}/{total_targets} Ziele erreicht**\n")

for name, metric in metrics.items():
status_icon = "" if metric.target_met else ""
print(f"{status_icon} {metric.name}")
print(f" Aktueller Wert: {metric.current_value:.2f} {metric.unit}")
print(f" Zielwert: {metric.target_value:.2f} {metric.unit}")
print(f" Baseline: {metric.baseline_value:.2f} {metric.unit}")
print(f" Richtung: {'höher besser' if metric.direction == 'higher_better' else 'niedriger besser'}")
print()

# Zusammenfassung
print("-" * 80)
if targets_met == total_targets:
print(" Alle Ziele erreicht! Phase 4 erfolgreich abgeschlossen.")
elif targets_met >= total_targets * 0.8:
print(f" {targets_met}/{total_targets} Ziele erreicht. Knapp verfehlte Ziele analysieren.")
else:
print(f" Nur {targets_met}/{total_targets} Ziele erreicht. Weitere Optimierung empfohlen.")

# Nicht erreichte Ziele
missed = [m for m in metrics.values() if not m.target_met]
if missed:
print("\nNicht erreichte Ziele:")
for m in missed:
gap = m.target_value - m.current_value if m.direction == "higher_better" else m.current_value - m.target_value
print(f" - {m.name}: Gap von {gap:.2f} {metric.unit}")

print("\n" + "=" * 80)


def cmd_ab_test(registry: RunRegistry, action: str, test_id: Optional[str] = None) -> None:
"""
A/B-Test Command.

Actions:
- create: Neuen Test erstellen
- list: Alle Tests auflisten
- analyze: Test analysieren
- summary: Test-Zusammenfassung
"""
print("\n" + "=" * 80)
print("A/B-TESTING FRAMEWORK")
print("=" * 80)

framework = ABTestFramework(registry)

if action == "create":
# Beispiel-Test erstellen
from datetime import timedelta
config = ABTestConfig(
test_name="autonomous_vs_manual",
start_date=datetime.now(),
end_date=datetime.now() + timedelta(days=14),
treatment_group="autonomous",
control_group="manual",
success_metrics=["delta_bpb", "efficiency_gain", "human_time_minutes"],
min_sample_size=30,
)
new_test_id = framework.create_test(config)
print(f"\n A/B-Test erstellt:")
print(f" Test-ID: {new_test_id}")
print(f" Name: {config.test_name}")
print(f" Treatment: {config.treatment_group}")
print(f" Control: {config.control_group}")
print(f" Success Metrics: {', '.join(config.success_metrics)}")
print(f" Min Sample Size: {config.min_sample_size}")
print(f"\nNächste Schritte:")
print(f" 1. Runs zuweisen: framework.assign_run_to_group('{new_test_id}')")
print(f" 2. Outcomes recorden: framework.record_outcome('{new_test_id}', group, run_id, metrics)")
print(f" 3. Analysieren: python -m orchestrator.meta_dashboard ab-test analyze {new_test_id}")

elif action == "list":
tests = framework.list_tests()
if not tests:
print("\nKeine A/B-Tests gefunden.")
else:
print(f"\n{'Test-ID':<12} {'Name':<30} {'Status':<12} {'Treatment':<12} {'Control':<10}")
print("-" * 80)
for test in tests:
print(f"{test['test_id']:<12} {test['test_name']:<30} {test['status']:<12} {test['treatment_runs']:<12} {test['control_runs']:<10}")

elif action == "analyze":
if not test_id:
print(" Fehler: test_id required für analyze")
return

try:
results = framework.analyze_test(test_id)
if not results:
print(f"\n Keine ausreichenden Daten für Test '{test_id}'")
print(" Mindestens 2 Outcomes pro Gruppe benötigt.")
return

summary = framework.get_test_summary(test_id)
print(f"\n A/B-Test: {summary['test_name']}")
print(f" Status: {summary['status']}")
print(f" Treatment ({summary['treatment_group']}): {summary['treatment_runs']} Runs")
print(f" Control ({summary['control_group']}): {summary['control_runs']} Runs")

print(f"\n{'Metrik':<25} {'Treatment':<12} {'Control':<12} {'Diff':<10} {'p-value':<10} {'Effect':<8} {'Signif.':<8}")
print("-" * 95)

for r in results:
diff = r.treatment_mean - r.control_mean
sig_icon = "" if r.is_significant else ""
print(f"{r.metric:<25} {r.treatment_mean:<12.4f} {r.control_mean:<12.4f} {diff:<+10.4f} {r.p_value:<10.4f} {r.effect_size:<8.2f} {sig_icon:<8}")

print(f"\n Empfehlung: {summary['recommendation']}")

if summary['significant_wins']:
print(f"\nSignifikante Ergebnisse:")
for win in summary['significant_wins']:
print(f" - {win['metric']}: {win['winner']} wins (p={win['p_value']:.4f}, d={win['effect_size']:.2f})")

except ValueError as e:
print(f" Fehler: {e}")

elif action == "summary":
if not test_id:
print(" Fehler: test_id required für summary")
return

try:
summary = framework.get_test_summary(test_id)
print(f"\n Test-Zusammenfassung: {summary['test_name']}")
print(f" Test-ID: {summary['test_id']}")
print(f" Status: {summary['status']}")
print(f" Treatment-Gruppe: {summary['treatment_group']} ({summary['treatment_runs']} Runs)")
print(f" Control-Gruppe: {summary['control_group']} ({summary['control_runs']} Runs)")

if summary['significant_wins']:
print(f"\nSignifikante Wins:")
for win in summary['significant_wins']:
print(f" - {win['metric']}: {win['winner']} (p={win['p_value']:.4f})")

print(f"\n Empfehlung: {summary['recommendation']}")

except ValueError as e:
print(f" Fehler: {e}")

else:
print(f" Unbekannte Action: {action}")
print(" Erlaubt: create, list, analyze, summary")

print("\n" + "=" * 80)


def cmd_refinement(registry: RunRegistry, top_n: int = 5) -> None:
"""
Refinement-Vorschläge anzeigen.

Analysiert Guardrails, Prediction Errors und Human Overrides.
"""
print("\n" + "=" * 80)
print("REFINEMENT ENGINE")
print("=" * 80)

# Initialisiere benötigte Komponenten
guardrail_manager = create_default_guardrails()
override_learner = create_override_learner(history_limit=1000)
scorer = SurrogateScorer(model_type="random_forest")

engine = RefinementEngine(
scorer=scorer,
guardrail_manager=guardrail_manager,
override_learner=override_learner,
registry=registry,
)

# Analysieren
print("\n Analysiere Guardrail-Performance...")
guardrail_suggestions = engine.analyze_guardrail_performance()

print(" Analysiere Prediction Errors...")
prediction_suggestions = engine.analyze_prediction_errors()

print(" Analysiere Human Overrides...")
override_suggestions = engine.analyze_human_overrides()

# Kombinieren und sortieren
all_suggestions = guardrail_suggestions + prediction_suggestions + override_suggestions
all_suggestions = sorted(all_suggestions, key=lambda s: (s.priority, -s.confidence))

if not all_suggestions:
print("\n Keine Refinement-Vorschläge identifiziert.")
print("\n" + "=" * 80)
return

print(f"\n **{len(all_suggestions)} Refinement-Vorschläge identifiziert**")
print(f" Priority 1 (Hoch): {sum(1 for s in all_suggestions if s.priority == 1)}")
print(f" Priority 2 (Mittel): {sum(1 for s in all_suggestions if s.priority == 2)}")
print(f" Priority 3-5 (Niedrig): {sum(1 for s in all_suggestions if s.priority >= 3)}")

# Top N anzeigen
print(f"\n## Top {min(top_n, len(all_suggestions))} Refinement-Vorschläge\n")

for i, sug in enumerate(all_suggestions[:top_n], 1):
priority_icon = "" if sug.priority == 1 else "🟡" if sug.priority == 2 else "🟢"
print(f"{i}. {sug.component.capitalize()} {priority_icon}")
print(f" Current: {sug.current_behavior[:100]}")
print(f" Change: {sug.suggested_change[:100]}")
print(f" Expected: {sug.expected_improvement}")
print(f" Confidence: {sug.confidence:.0%}")
print()

# Implementierungsempfehlungen
high_priority = [s for s in all_suggestions if s.priority == 1]
if high_priority:
print("## Sofort umsetzen (Priority 1):")
for sug in high_priority:
print(f" - {sug.suggested_change[:80]}")

print("\n" + "=" * 80)


def cmd_report(registry: RunRegistry, output_path: str = "reports/phase4_evaluation.md") -> None:
"""
Vollständigen Phase 4 Evaluations-Report generieren.
"""
from scripts.generate_phase4_docs import Phase4DocumentationGenerator

print("\n" + "=" * 80)
print("PHASE 4 EVALUATION REPORT")
print("=" * 80)

generator = Phase4DocumentationGenerator(registry)

print(f"\n Generiere Report...")
report = generator.generate_full_report(output_path=output_path)

print(f"\n Report gespeichert unter: {output_path}")

# Kurze Zusammenfassung drucken
metrics = generator.metrics_tracker.get_all_metrics()
targets_met = sum(1 for m in metrics.values() if m.target_met)
total_targets = len(metrics)

print(f"\n Success Metrics: {targets_met}/{total_targets} Ziele erreicht")

for name, metric in metrics.items():
status = "" if metric.target_met else ""
print(f" {status} {metric.name}: {metric.current_value:.1f}{metric.unit}")

print("\n" + "=" * 80)


# Import für datetime
from datetime import datetime


def parse_args() -> argparse.Namespace:
"""Parse command-line arguments."""
parser = argparse.ArgumentParser(
description="Meta-Feature Dashboard für NeuroWeave",
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog="""
Beispiele:
%(prog)s summary # Übersicht aller Features
%(prog)s summary --top 20 # Top 20 Features anzeigen
%(prog)s co-occurrence # Co-occurrence Matrix
%(prog)s co-occurrence --limit 30 # Top 30 Paare
%(prog)s feature-stats # Statistiken pro Feature
%(prog)s feature-stats --sort-by success_rate
%(prog)s lineage # Lineage-Analyse
%(prog)s budget # Budget-Klassen Analyse
%(prog)s quant # Quantisierungs-Analyse
%(prog)s predictions # Surrogate Scorer Vorhersagen
%(prog)s hypotheses --top 10 # Run-Vorschläge generieren
%(prog)s pareto # Pareto-Frontier anzeigen
%(prog)s pareto --plot # Mit Plot
%(prog)s recommendations --top 5 # Top-Empfehlungen
%(prog)s anomalies # Anomalien anzeigen (Phase 4B)
%(prog)s failures # Fehler-Analyse (Phase 4B)
%(prog)s quarantine # Quarantäne-Liste (Phase 4B)
%(prog)s drift # Drift-Reports (Phase 4B)
%(prog)s recovery # Recovery-Empfehlungen (Phase 4B)
""",
)

parser.add_argument(
"command",
choices=[
"summary", "co-occurrence", "feature-stats", "lineage",
"budget", "quant", "predictions", "hypotheses", "pareto",
"recommendations", "anomalies", "failures", "quarantine",
"drift", "recovery", "guardrails", "approvals", "alerts",
"autonomy-stats", "overrides",
# Phase 4 Evaluation (Woche 9-10)
"metrics", "ab-test", "refinement", "report"
],
help="Dashboard Command",
)

parser.add_argument(
"--json",
type=str,
help="JSON-Datei mit Meta-Features (default: lade aus Registry)",
)

parser.add_argument(
"--top",
type=int,
default=10,
help="Anzahl der Top-Einträge (default: 10)",
)

parser.add_argument(
"--limit",
type=int,
default=20,
help="Limit für Ausgabe (default: 20)",
)

parser.add_argument(
"--min-count",
type=int,
default=1,
help="Minimale Anzahl für Anzeige (default: 1)",
)

parser.add_argument(
"--sort-by",
type=str,
choices=["count", "avg_delta", "success_rate"],
default="count",
help="Sortierreihenfolge für feature-stats (default: count)",
)

parser.add_argument(
"--plot",
action="store_true",
help="Erstelle Plot (für pareto Command)",
)

# Phase 4 Evaluation Arguments
parser.add_argument(
"--action",
type=str,
choices=["create", "list", "analyze", "summary"],
default="list",
help="Action für ab-test Command (default: list)",
)

parser.add_argument(
"--test-id",
type=str,
help="Test-ID für ab-test analyze/summary",
)

parser.add_argument(
"--output",
type=str,
default="reports/phase4_evaluation.md",
help="Output-Pfad für report Command",
)

return parser.parse_args()


def main() -> None:
"""Hauptfunktion."""
args = parse_args()

# Initialisiere Registry und Extractor
results_dir = Path(__file__).parent.parent / "results"
registry = RunRegistry(results_dir=str(results_dir))
extractor = MetaFeatureExtractor(configs_dir=Path(__file__).parent.parent / "configs")

# Phase 4 Evaluation Commands (Woche 9-10) benötigen keine Features
phase4_evaluation_commands = ("metrics", "ab-test", "refinement", "report")

if args.command in phase4_evaluation_commands:
if args.command == "metrics":
print_success_metrics(registry)
elif args.command == "ab-test":
cmd_ab_test(registry, args.action, args.test_id)
elif args.command == "refinement":
cmd_refinement(registry, top_n=args.top)
elif args.command == "report":
cmd_report(registry, output_path=args.output)
return

# Phase 4B und 4C Commands benötigen keine Features
phase4b_4c_commands = (
"anomalies", "failures", "quarantine", "drift", "recovery",
"guardrails", "approvals", "alerts", "autonomy-stats", "overrides"
)

if args.command in phase4b_4c_commands:
if args.command == "anomalies":
print_anomalies(registry, top_n=args.top)
elif args.command == "failures":
print_failures(registry, top_n=args.top)
elif args.command == "quarantine":
print_quarantine(registry)
elif args.command == "drift":
print_drift(registry, top_n=args.top)
elif args.command == "recovery":
print_recovery(registry, top_n=args.top)
elif args.command == "guardrails":
print_guardrails()
elif args.command == "approvals":
print_approvals()
elif args.command == "alerts":
print_alerts()
elif args.command == "autonomy-stats":
print_autonomy_stats()
elif args.command == "overrides":
print_overrides()
return

# Lade Features für andere Commands
features = get_features(registry, extractor, json_path=args.json)

if not features:
print(" Keine Meta-Features gefunden.")
sys.exit(1)

# Führe Command aus
if args.command == "summary":
print_summary(features, top_n=args.top)
elif args.command == "co-occurrence":
print_co_occurrence(features, limit=args.limit, min_count=args.min_count)
elif args.command == "feature-stats":
print_feature_stats(features, min_count=args.min_count, sort_by=args.sort_by)
elif args.command == "lineage":
print_lineage_analysis(features)
elif args.command == "budget":
print_budget_analysis(features)
elif args.command == "quant":
print_quant_analysis(features)
elif args.command == "predictions":
print_predictions(features, top_n=args.top)
elif args.command == "hypotheses":
print_hypotheses(features, top_n=args.top)
elif args.command == "pareto":
print_pareto_frontier(features, plot=args.plot)
elif args.command == "recommendations":
print_recommendations(features, top_n=args.top)


if __name__ == "__main__":
main()
