#!/usr/bin/env python3
"""
Meta-Feature Dashboard für NeuroWeave.

Dieses Dashboard bietet Einblicke in die extrahierten Meta-Features
aller Runs und hilft bei der Analyse von Feature-Performance und
Co-occurrence Mustern.

Phase 3 Commands:
    python -m orchestrator.meta_dashboard summary       # Übersicht aller Features
    python -m orchestrator.meta_dashboard co-occurrence # Feature Co-occurrence Matrix
    python -m orchestrator.meta_dashboard feature-stats # Statistiken pro Feature
    python -m orchestrator.meta_dashboard lineage       # Lineage-Analyse
    python -m orchestrator.meta_dashboard budget        # Budget-Klassen Analyse
    python -m orchestrator.meta_dashboard quant         # Quantisierungs-Analyse

Phase 4A Commands (neu):
    python -m orchestrator.meta_dashboard predictions     # Surrogate Scorer Vorhersagen
    python -m orchestrator.meta_dashboard hypotheses      # Run-Vorschläge generieren
    python -m orchestrator.meta_dashboard pareto          # Pareto-Frontier anzeigen
    python -m orchestrator.meta_dashboard recommendations # Top-Empfehlungen

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
    print(f"\n📊 Gesamtübersicht:")
    print(f"   Total Runs: {len(features)}")
    
    # Unique Features
    all_features = set()
    for f in features:
        all_features.update(f.features_active)
    print(f"   Unique Features: {len(all_features)}")
    
    # Top Features nach Häufigkeit
    feature_counts = defaultdict(int)
    for f in features:
        for feat in f.features_active:
            feature_counts[feat] += 1
    
    if feature_counts:
        sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])[:top_n]
        print(f"\n   Top {top_n} Features nach Häufigkeit:")
        for feat, count in sorted_features:
            pct = 100 * count / len(features)
            print(f"     • {feat}: {count} ({pct:.1f}%)")
    
    # Budget Class Distribution
    budget_dist = defaultdict(int)
    for f in features:
        budget_dist[f.budget_class] += 1
    
    print(f"\n💰 Budget-Klassen:")
    for budget_class in ["low", "medium", "high"]:
        count = budget_dist.get(budget_class, 0)
        pct = 100 * count / len(features) if features else 0
        bar = "█" * int(pct / 5)
        print(f"   {budget_class.upper():8}: {count:3} ({pct:5.1f}%) {bar}")
    
    # Quantization Distribution
    quant_dist = defaultdict(int)
    for f in features:
        quant_dist[f.quantization_type] += 1
    
    print(f"\n🔧 Quantisierungs-Typen:")
    for quant_type in ["none", "int6", "int5", "mixed", "gptq_lite"]:
        count = quant_dist.get(quant_type, 0)
        pct = 100 * count / len(features) if features else 0
        bar = "█" * int(pct / 5)
        print(f"   {quant_type:8}: {count:3} ({pct:5.1f}%) {bar}")
    
    # Sequence Length Distribution
    seq_dist = defaultdict(int)
    for f in features:
        seq_dist[f.sequence_length] += 1
    
    print(f"\n📏 Sequenzlängen:")
    for seq_len in ["local", "remote"]:
        count = seq_dist.get(seq_len, 0)
        pct = 100 * count / len(features) if features else 0
        bar = "█" * int(pct / 5)
        print(f"   {seq_len:8}: {count:3} ({pct:5.1f}%) {bar}")
    
    # Lineage Statistics
    depths = [f.lineage_depth for f in features]
    siblings = [f.siblings_count for f in features]
    
    if depths:
        print(f"\n🌳 Lineage-Statistiken:")
        print(f"   Depth:       min={min(depths):2}, max={max(depths):2}, avg={sum(depths) / len(depths):5.2f}")
        print(f"   Siblings:    min={min(siblings):2}, max={max(siblings):2}, avg={sum(siblings) / len(siblings):5.2f}")
    
    # Performance Statistics
    completed_with_parent = [f for f in features if f.delta_bpb_vs_parent != 0.0 or f.parent_run_id is not None]
    
    if completed_with_parent:
        deltas = [f.delta_bpb_vs_parent for f in completed_with_parent if f.delta_bpb_vs_parent != 0.0]
        if deltas:
            print(f"\n📈 Performance (ΔBPB vs Parent):")
            print(f"   Count:  {len(deltas)} Runs mit Parent-Vergleich")
            print(f"   min:    {min(deltas):+.4f} (beste Verbesserung)")
            print(f"   max:    {max(deltas):+.4f} (schlechteste Veränderung)")
            print(f"   avg:    {sum(deltas) / len(deltas):+.4f}")
            
            # Zähle Verbesserungen vs Verschlechterungen
            improvements = sum(1 for d in deltas if d < 0)
            degradations = sum(1 for d in deltas if d > 0)
            print(f"   Verbesserungen: {improvements} ({100 * improvements / len(deltas):.1f}%)")
            print(f"   Verschlechterungen: {degradations} ({100 * degradations / len(deltas):.1f}%)")
    
    # Effizienz-Statistiken
    efficiency_gains = [f.efficiency_gain_percent for f in features if f.efficiency_gain_percent != 0.0]
    if efficiency_gains:
        print(f"\n⚡ Effizienz-Gewinn (%):")
        print(f"   min: {min(efficiency_gains):+.1f}%, max: {max(efficiency_gains):+.1f}%, avg: {sum(efficiency_gains) / len(efficiency_gains):+.1f}%")
    
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
        for f in sorted_features[:12]:  # Max 12 für Lesbarkeit
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
        stats.sort(key=lambda x: x["avg_delta"])  # Niedriger = besser
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
    
    print(f"\n📊 Übersicht:")
    print(f"   Root Runs (ohne Parent): {len(root_runs)}")
    print(f"   Runs mit Parent: {len(features) - len(root_runs)}")
    
    # Lineage-Tiefe Verteilung
    depth_dist = defaultdict(int)
    for f in features:
        depth_dist[f.lineage_depth] += 1
    
    print(f"\n🌳 Lineage-Tiefe Verteilung:")
    for depth in sorted(depth_dist.keys()):
        count = depth_dist[depth]
        bar = "█" * count
        print(f"   Depth {depth}: {count:3} {bar}")
    
    # Größte Familien
    family_sizes = [(parent_id, len(runs)) for parent_id, runs in parent_groups.items() if parent_id is not None]
    family_sizes.sort(key=lambda x: -x[1])
    
    if family_sizes:
        print(f"\n👨‍👩‍👧‍👦 Top 5 Parent-Familien (nach Children-Anzahl):")
        for parent_id, size in family_sizes[:5]:
            print(f"   {parent_id}: {size} Children")
    
    # Siblings-Statistiken
    siblings_counts = [f.siblings_count for f in features if f.siblings_count > 0]
    if siblings_counts:
        print(f"\n👫 Siblings-Statistiken:")
        print(f"   Runs mit Siblings: {len(siblings_counts)}")
        print(f"   Max Siblings: {max(siblings_counts)}")
        print(f"   Avg Siblings: {sum(siblings_counts) / len(siblings_counts):.2f}")
    
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
        
        print(f"\n💰 Budget-Klasse: {budget_class.upper()}")
        print(f"   Runs: {len(runs)}")
        
        # Feature-Verteilung
        feature_counts = defaultdict(int)
        for f in runs:
            for feat in f.features_active:
                feature_counts[feat] += 1
        
        if feature_counts:
            print(f"   Top Features:")
            sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])[:5]
            for feat, count in sorted_features:
                pct = 100 * count / len(runs)
                print(f"     • {feat}: {count} ({pct:.1f}%)")
        
        # Performance
        deltas = [f.delta_bpb_vs_parent for f in runs if f.delta_bpb_vs_parent != 0.0]
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            improvements = sum(1 for d in deltas if d < 0)
            print(f"   Performance:")
            print(f"     Avg ΔBPB: {avg_delta:+.4f}")
            print(f"     Verbesserungen: {improvements}/{len(deltas)} ({100 * improvements / len(deltas):.1f}%)")
    
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
        
        print(f"\n🔧 Quantisierung: {quant_type.upper()}")
        print(f"   Runs: {len(runs)}")
        
        # Quant-Gap Statistiken
        gaps = [f.quant_gap for f in runs if f.quant_gap != 0.0]
        if gaps:
            print(f"   Quant-Gap (BPB Degradation):")
            print(f"     min: {min(gaps):.4f}")
            print(f"     max: {max(gaps):.4f}")
            print(f"     avg: {sum(gaps) / len(gaps):.4f}")
        
        # Feature-Verteilung
        feature_counts = defaultdict(int)
        for f in runs:
            for feat in f.features_active:
                feature_counts[feat] += 1
        
        if feature_counts:
            print(f"   Häufige Features:")
            sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])[:5]
            for feat, count in sorted_features:
                pct = 100 * count / len(runs)
                print(f"     • {feat}: {count} ({pct:.1f}%)")
    
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
        print(f"❌ Nicht genügend Trainingsdaten ({len(train_features)} < 5)")
        print("   Führe mehr Runs durch um den Surrogate Scorer zu trainieren.")
        return

    # Modell trainieren
    try:
        metrics = scorer.train(train_features, {
            "delta_bpb": targets["delta_bpb"][:len(train_features)],
            "efficiency_gain": targets["efficiency_gain"][:len(train_features)],
        })

        print(f"\n📊 Modell-Statistiken:")
        print(f"   Modell-Typ: Random Forest")
        print(f"   Trainings-Runs: {len(train_features)}")
        print(f"   BPB CV-RMSE: {metrics.get('bpb_cv_rmse', 0):.4f}")
        print(f"   Efficiency CV-RMSE: {metrics.get('efficiency_cv_rmse', 0):.4f}")

        # Feature-Importance anzeigen
        importance = scorer.get_feature_importance()
        sorted_importance = sorted(importance.items(), key=lambda x: -x[1])[:10]

        print(f"\n🔍 Top 10 Feature-Importancen:")
        for feat, imp in sorted_importance:
            bar = "█" * int(imp * 20)
            print(f"   {feat:<30} {imp:.4f} {bar}")

        # Vorhersagen für alle Runs
        print(f"\n📈 Vorhersagen (Top {top_n}):")
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
            symbol = "✓" if pred_bpb < 0 else "✗"
            print(f"{run_id:<40} {bpb_str:>12} {eff_str:>12} {conf_str:>12} {symbol}")

    except Exception as e:
        print(f"❌ Fehler beim Trainieren: {e}")

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
        print(f"❌ Nicht genügend Trainingsdaten ({len(train_features)} < 5)")
        return

    targets = {
        "delta_bpb": [f.delta_bpb_vs_parent for f in train_features],
        "efficiency_gain": [f.efficiency_gain_percent for f in train_features],
    }

    try:
        scorer.train(train_features, targets)
    except Exception as e:
        print(f"❌ Fehler beim Trainieren: {e}")
        return

    # Hypothesis Generator initialisieren
    generator = HypothesisGenerator(scorer, features)

    # Alle Hypothesen generieren
    hypotheses = generator.generate_all()

    if not hypotheses:
        print("❌ Keine Hypothesen generiert.")
        return

    print(f"\n📊 Übersicht:")
    print(f"   Total Hypothesen: {len(hypotheses)}")

    exploitation_count = sum(1 for h in hypotheses if h.hypothesis_type == "exploitation")
    exploration_count = sum(1 for h in hypotheses if h.hypothesis_type == "exploration")
    pattern_count = sum(1 for h in hypotheses if h.hypothesis_type == "pattern_based")

    print(f"   Exploitation: {exploitation_count}")
    print(f"   Exploration: {exploration_count}")
    print(f"   Pattern-based: {pattern_count}")

    # Feature-Erfolgsraten
    success_rates = generator.get_feature_success_rates()
    if success_rates:
        print(f"\n🏆 Top Feature-Erfolgsraten:")
        sorted_rates = sorted(success_rates.items(), key=lambda x: -x[1])[:5]
        for feat, rate in sorted_rates:
            count = generator.feature_counts.get(feat, 0)
            print(f"   {feat:<25} {rate:.1%} ({count} Runs)")

    # Top-Hypothesen anzeigen
    print(f"\n💡 Top {top_n} Hypothesen:")
    print("-" * 80)

    for i, h in enumerate(hypotheses[:top_n], 1):
        risk_symbol = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(h.risk_level, "⚪")
        type_symbol = {
            "exploitation": "⚡",
            "exploration": "🔍",
            "pattern_based": "🧩"
        }.get(h.hypothesis_type, "•")

        print(f"\n{i:2}. {type_symbol} {', '.join(h.features_proposed)}")
        print(f"    Vorhergesagtes ΔBPB: {h.predicted_delta_bpb:+.4f}")
        print(f"    Confidence: {h.confidence:.1%}")
        print(f"    Risiko: {h.risk_level} {risk_symbol}")
        print(f"    Begründung: {h.reasoning}")

        if h.similar_successful_runs:
            similar_str = ", ".join(h.similar_successful_runs[:3])
            print(f"    Ähnliche Runs: {similar_str}")

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

    print(f"\n📊 Übersicht:")
    print(f"   Total Runs: {len(tracker.points)}")
    print(f"   Pareto-optimal: {len(frontier)}")
    print(f"   Dominiert: {len(dominated)}")

    # Statistik
    stats = tracker.get_statistics()
    print(f"\n📈 Frontier-Statistiken:")
    print(f"   Volumen: {stats.get('frontier_volume', 0):.2f}")
    print(f"   Bestes ΔBPB: {stats.get('best_delta_bpb', 0):+.4f}")
    print(f"   Beste Effizienz: {stats.get('best_efficiency_gain', 0):+.1f}%")
    print(f"   Frontier Expansion: {stats.get('frontier_expansion', 0):+.1%}")

    # Pareto-optimale Runs anzeigen
    if frontier:
        print(f"\n🏆 Pareto-optimale Runs:")
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
        print(f"\n🔍 Identifizierte Lücken:")
        for i, gap in enumerate(gaps, 1):
            print(f"   {i}. Target ΔBPB: {gap['target_bpb']:+.4f}, "
                  f"Effizienz: {gap['target_efficiency']:+.1f}%")
            print(f"      {gap['reason'][:80]}...")

    # Plot erstellen
    if plot:
        try:
            output_path = tracker.plot_frontier(output_path="results/pareto_frontier.png")
            print(f"\n📊 Plot erstellt: {output_path}")
        except ImportError as e:
            print(f"\n⚠️  Plotting nicht verfügbar: {e}")
        except Exception as e:
            print(f"\n⚠️  Fehler beim Plotten: {e}")

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
        print(f"❌ Nicht genügend Trainingsdaten ({len(train_features)} < 5)")
        return

    targets = {
        "delta_bpb": [f.delta_bpb_vs_parent for f in train_features],
        "efficiency_gain": [f.efficiency_gain_percent for f in train_features],
    }

    try:
        scorer.train(train_features, targets)
    except Exception as e:
        print(f"❌ Fehler beim Trainieren: {e}")
        return

    # Hypothesis Generator
    generator = HypothesisGenerator(scorer, features)
    hypotheses = generator.generate_all()

    # Adaptive Kill Thresholds
    kill_manager = AdaptiveKillThresholdManager()
    kill_stats = kill_manager.get_kill_statistics()

    print(f"\n📊 Analyse-Basis:")
    print(f"   Trainings-Runs: {len(train_features)}")
    print(f"   Generierte Hypothesen: {len(hypotheses)}")
    print(f"   Kill-Rate (recent): {kill_stats.get('kill_rate', 0):.1%}")

    # Top-Empfehlungen
    if hypotheses:
        print(f"\n💡 Top {top_n} Empfehlungen:")
        print("-" * 80)

        for i, h in enumerate(hypotheses[:top_n], 1):
            priority = "🔥" if i <= 2 else "⭐" if i <= 5 else "•"
            risk_symbol = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(h.risk_level, "⚪")

            print(f"\n{i}. {priority} {', '.join(h.features_proposed)}")
            print(f"   Typ: {h.hypothesis_type} | Risiko: {h.risk_level} {risk_symbol}")
            print(f"   Erwartetes ΔBPB: {h.predicted_delta_bpb:+.4f} (Confidence: {h.confidence:.1%})")
            print(f"   Begründung: {h.reasoning[:100]}...")

    # Diminishing Returns Features
    diminishing = generator.get_diminishing_returns_features()
    if diminishing:
        print(f"\n⚠️  Features mit sinkendem Grenznutzen:")
        for feat in diminishing[:5]:
            rate = generator.feature_success_rates.get(feat, 0)
            count = generator.feature_counts.get(feat, 0)
            print(f"   • {feat}: {rate:.1%} Erfolgsrate in {count} Runs")

    # Kill-Threshold Empfehlungen
    print(f"\n🛑 Kill-Threshold Empfehlungen:")
    for budget_class in ["low_budget", "medium_budget", "high_budget"]:
        thresh = kill_manager.get_thresholds(budget_class)
        print(f"   {budget_class}:")
        print(f"     Max ΔBPB: {thresh.max_delta_bpb:+.2%}")
        print(f"     Min Effizienz: {thresh.min_efficiency_gain:+.1f}%")

    print("\n" + "=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Meta-Feature Dashboard für NeuroWeave",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s summary                    # Übersicht aller Features
  %(prog)s summary --top 20           # Top 20 Features anzeigen
  %(prog)s co-occurrence              # Co-occurrence Matrix
  %(prog)s co-occurrence --limit 30   # Top 30 Paare
  %(prog)s feature-stats              # Statistiken pro Feature
  %(prog)s feature-stats --sort-by success_rate
  %(prog)s lineage                    # Lineage-Analyse
  %(prog)s budget                     # Budget-Klassen Analyse
  %(prog)s quant                      # Quantisierungs-Analyse
  %(prog)s predictions                # Surrogate Scorer Vorhersagen
  %(prog)s hypotheses --top 10        # Run-Vorschläge generieren
  %(prog)s pareto                     # Pareto-Frontier anzeigen
  %(prog)s pareto --plot              # Mit Plot
  %(prog)s recommendations --top 5    # Top-Empfehlungen
        """,
    )

    parser.add_argument(
        "command",
        choices=[
            "summary", "co-occurrence", "feature-stats", "lineage",
            "budget", "quant", "predictions", "hypotheses", "pareto",
            "recommendations"
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

    return parser.parse_args()


def main() -> None:
    """Hauptfunktion."""
    args = parse_args()

    # Initialisiere Registry und Extractor
    results_dir = Path(__file__).parent.parent / "results"
    registry = RunRegistry(results_dir=str(results_dir))
    extractor = MetaFeatureExtractor(configs_dir=Path(__file__).parent.parent / "configs")

    # Lade Features
    features = get_features(registry, extractor, json_path=args.json)

    if not features:
        print("❌ Keine Meta-Features gefunden.")
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
