#!/usr/bin/env python3
"""
Basic Dashboard für Meta-Feature Visualisierung.

Dieses Dashboard bietet Einblicke in die extrahierten Meta-Features
aller Runs und hilft bei der Analyse von Feature-Performance und
Co-occurrence Mustern.

Commands:
    python -m orchestrator.meta_dashboard summary       # Übersicht aller Features
    python -m orchestrator.meta_dashboard co-occurrence # Feature Co-occurrence Matrix
    python -m orchestrator.meta_dashboard feature-stats # Statistiken pro Feature
    python -m orchestrator.meta_dashboard lineage       # Lineage-Analyse
    python -m orchestrator.meta_dashboard budget        # Budget-Klassen Analyse
    python -m orchestrator.meta_dashboard quant         # Quantisierungs-Analyse

Beispiele:
    python -m orchestrator.meta_dashboard summary --top 10
    python -m orchestrator.meta_dashboard feature-stats --min-count 3
    python -m orchestrator.meta_dashboard co-occurrence --limit 20
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
        """,
    )
    
    parser.add_argument(
        "command",
        choices=["summary", "co-occurrence", "feature-stats", "lineage", "budget", "quant"],
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


if __name__ == "__main__":
    main()
