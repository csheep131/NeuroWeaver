#!/usr/bin/env python3
"""
Reichert bestehende Runs mit Meta-Features an.

Dieses Skript extrahiert Meta-Features aus allen bestehenden Runs
und speichert sie als JSON für weitere Analyse.

Usage:
python -m scripts.enrich_historical_runs --all
python -m scripts.enrich_historical_runs --run-ids run001,run002,run003
python -m scripts.enrich_historical_runs --status completed

Output:
Standardmäßig: results/meta_features.json

Das Output-Format ist eine Liste von Meta-Feature Dictionaries:
[
{
"run_id": "run001",
"features_active": ["gqa", "film"],
"budget_class": "medium",
...
},
...
]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.meta_features import MetaFeatureExtractor, RunMetaFeatures
from core.registry import RunRegistry


def parse_args() -> argparse.Namespace:
"""Parse command-line arguments."""
parser = argparse.ArgumentParser(
description="Extrahiere Meta-Features aus bestehenden Runs",
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog="""
Beispiele:
%(prog)s --all # Alle Runs verarbeiten
%(prog)s --run-ids run001,run002 # Spezifische Runs
%(prog)s --status completed # Nur abgeschlossene Runs
%(prog)s --all --output features.json # Custom Output-Pfad
""",
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
"--all",
action="store_true",
help="Alle Runs im Registry verarbeiten",
)
group.add_argument(
"--run-ids",
type=str,
help="Komma-separierte Liste von Run-IDs (z.B. run001,run002,run003)",
)

parser.add_argument(
"--status",
type=str,
choices=["pending", "running", "completed", "failed", "killed"],
help="Filtere Runs nach Status (nur mit --all)",
)

parser.add_argument(
"--output",
type=str,
default="results/meta_features.json",
help="Output JSON-Pfad (default: results/meta_features.json)",
)

parser.add_argument(
"--verbose",
action="store_true",
help="Ausführliche Ausgabe während der Verarbeitung",
)

parser.add_argument(
"--include-co-occurrence",
action="store_true",
help="Co-occurrence Statistiken im Output einschließen",
)

return parser.parse_args()


def get_run_ids(
registry: RunRegistry,
all_runs: bool = False,
run_ids_str: Optional[str] = None,
status: Optional[str] = None,
) -> List[str]:
"""
Bestimme Liste der zu verarbeitenden Run-IDs.

Args:
registry: RunRegistry Instanz
all_runs: Wenn True, alle Runs verwenden
run_ids_str: Komma-separierte Liste von Run-IDs
status: Optionaler Status-Filter

Returns:
Liste von Run-IDs
"""
if all_runs:
runs = registry.list_runs(status=status)
return [run.run_id for run in runs]
elif run_ids_str:
return [rid.strip() for rid in run_ids_str.split(",")]
else:
return []


def extract_features(
run_ids: List[str],
registry: RunRegistry,
extractor: MetaFeatureExtractor,
verbose: bool = False,
) -> List[RunMetaFeatures]:
"""
Extrahiere Meta-Features für gegebene Runs.

Args:
run_ids: Liste von Run-IDs
registry: RunRegistry Instanz
extractor: MetaFeatureExtractor Instanz
verbose: Ausführliche Ausgabe

Returns:
Liste von RunMetaFeatures Objekten
"""
features_list = []
total = len(run_ids)

for i, run_id in enumerate(run_ids):
if verbose:
print(f"[{i + 1}/{total}] Verarbeite {run_id}...")

try:
features = extractor.extract(run_id, registry)
features_list.append(features)
except ValueError as e:
if verbose:
print(f" Überspringe {run_id}: {e}")
except Exception as e:
if verbose:
print(f" Fehler bei {run_id}: {e}")

return features_list


def print_summary(features: List[RunMetaFeatures]) -> None:
"""Drucke Zusammenfassung der extrahierten Features."""
if not features:
print("Keine Features extrahiert.")
return

print("\n" + "=" * 60)
print("META-FEATURE ZUSAMMENFASSUNG")
print("=" * 60)

print(f"\nTotal Runs: {len(features)}")

# Unique Features
all_features = set()
for f in features:
all_features.update(f.features_active)
print(f"Unique Features: {len(all_features)}")
if all_features:
print(f" Features: {', '.join(sorted(all_features))}")

# Budget Class Distribution
budget_dist = {}
for f in features:
budget_dist[f.budget_class] = budget_dist.get(f.budget_class, 0) + 1
print(f"\nBudget Classes: {budget_dist}")

# Quantization Distribution
quant_dist = {}
for f in features:
quant_dist[f.quantization_type] = quant_dist.get(f.quantization_type, 0) + 1
print(f"Quantization Types: {quant_dist}")

# Sequence Length Distribution
seq_dist = {}
for f in features:
seq_dist[f.sequence_length] = seq_dist.get(f.sequence_length, 0) + 1
print(f"Sequence Lengths: {seq_dist}")

# Lineage Statistics
depths = [f.lineage_depth for f in features]
if depths:
print(f"\nLineage Depth: min={min(depths)}, max={max(depths)}, avg={sum(depths) / len(depths):.2f}")

# Performance Statistics (nur für completed Runs)
completed = [f for f in features if f.delta_bpb_vs_parent != 0.0]
if completed:
deltas = [f.delta_bpb_vs_parent for f in completed]
print(f"\nΔBPB (completed Runs mit Parent):")
print(f" min={min(deltas):.4f}, max={max(deltas):.4f}, avg={sum(deltas) / len(deltas):.4f}")

print("=" * 60)


def save_features(
features: List[RunMetaFeatures],
output_path: str,
include_co_occurrence: bool = False,
) -> None:
"""
Speichere Meta-Features als JSON.

Args:
features: Liste von RunMetaFeatures Objekten
output_path: Output JSON-Pfad
include_co_occurrence: Wenn True, Co-occurrence separat speichern
"""
output_file = Path(output_path)
output_file.parent.mkdir(parents=True, exist_ok=True)

# Konvertiere zu Dicts
data = [f.to_dict() for f in features]

with open(output_file, "w", encoding="utf-8") as f:
json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n Extrahiert {len(features)} Meta-Features → {output_path}")

# Optional: Co-occurrence separat speichern
if include_co_occurrence:
extractor = MetaFeatureExtractor()
co_occ = extractor.compute_co_occurrence(features)

# Konvertiere Tuple-Keys zu Strings für JSON
co_occ_json = {f"{k[0]}|{k[1]}": v for k, v in co_occ.items()}

co_occ_path = output_file.parent / "co_occurrence.json"
with open(co_occ_path, "w", encoding="utf-8") as f:
json.dump(co_occ_json, f, indent=2, ensure_ascii=False)

print(f" Co-occurrence Statistiken → {co_occ_path}")


def main() -> None:
"""Hauptfunktion."""
args = parse_args()

# Initialisiere Registry und Extractor
results_dir = Path(__file__).parent.parent / "results"
registry = RunRegistry(results_dir=str(results_dir))
extractor = MetaFeatureExtractor(configs_dir=Path(__file__).parent.parent / "configs")

# Bestimme Run-IDs
run_ids = get_run_ids(
registry,
all_runs=args.all,
run_ids_str=args.run_ids,
status=args.status,
)

if not run_ids:
print(" Keine Runs gefunden zum Verarbeiten.")
sys.exit(1)

print(f"Verarbeite {len(run_ids)} Runs...")

# Extrahiere Features
features = extract_features(
run_ids,
registry,
extractor,
verbose=args.verbose,
)

if not features:
print(" Keine Features extrahiert.")
sys.exit(1)

# Reichere mit Co-occurrence an wenn gewünscht
if args.include_co_occurrence:
features = extractor.enrich_features_with_co_occurrence(features)

# Drucke Zusammenfassung
print_summary(features)

# Speichere Results
save_features(
features,
args.output,
include_co_occurrence=args.include_co_occurrence,
)


if __name__ == "__main__":
main()
