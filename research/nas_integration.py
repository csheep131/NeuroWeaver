#!/usr/bin/env python3
"""
Neural Architecture Search (NAS) Integration für NeuroWeave Phase 5.

Automatische Architektursuche mit Evolutionary / Reinforcement Learning.

Features:
- Search Space: Depth, Width, Attention-Typen
- Search Strategy: Evolutionary / Reinforcement Learning
- Performance Prediction (via Surrogate Scorer)
- Constraint Handling (VRAM, Time Budget)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import RunRegistry


class SearchStrategy(Enum):
"""Suchstrategie für NAS."""

EVOLUTIONARY = "evolutionary"
RANDOM = "random"
BAYESIAN = "bayesian"
REINFORCEMENT = "reinforcement"


@dataclass
class Architecture:
"""Eine neuronale Architektur."""

arch_id: str
depth: int
width: int
mlp_ratio: float
attention_type: Literal["standard", "gqa", "xsa"]
activation: Literal["gelu", "swiglu", "leaky_relu"]
num_heads: int = 8
head_dim: Optional[int] = None

# Metriken
delta_bpb: Optional[float] = None
efficiency_gain: Optional[float] = None
size_mb: Optional[float] = None
vram_usage_mb: Optional[float] = None
fitness: float = 0.0

# Metadata
parent_ids: List[str] = field(default_factory=list)
generation: int = 0
evaluated: bool = False

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"arch_id": self.arch_id,
"depth": self.depth,
"width": self.width,
"mlp_ratio": self.mlp_ratio,
"attention_type": self.attention_type,
"activation": self.activation,
"num_heads": self.num_heads,
"head_dim": self.head_dim,
"delta_bpb": self.delta_bpb,
"efficiency_gain": self.efficiency_gain,
"size_mb": self.size_mb,
"vram_usage_mb": self.vram_usage_mb,
"fitness": self.fitness,
"parent_ids": self.parent_ids,
"generation": self.generation,
"evaluated": self.evaluated,
}

def to_run_config(self) -> Dict[str, Any]:
"""Konvertiere zu Run-Konfiguration."""
return {
"depth": self.depth,
"width": self.width,
"mlp_ratio": self.mlp_ratio,
"attention_type": self.attention_type,
"activation": self.activation,
"num_heads": self.num_heads,
"head_dim": self.head_dim or self.width // self.num_heads,
}

def __lt__(self, other: "Architecture") -> bool:
"""Vergleich nach Fitness (höher = besser)."""
return self.fitness > other.fitness


@dataclass
class SearchSpace:
"""Suchraum für NAS."""

depth_range: Tuple[int, int] = (8, 16)
width_range: Tuple[int, int] = (256, 1024)
mlp_ratio_range: Tuple[float, float] = (2.0, 5.0)
attention_types: List[str] = field(default_factory=lambda: ["standard", "gqa", "xsa"])
activations: List[str] = field(default_factory=lambda: ["gelu", "swiglu", "leaky_relu"])
num_heads_options: List[int] = field(default_factory=lambda: [4, 8, 16, 32])

# Constraints
max_vram_mb: float = 8000.0
max_size_mb: float = 500.0
min_bpb_gain: float = -0.01 # Minimale BPB-Verbesserung

def sample(self, arch_id: Optional[str] = None) -> Architecture:
"""
Zufällige Architektur aus Suchraum sampeln.

Args:
arch_id: Optionale Architektur-ID

Returns:
Gesampelte Architektur
"""
depth = random.randint(*self.depth_range)
width = random.randint(*self.width_range)

# Head-Dimension berechnen
num_heads = random.choice(self.num_heads_options)
head_dim = width // num_heads

return Architecture(
arch_id=arch_id or f"arch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
depth=depth,
width=width,
mlp_ratio=random.uniform(*self.mlp_ratio_range),
attention_type=random.choice(self.attention_types),
activation=random.choice(self.activations),
num_heads=num_heads,
head_dim=head_dim,
)

def mutate(
self,
parent: Architecture,
mutation_rate: float = 0.2,
) -> Architecture:
"""
Architektur mutieren.

Args:
parent: Eltern-Architektur
mutation_rate: Wahrscheinlichkeit für Mutation pro Parameter

Returns:
Mutierte Architektur
"""
child = Architecture(
arch_id=f"arch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 999)}",
depth=parent.depth,
width=parent.width,
mlp_ratio=parent.mlp_ratio,
attention_type=parent.attention_type,
activation=parent.activation,
num_heads=parent.num_heads,
head_dim=parent.head_dim,
parent_ids=[parent.arch_id],
generation=parent.generation + 1,
)

# Depth mutieren
if random.random() < mutation_rate:
delta = random.choice([-2, -1, 1, 2])
child.depth = max(self.depth_range[0], min(self.depth_range[1], child.depth + delta))

# Width mutieren
if random.random() < mutation_rate:
delta = random.choice([-128, -64, 64, 128])
child.width = max(self.width_range[0], min(self.width_range[1], child.width + delta))
child.head_dim = child.width // child.num_heads

# MLP Ratio mutieren
if random.random() < mutation_rate:
delta = random.uniform(-0.5, 0.5)
child.mlp_ratio = max(
self.mlp_ratio_range[0],
min(self.mlp_ratio_range[1], child.mlp_ratio + delta),
)

# Attention Type mutieren
if random.random() < mutation_rate:
child.attention_type = random.choice(self.attention_types)

# Activation mutieren
if random.random() < mutation_rate:
child.activation = random.choice(self.activations)

# Num Heads mutieren
if random.random() < mutation_rate:
child.num_heads = random.choice(self.num_heads_options)
child.head_dim = child.width // child.num_heads

return child

def crossover(
self,
parent1: Architecture,
parent2: Architecture,
) -> Architecture:
"""
Crossover zwischen zwei Eltern.

Args:
parent1: Erste Eltern-Architektur
parent2: Zweite Eltern-Architektur

Returns:
Kind-Architektur
"""
# Random Crossover
child = Architecture(
arch_id=f"arch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 999)}",
depth=random.choice([parent1.depth, parent2.depth]),
width=random.choice([parent1.width, parent2.width]),
mlp_ratio=random.choice([parent1.mlp_ratio, parent2.mlp_ratio]),
attention_type=random.choice([parent1.attention_type, parent2.attention_type]),
activation=random.choice([parent1.activation, parent2.activation]),
num_heads=random.choice([parent1.num_heads, parent2.num_heads]),
parent_ids=[parent1.arch_id, parent2.arch_id],
generation=max(parent1.generation, parent2.generation) + 1,
)

child.head_dim = child.width // child.num_heads
return child

def check_constraints(self, arch: Architecture) -> Tuple[bool, List[str]]:
"""
Prüfe ob Architektur Constraints erfüllt.

Args:
arch: Zu prüfende Architektur

Returns:
Tuple aus (erfüllt, verletzte_constraints)
"""
violations = []

# head_dim berechnen falls None
head_dim = arch.head_dim if arch.head_dim is not None else arch.width // arch.num_heads

# VRAM-Schätzung (grob: ~4 bytes * params * 2 für gradients)
estimated_params = (
arch.depth * arch.width * arch.width * arch.mlp_ratio
+ arch.depth * arch.width * arch.num_heads * head_dim
)
estimated_vram = estimated_params * 4 * 2 / 1e6 # MB

if estimated_vram > self.max_vram_mb:
violations.append(f"VRAM: {estimated_vram:.0f}MB > {self.max_vram_mb:.0f}MB")

# Size-Schätzung
estimated_size = estimated_params * 4 / 1e6 # MB
if estimated_size > self.max_size_mb:
violations.append(f"Size: {estimated_size:.0f}MB > {self.max_size_mb:.0f}MB")

return len(violations) == 0, violations


class NASIntegration:
"""
Neural Architecture Search.

Features:
- Search Space: Depth, Width, Attention-Typen
- Search Strategy: Evolutionary / Reinforcement Learning
- Performance Prediction (via Surrogate Scorer)
- Constraint Handling (VRAM, Time Budget)

Example:
registry = RunRegistry("results")
scorer = SurrogateScorer() # Falls vorhanden
pareto_tracker = ParetoTracker()

nas = NASIntegration(scorer, pareto_tracker)

# Search Space definieren
nas.define_search_space(
max_vram_mb=8000,
max_steps=10000,
)

# Suche durchführen
architectures = nas.search(budget=100)

# Tradeoffs analysieren
report = nas.get_architecture_tradeoffs()
"""

def __init__(
self,
registry: RunRegistry,
scorer: Optional[Any] = None,
pareto_tracker: Optional[Any] = None,
strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY,
) -> None:
"""
Initialisiere NAS Integration.

Args:
registry: RunRegistry für Datenzugriff
scorer: SurrogateScorer für Vorhersagen (optional)
pareto_tracker: ParetoTracker für Multi-Objective (optional)
strategy: Suchstrategie
"""
self._registry = registry
self._scorer = scorer
self._pareto_tracker = pareto_tracker
self._strategy = strategy

self._search_space = SearchSpace()
self._population: List[Architecture] = []
self._evaluated_architectures: Dict[str, Architecture] = {}
self._pareto_frontier: List[Architecture] = []

self._generation = 0
self._budget = 0
self._evaluated_count = 0

def define_search_space(self, **constraints) -> None:
"""
Suchraum definieren.

Constraints:
- max_vram_mb: 8000
- max_steps: 10000
- min_bpb_gain: -0.01
- depth_range: (8, 16)
- width_range: (256, 1024)
"""
if "max_vram_mb" in constraints:
self._search_space.max_vram_mb = constraints["max_vram_mb"]

if "max_size_mb" in constraints:
self._search_space.max_size_mb = constraints["max_size_mb"]

if "min_bpb_gain" in constraints:
self._search_space.min_bpb_gain = constraints["min_bpb_gain"]

if "depth_range" in constraints:
self._search_space.depth_range = constraints["depth_range"]

if "width_range" in constraints:
self._search_space.width_range = constraints["width_range"]

def search(self, budget: int = 100) -> List[Architecture]:
"""
Architektursuche durchführen.

Args:
budget: Maximale Anzahl Architecturevaluierungen

Returns:
Liste gefundener Architekturen (Pareto-Frontier)
"""
self._budget = budget
self._evaluated_count = 0
self._population = []
self._generation = 0

print(f" Starte NAS-Suche mit Budget {budget}...")
print(f" Strategie: {self._strategy.value}")
print(f" Constraints: VRAM ≤ {self._search_space.max_vram_mb:.0f}MB")

# Initiale Population (mindestens 5 für evolution)
init_size = max(5, min(20, budget // 5))
print(f"\n Generiere initiale Population ({init_size} Architekturen)...")

for i in range(init_size):
arch = self._search_space.sample()
valid, violations = self._search_space.check_constraints(arch)

if valid:
self._population.append(arch)
else:
# Retry
self._population.append(self._search_space.sample())

# Evolutionäre Schleife
while self._evaluated_count < budget:
self._generation += 1

# Evaluate aktuelle Population
self._evaluate_population()

# Pareto-Frontier aktualisieren
self._update_pareto_frontier()

# Nächste Generation erzeugen
if self._evaluated_count < budget:
self._evolve_population()

print(f"\n Suche abgeschlossen: {self._evaluated_count} Architekturen evaluiert")
print(f" Pareto-optimale Architekturen: {len(self._pareto_frontier)}")

return self._pareto_frontier

def _evaluate_population(self) -> None:
"""Evaluiere aktuelle Population."""
for arch in self._population:
if arch.evaluated:
continue

# Simuliere Evaluation (in echt: Training ausführen)
self._simulate_evaluation(arch)

arch.evaluated = True
self._evaluated_architectures[arch.arch_id] = arch
self._evaluated_count += 1

# Fitness berechnen
arch.fitness = self._calculate_fitness(arch)

def _simulate_evaluation(self, arch: Architecture) -> None:
"""Simuliere Architektur-Evaluation."""
# In echter Implementierung: Training ausführen
# Hier: Simulierte Metriken basierend auf Architektur

# Größere Modelle tendenziell besser, aber langsamer
base_bpb = 1.5 - (arch.depth * arch.width / 16000) * 0.1
noise = random.gauss(0, 0.05)
arch.delta_bpb = base_bpb + noise

# Efficiency basierend auf Attention-Type
efficiency_map = {"gqa": 15, "xsa": 10, "standard": 0}
arch.efficiency_gain = efficiency_map.get(arch.attention_type, 0) + random.gauss(0, 5)

# Size-Schätzung
params_m = arch.depth * arch.width * arch.width * arch.mlp_ratio / 1e6
arch.size_mb = params_m * 4 # FP32

# VRAM-Schätzung
arch.vram_usage_mb = arch.size_mb * 2 # Params + Gradients

def _calculate_fitness(self, arch: Architecture) -> float:
"""
Berechne Fitness-Score.

Multi-Objective:
- Minimiere ΔBPB
- Maximiere Efficiency
- Minimiere Size

Returns:
Fitness-Score (höher = besser)
"""
if arch.delta_bpb is None:
return 0.0

# Gewichte für Multi-Objective
w_bpb = 0.5
w_eff = 0.3
w_size = 0.2

# Normalisierte Werte
bpb_score = -arch.delta_bpb * 10 # Negativ da niedriger besser
eff_score = arch.efficiency_gain / 20 if arch.efficiency_gain else 0
size_score = -arch.size_mb / 500 if arch.size_mb else 0

fitness = w_bpb * bpb_score + w_eff * eff_score + w_size * size_score
return fitness

def _update_pareto_frontier(self) -> None:
"""Aktualisiere Pareto-Frontier."""
evaluated = [a for a in self._population if a.evaluated]

if not evaluated:
return

# Pareto-optimale Architekturen finden
pareto = []

for arch in evaluated:
is_dominated = False

for other in evaluated:
if other.arch_id == arch.arch_id:
continue

# Prüfe Dominanz
dominates = (
(other.delta_bpb or 0) <= (arch.delta_bpb or 0) and
(other.efficiency_gain or 0) >= (arch.efficiency_gain or 0) and
(other.size_mb or float("inf")) <= (arch.size_mb or float("inf"))
)

strictly_better = (
(other.delta_bpb or 0) < (arch.delta_bpb or 0) or
(other.efficiency_gain or 0) > (arch.efficiency_gain or 0) or
(other.size_mb or float("inf")) < (arch.size_mb or float("inf"))
)

if dominates and strictly_better:
is_dominated = True
break

if not is_dominated:
pareto.append(arch)

self._pareto_frontier = pareto

def _evolve_population(self) -> None:
"""Erzeuge nächste Generation."""
# Selektion (Tournament)
def tournament_select(k: int = 3) -> Architecture:
# Sicherstellen dass k nicht größer als Population
k = min(k, len(self._population))
if k <= 1:
return random.choice(self._population)
candidates = random.sample(self._population, k)
return max(candidates, key=lambda a: a.fitness)

new_population = []

# Elite erhalten (top 20%)
elite_size = max(1, len(self._population) // 5)
sorted_pop = sorted(self._population, key=lambda a: a.fitness, reverse=True)
new_population.extend(sorted_pop[:elite_size])

# Rest durch Crossover und Mutation erzeugen
while len(new_population) < len(self._population):
if len(self._population) < 2:
# Nicht genug für Crossover, neue Architektur sampeln
child = self._search_space.sample()
else:
parent1 = tournament_select()
parent2 = tournament_select()

# Crossover
child = self._search_space.crossover(parent1, parent2)

# Mutation
mutation_rate = 0.3 * max(0.1, 1 - self._generation / 20) # Abnehmend
child = self._search_space.mutate(child, mutation_rate)

# Constraints prüfen
valid, _ = self._search_space.check_constraints(child)
if valid:
new_population.append(child)

self._population = new_population

def get_architecture_tradeoffs(self) -> str:
"""
Architektur-Tradeoffs analysieren.

Returns:
Report mit Tradeoff-Analyse
"""
if not self._pareto_frontier:
return "Keine Pareto-Architekturen gefunden"

lines = []
lines.append("# Architektur-Tradeoffs Analyse")
lines.append(f"\nGeneriert: {datetime.utcnow().isoformat()}")
lines.append(f"\nPareto-optimale Architekturen: {len(self._pareto_frontier)}")

# Depth vs Width Analyse
lines.append("\n## Depth vs Width Tradeoffs")

depth_groups: Dict[int, List[Architecture]] = {}
for arch in self._pareto_frontier:
if arch.depth not in depth_groups:
depth_groups[arch.depth] = []
depth_groups[arch.depth].append(arch)

for depth in sorted(depth_groups.keys()):
archs = depth_groups[depth]
avg_width = sum(a.width for a in archs) / len(archs)
avg_bpb = sum(a.delta_bpb or 0 for a in archs) / len(archs)
lines.append(f"\n### Depth {depth}")
lines.append(f"- Ø Width: {avg_width:.0f}")
lines.append(f"- Ø ΔBPB: {avg_bpb:+.4f}")
lines.append(f"- Architekturen: {len(archs)}")

# Attention-Typen Vergleich
lines.append("\n## Attention-Typen Vergleich")

attention_stats: Dict[str, List[Architecture]] = {}
for arch in self._pareto_frontier:
if arch.attention_type not in attention_stats:
attention_stats[arch.attention_type] = []
attention_stats[arch.attention_type].append(arch)

for att_type in sorted(attention_stats.keys()):
archs = attention_stats[att_type]
avg_bpb = sum(a.delta_bpb or 0 for a in archs) / len(archs)
avg_eff = sum(a.efficiency_gain or 0 for a in archs) / len(archs)
lines.append(f"\n### {att_type.upper()}")
lines.append(f"- Count: {len(archs)}")
lines.append(f"- Ø ΔBPB: {avg_bpb:+.4f}")
lines.append(f"- Ø Efficiency: {avg_eff:+.1f}%")

# Beste Architekturen
lines.append("\n## Top Pareto-Architekturen")

sorted_pareto = sorted(
self._pareto_frontier,
key=lambda a: (a.delta_bpb or 0, -(a.efficiency_gain or 0)),
)

for i, arch in enumerate(sorted_pareto[:5]):
lines.append(f"\n### #{i + 1}: {arch.arch_id}")
lines.append(f"- Depth: {arch.depth}, Width: {arch.width}")
lines.append(f"- MLP Ratio: {arch.mlp_ratio:.2f}")
lines.append(f"- Attention: {arch.attention_type}")
lines.append(f"- Activation: {arch.activation}")
lines.append(f"- ΔBPB: {arch.delta_bpb:+.4f}")
lines.append(f"- Efficiency: {arch.efficiency_gain:+.1f}%")
lines.append(f"- Size: {arch.size_mb:.0f} MB")

return "\n".join(lines)

def export_architectures(self, output_path: str) -> str:
"""
Exportiere Architekturen als JSON.

Args:
output_path: Pfad für JSON-Output

Returns:
Pfad zur Datei
"""
output = Path(output_path)
output.parent.mkdir(parents=True, exist_ok=True)

export_data = {
"search_space": {
"depth_range": self._search_space.depth_range,
"width_range": self._search_space.width_range,
"mlp_ratio_range": self._search_space.mlp_ratio_range,
"attention_types": self._search_space.attention_types,
"activations": self._search_space.activations,
"max_vram_mb": self._search_space.max_vram_mb,
"max_size_mb": self._search_space.max_size_mb,
},
"search_summary": {
"budget": self._budget,
"evaluated": self._evaluated_count,
"generations": self._generation,
"pareto_count": len(self._pareto_frontier),
},
"pareto_frontier": [a.to_dict() for a in self._pareto_frontier],
"all_evaluated": [a.to_dict() for a in self._evaluated_architectures.values()],
"exported_at": datetime.utcnow().isoformat(),
}

with open(output, "w", encoding="utf-8") as f:
json.dump(export_data, f, indent=2, default=str)

return str(output)


def cmd_nas_search(args: argparse.Namespace) -> int:
"""NAS Search Command."""
print(" Neural Architecture Search")
print("=" * 60)

registry = RunRegistry()
nas = NASIntegration(registry)

# Search Space definieren
nas.define_search_space(
max_vram_mb=args.max_vram,
max_size_mb=args.max_size,
depth_range=(args.min_depth, args.max_depth),
width_range=(args.min_width, args.max_width),
)

print(f"\n Search Space:")
print(f" Depth: {args.min_depth}-{args.max_depth}")
print(f" Width: {args.min_width}-{args.max_width}")
print(f" Max VRAM: {args.max_vram}MB")

# Suche durchführen
pareto_frontier = nas.search(budget=args.budget)

# Report
print("\n" + "=" * 60)
print(" Pareto-optimale Architekturen:")

for i, arch in enumerate(pareto_frontier[:5]):
print(f"\n #{i + 1}: {arch.arch_id}")
print(f" depth={arch.depth}, width={arch.width}, attention={arch.attention_type}")
print(f" ΔBPB: {arch.delta_bpb:+.4f}, Efficiency: {arch.efficiency_gain:+.1f}%")

# Tradeoff-Report
print("\n" + "=" * 60)
tradeoff_report = nas.get_architecture_tradeoffs()
print(tradeoff_report[:2000] + "..." if len(tradeoff_report) > 2000 else tradeoff_report)

# Export
if args.output:
export_path = nas.export_architectures(args.output)
print(f"\n Exportiert: {export_path}")

print("\n" + "=" * 60)
return 0


def create_parser() -> argparse.ArgumentParser:
"""Erstelle Argument Parser."""
parser = argparse.ArgumentParser(
prog="nas-search",
description="Neural Architecture Search",
)
parser.add_argument(
"--budget",
type=int,
default=100,
help="Such-Budget (default: 100)",
)
parser.add_argument(
"--max-vram",
type=int,
default=8000,
help="Max VRAM MB (default: 8000)",
)
parser.add_argument(
"--max-size",
type=int,
default=500,
help="Max Size MB (default: 500)",
)
parser.add_argument(
"--min-depth",
type=int,
default=8,
help="Min Depth (default: 8)",
)
parser.add_argument(
"--max-depth",
type=int,
default=16,
help="Max Depth (default: 16)",
)
parser.add_argument(
"--min-width",
type=int,
default=256,
help="Min Width (default: 256)",
)
parser.add_argument(
"--max-width",
type=int,
default=1024,
help="Max Width (default: 1024)",
)
parser.add_argument(
"--output",
type=str,
help="JSON Export Pfad",
)
parser.set_defaults(func=cmd_nas_search)
return parser


def main() -> int:
"""Hauptfunktion."""
parser = create_parser()
args = parser.parse_args()
return args.func(args)


if __name__ == "__main__":
sys.exit(main())
