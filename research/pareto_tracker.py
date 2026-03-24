#!/usr/bin/env python3
"""
Pareto Tracker für Phase 4A.

Multi-Objective Frontier Monitoring für BPB vs Efficiency vs Size.

Objectives (alle zu minimieren):
1. ΔBPB (niedriger = besser)
2. 1/Efficiency (höher = besser → minimiere Kehrwert)
3. Size Change (niedriger = besser)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ParetoPoint:
    """Ein Punkt auf der Pareto-Frontier."""
    
    run_id: str
    delta_bpb: float
    efficiency_gain: float
    size_change: float
    is_pareto_optimal: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary."""
        return {
            "run_id": self.run_id,
            "delta_bpb": self.delta_bpb,
            "efficiency_gain": self.efficiency_gain,
            "size_change": self.size_change,
            "is_pareto_optimal": self.is_pareto_optimal,
        }
    
    def dominates(self, other: "ParetoPoint") -> bool:
        """
        Prüfe ob dieser Punkt einen anderen dominiert.
        
        Ein Punkt dominiert einen anderen wenn er in ALLEN Objectives
        besser oder gleich ist und in MINDESTENS EINEM Objective strikt besser.
        
        Args:
            other: Anderer ParetoPoint
            
        Returns:
            True wenn dieser Punkt other dominiert
        """
        # Objectives: niedriger ist besser für alle drei
        at_least_one_better = (
            self.delta_bpb < other.delta_bpb or
            self.efficiency_gain > other.efficiency_gain or  # Höher = besser
            self.size_change < other.size_change
        )
        
        all_at_least_as_good = (
            self.delta_bpb <= other.delta_bpb and
            self.efficiency_gain >= other.efficiency_gain and
            self.size_change <= other.size_change
        )
        
        return at_least_one_better and all_at_least_as_good


class ParetoTracker:
    """
    Tracking der Pareto-Frontier über mehrere Runs.
    
    Objectives (alle zu minimieren):
    1. ΔBPB (niedriger = besser)
    2. 1/Efficiency (höher = besser → minimiere Kehrwert)
    3. Size Change (niedriger = besser)
    
    Attributes:
        points: Alle hinzugefügten Punkte
        frontier_history: Historische Snapshots der Frontier
    """
    
    def __init__(self) -> None:
        """Initialisiere Pareto Tracker."""
        self.points: List[ParetoPoint] = []
        self.frontier_history: List[List[ParetoPoint]] = []
        self._current_frontier: List[ParetoPoint] = []
        self._frontier_volume_history: List[float] = []
    
    def add_run(
        self,
        run_id: str,
        delta_bpb: float,
        efficiency_gain: float,
        size_change: float
    ) -> ParetoPoint:
        """
        Neuen Run hinzufügen.
        
        Args:
            run_id: Eindeutige ID des Runs
            delta_bpb: BPB-Veränderung (negativ = Verbesserung)
            efficiency_gain: Effizienz-Gewinn in Prozent
            size_change: Modell-Größen-Änderung in Prozent
            
        Returns:
            Erstellter ParetoPoint
        """
        point = ParetoPoint(
            run_id=run_id,
            delta_bpb=delta_bpb,
            efficiency_gain=efficiency_gain,
            size_change=size_change,
            is_pareto_optimal=False,  # Wird bei compute_pareto_frontier gesetzt
        )
        
        self.points.append(point)
        
        # Frontier neu berechnen
        self._current_frontier = self._compute_frontier_internal()
        
        return point
    
    def _compute_frontier_internal(self) -> List[ParetoPoint]:
        """
        Interne Frontier-Berechnung ohne History-Update.
        
        Returns:
            Liste der Pareto-optimalen Punkte
        """
        if not self.points:
            return []
        
        pareto_optimal: List[ParetoPoint] = []
        
        for point in self.points:
            is_dominated = False
            
            for other in self.points:
                if other.run_id == point.run_id:
                    continue
                
                if other.dominates(point):
                    is_dominated = True
                    break
            
            if not is_dominated:
                point.is_pareto_optimal = True
                pareto_optimal.append(point)
            else:
                point.is_pareto_optimal = False
        
        return pareto_optimal
    
    def compute_pareto_frontier(self) -> List[ParetoPoint]:
        """
        Aktuelle Pareto-Frontier berechnen.
        
        Ein Punkt ist Pareto-optimal, wenn kein anderer Punkt in ALLEN
        Objectives besser ist.
        
        Returns:
            Liste der Pareto-optimalen Punkte
        """
        self._current_frontier = self._compute_frontier_internal()
        return self._current_frontier.copy()
    
    def get_frontier_points(self) -> List[ParetoPoint]:
        """
        Pareto-optimale Punkte zurückgeben.
        
        Returns:
            Liste der Pareto-optimalen Punkte
        """
        if not self._current_frontier:
            self.compute_pareto_frontier()
        return self._current_frontier.copy()
    
    def get_dominated_points(self) -> List[ParetoPoint]:
        """
        Dominierte Punkte zurückgeben.
        
        Returns:
            Liste der dominierten Punkte
        """
        if not self._current_frontier:
            self.compute_pareto_frontier()
        
        return [p for p in self.points if not p.is_pareto_optimal]
    
    def compute_frontier_volume(self) -> float:
        """
        Volumen der Pareto-Frontier berechnen (Fortschrittsmaß).
        
        Verwendet Hypervolumen-Indikator mit Referenzpunkt.
        
        Returns:
            Hypervolumen der Frontier
        """
        frontier = self.get_frontier_points()
        
        if not frontier:
            return 0.0
        
        # Referenzpunkt (worst-case Werte)
        ref_bpb = max(p.delta_bpb for p in self.points) if self.points else 0.0
        ref_eff = min(p.efficiency_gain for p in self.points) if self.points else 0.0
        
        # Hypervolumen berechnen (2D-Projektion: BPB vs Efficiency)
        volume = 0.0
        
        for point in frontier:
            # Beitrag dieses Punktes zum Hypervolumen
            bpb_contrib = max(0, ref_bpb - point.delta_bpb)
            eff_contrib = max(0, point.efficiency_gain - ref_eff)
            
            volume += bpb_contrib * eff_contrib
        
        return volume
    
    def get_frontier_expansion(self) -> float:
        """
        Expansion der Frontier seit letztem Snapshot.
        
        Returns:
            Relative Expansion in Prozent (0.0 = keine Änderung)
        """
        if not self.frontier_history:
            return 0.0
        
        current_volume = self.compute_frontier_volume()
        previous_volume = self._frontier_volume_history[-1] if self._frontier_volume_history else 0.0
        
        if previous_volume == 0.0:
            return 1.0 if current_volume > 0 else 0.0
        
        expansion = (current_volume - previous_volume) / abs(previous_volume)
        return expansion
    
    def identify_gaps(self, num_gaps: int = 5) -> List[Dict[str, Any]]:
        """
        Lücken in der Frontier identifizieren.
        
        Findet Regionen zwischen Pareto-optimalen Punkten die
        potenziell interessante Trade-offs bieten.
        
        Args:
            num_gaps: Anzahl der zu identifizierenden Lücken
            
        Returns:
            Liste von {"target_bpb": ..., "target_efficiency": ..., "reason": ...}
        """
        frontier = self.get_frontier_points()
        
        if len(frontier) < 2:
            return []
        
        gaps: List[Dict[str, Any]] = []
        
        # Sortiere nach ΔBPB
        sorted_frontier = sorted(frontier, key=lambda p: p.delta_bpb)
        
        # Lücken zwischen benachbarten Punkten finden
        for i in range(len(sorted_frontier) - 1):
            p1 = sorted_frontier[i]
            p2 = sorted_frontier[i + 1]
            
            # Mittelpunkt als Target
            target_bpb = (p1.delta_bpb + p2.delta_bpb) / 2
            target_eff = (p1.efficiency_gain + p2.efficiency_gain) / 2
            
            # Größe der Lücke
            bpb_gap = abs(p2.delta_bpb - p1.delta_bpb)
            eff_gap = abs(p2.efficiency_gain - p1.efficiency_gain)
            
            gap_size = np.sqrt(bpb_gap ** 2 + eff_gap ** 2)
            
            reason = (
                f"Lücke zwischen {p1.run_id} (ΔBPB={p1.delta_bpb:.4f}) und "
                f"{p2.run_id} (ΔBPB={p2.delta_bpb:.4f}). "
                f"Potenzieller Trade-off: {bpb_gap:.4f} BPB vs {eff_gap:.1f}% Effizienz."
            )
            
            gaps.append({
                "target_bpb": target_bpb,
                "target_efficiency": target_eff,
                "target_size_change": (p1.size_change + p2.size_change) / 2,
                "gap_size": gap_size,
                "reason": reason,
                "neighbor_runs": [p1.run_id, p2.run_id],
            })
        
        # Nach Gap-Größe sortieren und Top-N zurückgeben
        gaps.sort(key=lambda g: g["gap_size"], reverse=True)
        return gaps[:num_gaps]
    
    def plot_frontier(
        self,
        output_path: str = "results/pareto_frontier.png",
        show_3d: bool = False
    ) -> str:
        """
        2D/3D Pareto-Frontier plotten.
        
        Args:
            output_path: Pfad zur Ausgabe-Datei
            show_3d: Wenn True, erstelle 3D-Plot
            
        Returns:
            Pfad zur erstellten Plot-Datei
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib ist für Plotting erforderlich. "
                "Installiere mit: pip install matplotlib"
            )
        
        frontier = self.get_frontier_points()
        dominated = self.get_dominated_points()
        
        if not frontier:
            # Leeren Plot erstellen
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "Keine Daten verfügbar", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("ΔBPB")
            ax.set_ylabel("Efficiency Gain (%)")
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return output_path
        
        if show_3d and len(frontier) >= 3:
            # 3D Plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")
            
            # Pareto-optimale Punkte
            ax.scatter(
                [p.delta_bpb for p in frontier],
                [p.efficiency_gain for p in frontier],
                [p.size_change for p in frontier],
                c="green", s=100, label="Pareto-optimal", alpha=0.8
            )
            
            # Dominierte Punkte
            if dominated:
                ax.scatter(
                    [p.delta_bpb for p in dominated],
                    [p.efficiency_gain for p in dominated],
                    [p.size_change for p in dominated],
                    c="red", s=50, label="Dominiert", alpha=0.5
                )
            
            ax.set_xlabel("ΔBPB")
            ax.set_ylabel("Efficiency Gain (%)")
            ax.set_zlabel("Size Change (%)")
            ax.set_title("3D Pareto-Frontier")
            ax.legend()
        else:
            # 2D Plot (BPB vs Efficiency)
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Pareto-optimale Punkte
            ax.scatter(
                [p.delta_bpb for p in frontier],
                [p.efficiency_gain for p in frontier],
                c="green", s=150, label="Pareto-optimal", 
                alpha=0.8, edgecolors="darkgreen", linewidths=2
            )
            
            # Dominierte Punkte
            if dominated:
                ax.scatter(
                    [p.delta_bpb for p in dominated],
                    [p.efficiency_gain for p in dominated],
                    c="red", s=80, label="Dominiert", 
                    alpha=0.5, edgecolors="darkred", linewidths=1
                )
            
            # Punkte beschriften
            for p in frontier:
                ax.annotate(
                    p.run_id[-8:],  # Letzte 8 Zeichen der ID
                    (p.delta_bpb, p.efficiency_gain),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )
            
            ax.set_xlabel("ΔBPB (niedriger = besser)", fontsize=12)
            ax.set_ylabel("Efficiency Gain % (höher = besser)", fontsize=12)
            ax.set_title("Pareto-Frontier: BPB vs Efficiency", fontsize=14)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            # Vertikale Linie bei ΔBPB = 0
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        
        # Speichern
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def snapshot(self) -> None:
        """
        Snapshot der aktuellen Frontier für History-Tracking.
        
        Erstellt eine Kopie der aktuellen Frontier und speichert
        sie in frontier_history.
        """
        frontier = self.get_frontier_points()
        
        # Deep copy der Punkte
        snapshot_points = [
            ParetoPoint(
                run_id=p.run_id,
                delta_bpb=p.delta_bpb,
                efficiency_gain=p.efficiency_gain,
                size_change=p.size_change,
                is_pareto_optimal=p.is_pareto_optimal,
            )
            for p in frontier
        ]
        
        self.frontier_history.append(snapshot_points)
        self._frontier_volume_history.append(self.compute_frontier_volume())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Statistik über die Frontier berechnen.
        
        Returns:
            Dictionary mit Statistiken
        """
        frontier = self.get_frontier_points()
        
        if not frontier:
            return {
                "total_points": 0,
                "pareto_optimal_count": 0,
                "frontier_volume": 0.0,
            }
        
        return {
            "total_points": len(self.points),
            "pareto_optimal_count": len(frontier),
            "frontier_volume": self.compute_frontier_volume(),
            "avg_delta_bpb": np.mean([p.delta_bpb for p in frontier]),
            "avg_efficiency_gain": np.mean([p.efficiency_gain for p in frontier]),
            "best_delta_bpb": min(p.delta_bpb for p in frontier),
            "best_efficiency_gain": max(p.efficiency_gain for p in frontier),
            "frontier_expansion": self.get_frontier_expansion(),
        }
    
    def to_json(self) -> str:
        """
        Frontier-Daten als JSON exportieren.
        
        Returns:
            JSON-String mit allen Punkten und Statistiken
        """
        data = {
            "points": [p.to_dict() for p in self.points],
            "frontier": [p.to_dict() for p in self.get_frontier_points()],
            "statistics": self.get_statistics(),
        }
        
        return json.dumps(data, indent=2)
    
    def load_from_runs(self, runs: List[Dict[str, Any]]) -> int:
        """
        Lade Runs aus Dictionary-Liste.
        
        Args:
            runs: Liste von Run-Dictionaries mit Keys:
                  run_id, delta_bpb, efficiency_gain, size_change
            
        Returns:
            Anzahl geladener Runs
        """
        count = 0
        for run in runs:
            try:
                self.add_run(
                    run_id=run["run_id"],
                    delta_bpb=run["delta_bpb"],
                    efficiency_gain=run["efficiency_gain"],
                    size_change=run["size_change"],
                )
                count += 1
            except (KeyError, TypeError):
                continue
        
        return count
