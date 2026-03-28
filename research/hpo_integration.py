#!/usr/bin/env python3
"""
Hyperparameter Optimizer für NeuroWeave Phase 5.

Optuna-Integration für automatische Hyperparameter-Optimierung.

Features:
- Bayesian Optimization für Run-Configs
- Multi-Objective (BPB, Efficiency, Size)
- Pruning (unpromising Runs früh stoppen)
- Transfer Learning (von ähnlichen Runs lernen)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import RunRegistry

try:
    import optuna
    from optuna.study import StudyDirection
    from optuna.trial import Trial, TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warnung: Optuna nicht installiert. Installiere mit: pip install optuna")
    # Mock classes for when optuna is not available
    class TrialState:
        RUNNING = "RUNNING"
        COMPLETE = "COMPLETE"
        PRUNED = "PRUNED"
        FAIL = "FAIL"
    class StudyDirection:
        MINIMIZE = "MINIMIZE"
        MAXIMIZE = "MAXIMIZE"

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class TrialResult:
    """Ergebnis eines Trials."""

    trial_number: int
    run_id: str
    params: Dict[str, Any]
    delta_bpb: Optional[float] = None
    efficiency_gain: Optional[float] = None
    size_change: Optional[float] = None
    state: str = "RUNNING"
    value: Optional[float] = None
    values: Optional[List[float]] = None
    datetime_start: Optional[datetime] = None
    datetime_complete: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary."""
        return {
            "trial_number": self.trial_number,
            "run_id": self.run_id,
            "params": self.params,
            "delta_bpb": self.delta_bpb,
            "efficiency_gain": self.efficiency_gain,
            "size_change": self.size_change,
            "state": self.state.name,
            "value": self.value,
            "values": self.values,
            "datetime_start": self.datetime_start.isoformat() if self.datetime_start else None,
            "datetime_complete": self.datetime_complete.isoformat() if self.datetime_complete else None,
        }


class HyperparameterOptimizer:
    """
    AutoML Integration (Optuna).

    Features:
    - Bayesian Optimization für Run-Configs
    - Multi-Objective (BPB, Efficiency, Size)
    - Pruning (unpromising Runs früh stoppen)
    - Transfer Learning (von ähnlichen Runs lernen)

    Example:
        registry = RunRegistry("results")
        scorer = SurrogateScorer()  # Falls vorhanden
        optimizer = HyperparameterOptimizer(registry, scorer)

        # Config vorschlagen lassen
        config = optimizer.suggest_config(trial_number=0)

        # Training ausführen und Ergebnis reporten
        metrics = run_training(config)
        optimizer.report_result(trial_number=0, metrics=metrics)

        # Beste Configs holen
        best_configs = optimizer.get_best_configs(top_k=5)
    """

    def __init__(
        self,
        registry: RunRegistry,
        scorer: Optional[Any] = None,
        study_name: str = "neuro weaver_hpo",
        storage: Optional[str] = None,
        multi_objective: bool = True,
    ) -> None:
        """
        Initialisiere Hyperparameter Optimizer.

        Args:
            registry: RunRegistry für Datenzugriff
            scorer: SurrogateScorer für Vorhersagen (optional)
            study_name: Name der Optuna Study
            storage: Optuna Storage URL (optional)
            multi_objective: Multi-Objective Optimization
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna ist erforderlich. Installiere mit: pip install optuna"
            )

        self._registry = registry
        self._scorer = scorer
        self._study_name = study_name
        self._multi_objective = multi_objective

        # Study erstellen
        if multi_objective:
            self._study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                directions=[
                    StudyDirection.MINIMIZE,  # delta_bpb (niedriger = besser)
                    StudyDirection.MAXIMIZE,  # efficiency_gain
                    StudyDirection.MINIMIZE,  # size_change
                ],
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            )
        else:
            self._study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=StudyDirection.MINIMIZE,
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            )

        # Search Space Definition
        self._search_space = {
            "depth": (8, 16),
            "width": (256, 1024),
            "mlp_ratio": (2.0, 5.0),
            "learning_rate": (1e-5, 1e-2),
            "weight_decay": (0.01, 0.1),
            "attention_type": ["standard", "gqa", "xsa"],
            "activation": ["gelu", "swiglu", "leaky_relu"],
        }

        # Trial-Ergebnisse
        self._trial_results: Dict[int, TrialResult] = {}
        self._run_to_trial: Dict[str, int] = {}

        # Transfer Learning Cache
        self._similar_runs_cache: Dict[str, List[Dict]] = {}

    def suggest_config(self, trial_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Nächste Config vorschlagen.

        Optimiert:
        - depth (8-16)
        - width (256-1024)
        - mlp_ratio (2-5)
        - learning_rate (1e-5 - 1e-2)
        - weight_decay (0.01 - 0.1)

        Args:
            trial_number: Trial-Nummer (optional)

        Returns:
            Dictionary mit Config-Parametern
        """
        if trial_number is not None:
            # Bestehendes Trial verwenden
            if trial_number < len(self._study.trials):
                trial = self._study.trials[trial_number]
            else:
                trial = self._study.ask()
        else:
            trial = self._study.ask()

        # Parameter vorschlagen
        config = {}

        # Integer-Parameter
        config["depth"] = trial.suggest_int("depth", *self._search_space["depth"])
        config["width"] = trial.suggest_int("width", *self._search_space["width"])

        # Float-Parameter
        config["mlp_ratio"] = trial.suggest_float(
            "mlp_ratio",
            *self._search_space["mlp_ratio"],
            log=False,
        )
        config["learning_rate"] = trial.suggest_float(
            "learning_rate",
            *self._search_space["learning_rate"],
            log=True,
        )
        config["weight_decay"] = trial.suggest_float(
            "weight_decay",
            *self._search_space["weight_decay"],
            log=True,
        )

        # Categorical-Parameter
        config["attention_type"] = trial.suggest_categorical(
            "attention_type",
            self._search_space["attention_type"],
        )
        config["activation"] = trial.suggest_categorical(
            "activation",
            self._search_space["activation"],
        )

        # Run-ID generieren
        run_id = f"hpo_trial_{trial.number:03d}"
        config["run_id"] = run_id

        # Speichere Mapping
        self._run_to_trial[run_id] = trial.number

        # Trial-Result erstellen
        self._trial_results[trial.number] = TrialResult(
            trial_number=trial.number,
            run_id=run_id,
            params=config,
            datetime_start=datetime.utcnow(),
        )

        return config

    def report_result(
        self,
        trial_number: int,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Ergebnis reporten.

        Args:
            trial_number: Trial-ID
            metrics: {"delta_bpb": ..., "efficiency": ..., "size": ...}
        """
        if trial_number not in self._trial_results:
            raise ValueError(f"Trial {trial_number} nicht gefunden")

        trial_result = self._trial_results[trial_number]

        # Metriken extrahieren
        delta_bpb = metrics.get("delta_bpb", metrics.get("val_bpb", 0.0))
        efficiency_gain = metrics.get("efficiency_gain", 0.0)
        size_change = metrics.get("size_change", 0.0)

        # Update Trial-Result
        trial_result.delta_bpb = delta_bpb
        trial_result.efficiency_gain = efficiency_gain
        trial_result.size_change = size_change
        trial_result.datetime_complete = datetime.utcnow()

        if self._multi_objective:
            # Multi-Objective Werte
            values = [delta_bpb, efficiency_gain, size_change]
            trial_result.values = values

            # Report an Optuna
            self._study.tell(
                trial_number,
                values,
                state=TrialState.COMPLETE,
            )
        else:
            # Single-Objective (nur delta_bpb)
            trial_result.value = delta_bpb

            self._study.tell(
                trial_number,
                delta_bpb,
                state=TrialState.COMPLETE,
            )

        # In Registry speichern
        run_id = trial_result.run_id
        self._registry.complete_run(run_id, metrics)

    def prune_trial(self, trial_number: int, reason: str = "poor_performance") -> bool:
        """
        Trial frühzeitig stoppen (Pruning).

        Args:
            trial_number: Trial-ID
            reason: Grund für Pruning

        Returns:
            True wenn gepruned
        """
        if trial_number not in self._trial_results:
            return False

        trial_result = self._trial_results[trial_number]
        trial_result.state = TrialState.PRUNED

        self._study.tell(
            trial_number,
            None,
            state=TrialState.PRUNED,
        )

        # Run in Registry als gekillt markieren
        self._registry.kill_run(trial_result.run_id, reason=f"HPO Pruning: {reason}")

        return True

    def get_best_configs(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Beste Configs zurückgeben.

        Args:
            top_k: Anzahl der Top-Configs

        Returns:
            Liste der besten Configs
        """
        if self._multi_objective:
            # Pareto-Front für Multi-Objective
            pareto_trials = self._study.best_trials

            best_configs = []
            for trial in pareto_trials[:top_k]:
                config = dict(trial.params)
                config["run_id"] = f"hpo_trial_{trial.number:03d}"
                config["values"] = trial.values
                best_configs.append(config)

            return best_configs
        else:
            # Single-Objective
            best_trial = self._study.best_trial

            return [
                {
                    **best_trial.params,
                    "run_id": f"hpo_trial_{best_trial.number:03d}",
                    "value": best_trial.value,
                }
            ]

    def get_trial_history(self) -> List[TrialResult]:
        """
        Historie aller Trials.

        Returns:
            Liste von Trial-Results
        """
        return list(self._trial_results.values())

    def get_optimization_progress(self) -> Dict[str, Any]:
        """
        Fortschritt der Optimierung.

        Returns:
            Dictionary mit Progress-Informationen
        """
        trials = self._study.trials

        completed = sum(1 for t in trials if t.state == TrialState.COMPLETE)
        pruned = sum(1 for t in trials if t.state == TrialState.PRUNED)
        running = sum(1 for t in trials if t.state == TrialState.RUNNING)

        # Beste Werte
        if self._multi_objective:
            best_values = None
            if self._study.best_trials:
                best_values = [t.values for t in self._study.best_trials]
        else:
            best_values = self._study.best_value if self._study.best_value else None

        return {
            "total_trials": len(trials),
            "completed": completed,
            "pruned": pruned,
            "running": running,
            "best_values": best_values,
            "study_name": self._study_name,
        }

    def create_study_visualization(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Optuna Study-Visualisierung.

        Zeigt:
        - Optimization History
        - Parameter Importances
        - Parallel Coordinate Plot
        - Contour Plot

        Args:
            output_path: Pfad für HTML-Output (optional)

        Returns:
            Pfad zur generierten HTML-Datei
        """
        if not OPTUNA_AVAILABLE:
            return ""

        plots_dir = Path(__file__).parent.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            output_path = str(plots_dir / "hpo_study.html")

        # Verschiedene Plots erstellen
        try:
            # Optimization History
            fig_history = optuna.visualization.plot_optimization_history(self._study)

            # Parameter Importances
            fig_importance = optuna.visualization.plot_param_importances(self._study)

            # Parallel Coordinate
            fig_parallel = optuna.visualization.plot_parallel_coordinate(self._study)

            # Contour Plot
            fig_contour = optuna.visualization.plot_contour(self._study)

            # HTML mit allen Plots erstellen
            html_content = self._create_visualization_html(
                history_fig=fig_history,
                importance_fig=fig_importance,
                parallel_fig=fig_parallel,
                contour_fig=fig_contour,
            )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"✅ HPO Visualization gespeichert: {output_path}")
            return output_path

        except Exception as e:
            print(f"⚠️  Visualisierung fehlgeschlagen: {e}")
            return ""

    def _create_visualization_html(
        self,
        history_fig: Any,
        importance_fig: Any,
        parallel_fig: Any,
        contour_fig: Any,
    ) -> str:
        """Erstelle HTML für Visualisierungen."""
        # Konvertiere Figures zu JSON
        try:
            import plotly.io as pio
            history_json = pio.to_json(history_fig)
            importance_json = pio.to_json(importance_fig)
            parallel_json = pio.to_json(parallel_fig)
            contour_json = pio.to_json(contour_fig)
        except Exception:
            history_json = "{}"
            importance_json = "{}"
            parallel_json = "{}"
            contour_json = "{}"

        progress = self.get_optimization_progress()

        html_content = f"""
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HPO Study Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 40px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }}
        .plot-container {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
        }}
        .tab-button {{
            padding: 10px 20px;
            background: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }}
        .tab-button.active {{
            background: #667eea;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 HPO Study Visualization</h1>
        <p>Study: {self._study_name}</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{progress['total_trials']}</div>
            <div class="stat-label">Total Trials</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{progress['completed']}</div>
            <div class="stat-label">Completed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{progress['pruned']}</div>
            <div class="stat-label">Pruned</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{progress['running']}</div>
            <div class="stat-label">Running</div>
        </div>
    </div>

    <div class="tabs">
        <button class="tab-button active" onclick="switchTab('history')">History</button>
        <button class="tab-button" onclick="switchTab('importance')">Importance</button>
        <button class="tab-button" onclick="switchTab('parallel')">Parallel</button>
        <button class="tab-button" onclick="switchTab('contour')">Contour</button>
    </div>

    <div id="history" class="tab-content active">
        <div class="plot-container" id="history-plot"></div>
    </div>
    <div id="importance" class="tab-content">
        <div class="plot-container" id="importance-plot"></div>
    </div>
    <div id="parallel" class="tab-content">
        <div class="plot-container" id="parallel-plot"></div>
    </div>
    <div id="contour" class="tab-content">
        <div class="plot-container" id="contour-plot"></div>
    </div>

    <script>
        const historyData = {history_json};
        const importanceData = {importance_json};
        const parallelData = {parallel_json};
        const contourData = {contour_json};

        Plotly.newPlot('history-plot', historyData.data, historyData.layout);
        Plotly.newPlot('importance-plot', importanceData.data, importanceData.layout);
        Plotly.newPlot('parallel-plot', parallelData.data, parallelData.layout);
        Plotly.newPlot('contour-plot', contourData.data, contourData.layout);

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""
        return html_content

    def export_study(self, output_path: str) -> str:
        """
        Exportiere Study als JSON.

        Args:
            output_path: Pfad für JSON-Output

        Returns:
            Pfad zur Datei
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "study_name": self._study_name,
            "multi_objective": self._multi_objective,
            "search_space": self._search_space,
            "progress": self.get_optimization_progress(),
            "trials": [
                result.to_dict() for result in self._trial_results.values()
            ],
            "best_configs": self.get_best_configs(top_k=10),
            "exported_at": datetime.utcnow().isoformat(),
        }

        with open(output, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        return str(output)


def cmd_hpo_optimize(args: argparse.Namespace) -> int:
    """HPO Optimize Command."""
    print("🔬 Hyperparameter Optimizer")
    print("=" * 60)

    if not OPTUNA_AVAILABLE:
        print("❌ Fehler: Optuna nicht installiert")
        print("   Installiere mit: pip install optuna")
        return 1

    registry = RunRegistry()
    optimizer = HyperparameterOptimizer(registry)

    print(f"\n🎯 Starte Optimierung mit {args.trials} Trials...")
    print(f"   Multi-Objective: {optimizer._multi_objective}")
    print(f"   Search Space: {list(optimizer._search_space.keys())}")

    # Simuliere Optimierung
    for i in range(args.trials):
        # Config vorschlagen
        config = optimizer.suggest_config()

        # Simuliere Training (in echt: Training ausführen)
        import random
        simulated_metrics = {
            "delta_bpb": random.uniform(-0.05, 0.02),
            "efficiency_gain": random.uniform(-10, 20),
            "size_change": random.uniform(-20, 10),
        }

        # Ergebnis reporten
        optimizer.report_result(i, simulated_metrics)

        # Fortschritt
        if (i + 1) % 10 == 0:
            progress = optimizer.get_optimization_progress()
            print(f"\n   Trial {i + 1}/{args.trials}")
            print(f"   Completed: {progress['completed']}, Pruned: {progress['pruned']}")

    # Beste Configs
    print("\n" + "=" * 60)
    print("🏆 Beste Configs:")

    best_configs = optimizer.get_best_configs(top_k=3)
    for i, config in enumerate(best_configs):
        print(f"\n   #{i + 1}:")
        print(f"      depth={config.get('depth')}, width={config.get('width')}")
        print(f"      lr={config.get('learning_rate'):.2e}")
        if "values" in config:
            print(f"      Values: {config['values']}")

    # Visualization
    print("\n📊 Generiere Visualisierung...")
    viz_path = optimizer.create_study_visualization()
    print(f"   Saved: {viz_path}")

    # Export
    if args.output:
        export_path = optimizer.export_study(args.output)
        print(f"📥 Exportiert: {export_path}")

    print("\n" + "=" * 60)
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Erstelle Argument Parser."""
    parser = argparse.ArgumentParser(
        prog="hpo-optimize",
        description="Hyperparameter Optimization",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Anzahl Trials (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="JSON Export Pfad",
    )
    parser.set_defaults(func=cmd_hpo_optimize)
    return parser


def main() -> int:
    """Hauptfunktion."""
    parser = create_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
