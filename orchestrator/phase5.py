#!/usr/bin/env python3
"""
Phase 5: Advanced Features - Zentrale Steuerung

Usage:
    python3 -m orchestrator.phase5 advanced-dashboard    # Plotly Dashboard
    python3 -m orchestrator.phase5 run-explorer          # Interaktive CLI
    python3 -m orchestrator.phase5 live-monitor <run_id> # Live Monitoring
    python3 -m orchestrator.phase5 health-check <run_id> # Health Check
    python3 -m orchestrator.phase5 distributed-submit    # Distributed Runs
    python3 -m orchestrator.phase5 queue-manager         # Queue Management
    python3 -m orchestrator.phase5 hpo-optimize          # Hyperparameter-Opt
    python3 -m orchestrator.phase5 nas-search            # Architecture Search

Phase 5 Komponenten:
- Advanced Visualization (Woche 11-12)
- Real-time Monitoring & Alerting (Woche 13-14)
- Distributed Execution (Woche 15-16)
- AutoML Integration (Woche 17-18)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_advanced_dashboard(args: argparse.Namespace) -> int:
    """Advanced Dashboard Command."""
    from orchestrator.dashboard_advanced import main as dashboard_main
    return dashboard_main()


def cmd_run_explorer(args: argparse.Namespace) -> int:
    """Run Explorer Command."""
    from orchestrator.run_explorer import main as explorer_main
    return explorer_main()


def cmd_live_monitor(args: argparse.Namespace) -> int:
    """Live Monitor Command."""
    from orchestrator.realtime_monitor import main as monitor_main
    return monitor_main()


def cmd_health_check(args: argparse.Namespace) -> int:
    """Health Check Command."""
    from orchestrator.health_checker import main as health_main
    return health_main()


def cmd_distributed_runner(args: argparse.Namespace) -> int:
    """Distributed Runner Command."""
    from orchestrator.distributed_runner import main as distributed_main
    return distributed_main()


def cmd_queue_manager(args: argparse.Namespace) -> int:
    """Queue Manager Command."""
    from orchestrator.run_queue import main as queue_main
    return queue_main()


def cmd_hpo_optimize(args: argparse.Namespace) -> int:
    """HPO Optimize Command."""
    from research.hpo_integration import main as hpo_main
    return hpo_main()


def cmd_nas_search(args: argparse.Namespace) -> int:
    """NAS Search Command."""
    from research.nas_integration import main as nas_main
    return nas_main()


def create_parser() -> argparse.ArgumentParser:
    """Erstelle Argument Parser."""
    parser = argparse.ArgumentParser(
        prog="phase5",
        description="Phase 5: Advanced Features - Zentrale Steuerung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s advanced-dashboard              # Interaktives Dashboard
  %(prog)s run-explorer                    # Interaktive Run-Analyse
  %(prog)s live-monitor run001             # Live Monitoring
  %(prog)s health-check run001             # Health Check
  %(prog)s distributed-runner --workers 4  # Distributed Execution
  %(prog)s queue-manager                   # Queue Management
  %(prog)s hpo-optimize --trials 50        # Hyperparameter-Optimierung
  %(prog)s nas-search --budget 100         # Architecture Search
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Advanced Dashboard
    dashboard_parser = subparsers.add_parser(
        "advanced-dashboard",
        help="Interaktives Dashboard mit Plotly",
    )
    dashboard_parser.add_argument(
        "--results-dir",
        default="results",
        help="Results Verzeichnis",
    )
    dashboard_parser.add_argument(
        "--output",
        type=str,
        help="Output Pfad für Dashboard HTML",
    )
    dashboard_parser.set_defaults(func=cmd_advanced_dashboard)

    # Run Explorer
    explorer_parser = subparsers.add_parser(
        "run-explorer",
        help="Interaktive Run-Analyse CLI",
    )
    explorer_parser.add_argument(
        "--results-dir",
        default="results",
        help="Results Verzeichnis",
    )
    explorer_parser.set_defaults(func=cmd_run_explorer)

    # Live Monitor
    monitor_parser = subparsers.add_parser(
        "live-monitor",
        help="Live Monitoring für Training Runs",
    )
    monitor_parser.add_argument(
        "run_id",
        type=str,
        help="Run-ID zu überwachen",
    )
    monitor_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket Port (default: 8765)",
    )
    monitor_parser.set_defaults(func=cmd_live_monitor)

    # Health Check
    health_parser = subparsers.add_parser(
        "health-check",
        help="Training Health Checker",
    )
    health_parser.add_argument(
        "run_id",
        type=str,
        nargs="?",
        default="demo_run",
        help="Run-ID für Health Check",
    )
    health_parser.set_defaults(func=cmd_health_check)

    # Distributed Runner
    distributed_parser = subparsers.add_parser(
        "distributed-runner",
        help="Distributed Run Execution",
    )
    distributed_parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Anzahl Worker (default: 2)",
    )
    distributed_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Max Runs pro Worker (default: 1)",
    )
    distributed_parser.add_argument(
        "--memory-limit",
        type=int,
        default=8000,
        help="Memory Limit MB pro Worker (default: 8000)",
    )
    distributed_parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Anzahl Runs zum Testen (default: 10)",
    )
    distributed_parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test-Dauer in Sekunden (default: 30)",
    )
    distributed_parser.set_defaults(func=cmd_distributed_runner)

    # Queue Manager
    queue_parser = subparsers.add_parser(
        "queue-manager",
        help="Run Queue Manager",
    )
    queue_parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Anzahl Test-Runs (default: 10)",
    )
    queue_parser.set_defaults(func=cmd_queue_manager)

    # HPO Optimize
    hpo_parser = subparsers.add_parser(
        "hpo-optimize",
        help="Hyperparameter Optimization",
    )
    hpo_parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Anzahl Trials (default: 50)",
    )
    hpo_parser.add_argument(
        "--output",
        type=str,
        help="JSON Export Pfad",
    )
    hpo_parser.set_defaults(func=cmd_hpo_optimize)

    # NAS Search
    nas_parser = subparsers.add_parser(
        "nas-search",
        help="Neural Architecture Search",
    )
    nas_parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Such-Budget (default: 100)",
    )
    nas_parser.add_argument(
        "--max-vram",
        type=int,
        default=8000,
        help="Max VRAM MB (default: 8000)",
    )
    nas_parser.add_argument(
        "--max-size",
        type=int,
        default=500,
        help="Max Size MB (default: 500)",
    )
    nas_parser.add_argument(
        "--min-depth",
        type=int,
        default=8,
        help="Min Depth (default: 8)",
    )
    nas_parser.add_argument(
        "--max-depth",
        type=int,
        default=16,
        help="Max Depth (default: 16)",
    )
    nas_parser.add_argument(
        "--min-width",
        type=int,
        default=256,
        help="Min Width (default: 256)",
    )
    nas_parser.add_argument(
        "--max-width",
        type=int,
        default=1024,
        help="Max Width (default: 1024)",
    )
    nas_parser.add_argument(
        "--output",
        type=str,
        help="JSON Export Pfad",
    )
    nas_parser.set_defaults(func=cmd_nas_search)

    return parser


def main() -> int:
    """Hauptfunktion."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
