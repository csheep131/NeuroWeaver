"""Dashboard CLI for run overview and management.

Provides commands for:
- Listing runs with filtering
- Viewing run details
- Comparing runs
- Managing promotions
- Generating reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from core.registry import RunRegistry
from core.config import load_config
from reports.leaderboard import LeaderboardGenerator
from research.ablation_engine import AblationReporter
from orchestrator.promote import PromotionSystem, Stage


def cmd_list(args: argparse.Namespace) -> int:
    """List runs command."""
    registry = RunRegistry(args.results_dir)

    # Filter by status
    status = args.status if args.status else None
    runs = registry.list_runs(status)

    if not runs:
        print("No runs found")
        return 0

    # Print table
    print(f"{'Run ID':<25} {'Status':<12} {'BPB':<10} {'Size (MB)':<12} {'ms/step':<10}")
    print("-" * 70)

    for run in runs:
        bpb = f"{run.val_bpb:.4f}" if run.val_bpb else "N/A"
        size_mb = f"{run.artifact_bytes / 1e6:.2f}" if run.artifact_bytes else "N/A"
        ms = f"{run.ms_per_step:.2f}" if run.ms_per_step else "N/A"

        print(f"{run.run_id:<25} {run.status:<12} {bpb:<10} {size_mb:<12} {ms:<10}")

    print(f"\nTotal: {len(runs)} runs")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show run details command."""
    registry = RunRegistry(args.results_dir)

    entry = registry.get(args.run_id)
    if not entry:
        print(f"Run not found: {args.run_id}")
        return 1

    print(f"Run: {entry.run_id}")
    print(f"  Status: {entry.status}")
    print(f"  Config Hash: {entry.config_hash}")
    print(f"  Git Commit: {entry.git_commit or 'N/A'}")
    print(f"  Parent: {entry.parent_run_id or 'None'}")
    print(f"  Seed: {entry.seed}")
    print(f"  Tags: {', '.join(entry.tags) if entry.tags else 'None'}")
    print()
    print("Metrics:")
    print(f"  BPB: {entry.val_bpb:.4f}" if entry.val_bpb else "  BPB: N/A")
    print(f"  ms/step: {entry.ms_per_step:.2f}" if entry.ms_per_step else "  ms/step: N/A")
    print(f"  Steps: {entry.steps_completed}")
    print(f"  Artifact Size: {entry.artifact_bytes:,} bytes")
    if entry.quantized_val_bpb:
        print(f"  Quantized BPB: {entry.quantized_val_bpb:.4f}")
    if entry.delta_bpb is not None:
        print(f"  Delta BPB: {entry.delta_bpb:+.4f}")
    if entry.delta_ms is not None:
        print(f"  Delta ms/step: {entry.delta_ms:+.2f}")

    # Show lineage
    lineage = registry.get_lineage(entry.run_id)
    if lineage:
        print()
        print("Lineage:")
        for ancestor in lineage:
            print(f"  ← {ancestor.run_id} (BPB={ancestor.val_bpb:.4f})")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare runs command."""
    registry = RunRegistry(args.results_dir)
    reporter = AblationReporter(registry)

    if args.run_ids:
        run_ids = args.run_ids
    else:
        run_ids = None

    comparison = reporter.generate_report()
    print(comparison.print_summary())

    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    """Show leaderboard command."""
    registry = RunRegistry(args.results_dir)
    gen = LeaderboardGenerator(args.results_dir)

    if args.category == "bpb":
        lb = gen.generate_by_bpb(args.top_k)
    elif args.category == "efficiency":
        lb = gen.generate_by_efficiency(args.top_k)
    elif args.category == "speed":
        lb = gen.generate_by_speed(args.top_k)
    else:
        print(f"Unknown category: {args.category}")
        return 1

    print(lb.print_table())
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """Promotion management command."""
    registry = RunRegistry(args.results_dir)
    promo = PromotionSystem(registry)

    if args.action == "status":
        print(promo.print_summary())

    elif args.action == "evaluate":
        if args.stage == "screening":
            results = promo.evaluate_screening()
        elif args.stage == "focus":
            results = promo.evaluate_focus()
        elif args.stage == "final":
            results = promo.evaluate_final()
        else:
            print(f"Unknown stage: {args.stage}")
            return 1

        promoted = [r for r in results if r.was_promoted()]
        print(f"Evaluated {len(results)} runs, {len(promoted)} promoted:")
        for r in promoted:
            print(f"  {r.run_id}: {r.from_stage.value} → {r.to_stage.value}")

    elif args.action == "apply":
        if args.stage == "screening":
            results = promo.evaluate_screening()
        elif args.stage == "focus":
            results = promo.evaluate_focus()
        elif args.stage == "final":
            results = promo.evaluate_final()
        else:
            print(f"Unknown stage: {args.stage}")
            return 1

        # Apply promotions
        for r in results:
            if r.was_promoted() and r.to_stage:
                promo.promote(r.run_id, r.to_stage)
                print(f"Promoted {r.run_id} to {r.to_stage.value}")

    elif args.action == "manual":
        stage_map = {
            "screening": Stage.SCREENING,
            "focus": Stage.FOCUS,
            "final": Stage.FINAL,
            "submission": Stage.SUBMISSION,
        }
        stage = stage_map.get(args.to_stage)
        if not stage:
            print(f"Unknown stage: {args.to_stage}")
            return 1

        if promo.promote(args.run_id, stage):
            print(f"Promoted {args.run_id} to {stage.value}")
        else:
            print(f"Failed to promote {args.run_id}")
            return 1

    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    """Apply kill rules command."""
    registry = RunRegistry(args.results_dir)
    reporter = AblationReporter(registry)

    if args.dry_run:
        evals = reporter.evaluate_all_runs()
        to_kill = [(rid, reason) for rid, (kill, reason) in evals.items() if kill]

        if not to_kill:
            print("No runs would be killed")
            return 0

        print(f"Would kill {len(to_kill)} runs:")
        for run_id, reason in to_kill:
            print(f"  {run_id}: {reason}")
    else:
        killed = reporter.apply_kills()
        if not killed:
            print("No runs killed")
            return 0

        print(f"Killed {len(killed)} runs:")
        for run_id in killed:
            entry = registry.get(run_id)
            if entry:
                print(f"  {run_id}: {entry.notes}")

    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run parameter sweep command."""
    from orchestrator.sweep import create_sweep

    runner = create_sweep(
        sweep_id=args.sweep_id,
        base_config=args.base_config,
        parameters=args.params,
        seeds=args.seeds.split(",") if args.seeds else None,
        run_prefix=args.prefix,
    )

    runs = runner.generate_runs()
    print(runner.print_summary())

    if args.execute and not args.dry_run:
        print("\nExecuting sweep...")
        summary = runner.execute_runs()
        print(f"\nExecuted: {summary['executed']}, Failed: {summary['failed']}")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="ablation",
        description="Ablation Machine Dashboard",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List runs")
    list_parser.add_argument(
        "--status",
        choices=["pending", "running", "completed", "failed", "killed"],
        help="Filter by status",
    )
    list_parser.set_defaults(func=cmd_list)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show run details")
    show_parser.add_argument("run_id", help="Run ID to show")
    show_parser.set_defaults(func=cmd_show)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare runs")
    compare_parser.add_argument(
        "run_ids", nargs="*", help="Run IDs to compare (all if none specified)"
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Leaderboard command
    lb_parser = subparsers.add_parser("leaderboard", help="Show leaderboard")
    lb_parser.add_argument(
        "--category",
        choices=["bpb", "efficiency", "speed"],
        default="bpb",
        help="Leaderboard category",
    )
    lb_parser.add_argument(
        "--top-k", type=int, default=20, help="Number of entries to show"
    )
    lb_parser.set_defaults(func=cmd_leaderboard)

    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Manage promotions")
    promote_parser.add_argument(
        "action",
        choices=["status", "evaluate", "apply", "manual"],
        help="Promotion action",
    )
    promote_parser.add_argument(
        "--stage",
        choices=["screening", "focus", "final", "submission"],
        help="Stage for evaluation",
    )
    promote_parser.add_argument("--run-id", help="Run ID for manual promotion")
    promote_parser.add_argument(
        "--to-stage", help="Target stage for manual promotion"
    )
    promote_parser.set_defaults(func=cmd_promote)

    # Kill command
    kill_parser = subparsers.add_parser("kill", help="Apply kill rules")
    kill_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be killed"
    )
    kill_parser.set_defaults(func=cmd_kill)

    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run parameter sweep")
    sweep_parser.add_argument("--sweep-id", required=True, help="Sweep identifier")
    sweep_parser.add_argument(
        "--base-config", required=True, help="Base config path"
    )
    sweep_parser.add_argument(
        "--params",
        type=str,
        nargs="+",
        action="append",
        help="Parameter name and values",
    )
    sweep_parser.add_argument("--seeds", help="Comma-separated seeds")
    sweep_parser.add_argument("--prefix", default="", help="Run ID prefix")
    sweep_parser.add_argument(
        "--execute", action="store_true", help="Execute the sweep"
    )
    sweep_parser.add_argument(
        "--dry-run", action="store_true", help="Don't execute, just show"
    )
    sweep_parser.set_defaults(func=cmd_sweep)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
