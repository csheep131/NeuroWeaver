#!/usr/bin/env python3
"""
Run Explorer CLI für NeuroWeave Phase 5.

Interaktive Terminal-UI für Run-Analyse mit Rich.

Features:
- Fuzzy Search nach Runs
- Filtern nach Features, Status, Budget
- Sortieren nach Metriken
- Detail-Ansicht pro Run
- Vergleichs-Modus (2-5 Runs)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import RunRegistry, RunEntry

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warnung: Rich nicht installiert. Installiere mit: pip install rich")

try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Warnung: FuzzyWuzzy nicht installiert. Installiere mit: pip install fuzzywuzzy python-Levenshtein")


class RunExplorer:
    """
    Interaktive CLI für Run-Analyse (TUI).

    Features:
    - Fuzzy Search nach Runs
    - Filtern nach Features, Status, Budget
    - Sortieren nach Metriken
    - Detail-Ansicht pro Run
    - Vergleichs-Modus (2-5 Runs)

    Example:
        registry = RunRegistry("results")
        explorer = RunExplorer(registry)

        # Interaktive Session starten
        explorer.run_interactive()

        # Programmatische Nutzung
        results = explorer.search("gqa")
        comparison = explorer.compare_runs(["run001", "run009"])
    """

    def __init__(self, registry: RunRegistry) -> None:
        """
        Initialisiere Run Explorer.

        Args:
            registry: RunRegistry für Datenzugriff
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich ist erforderlich. Installiere mit: pip install rich"
            )

        self._registry = registry
        self._console = Console()
        self._current_runs: List[RunEntry] = []
        self._filtered_runs: List[RunEntry] = []
        self._compare_list: List[str] = []  # Run-IDs zum Vergleichen

        # Commands-Hilfe
        self._help_text = """
[bold cyan]Verfügbare Commands:[/bold cyan]

  [green]/search <query>[/green]     - Fuzzy Search nach Runs
  [green]/filter <feature>[/green]   - Nach Feature filtern
  [green]/sort <metric>[/green]      - Nach Metrik sortieren
  [green]/compare <ids>[/green]      - Runs vergleichen
  [green]/details <id>[/green]       - Detail-Ansicht
  [green]/list[/green]               - Alle Runs anzeigen
  [green]/clear[/green]              - Filter zurücksetzen
  [green]/export <format>[/green]    - Export (csv, json, markdown)
  [green]/help[/green]               - Diese Hilfe
  [green]/quit[/green]               - Beenden

[bold cyan]Beispiele:[/bold cyan]
  /search gqa
  /filter completed
  /sort delta_bpb
  /compare run001 run009 run015
  /details run001
  /export markdown
"""

    def _get_all_runs(self) -> List[RunEntry]:
        """Hole alle Runs."""
        return list(self._registry.entries.values())

    def _print_header(self) -> None:
        """Drucke Header."""
        header = Panel(
            Text("🔍 NeuroWeave Run Explorer", style="bold cyan"),
            subtitle="Phase 5 - Interactive Run Analysis",
            box=box.DOUBLE,
        )
        self._console.print(header)
        self._console.print()

    def _print_help(self) -> None:
        """Drucke Hilfe."""
        self._console.print(Syntax(self._help_text.strip(), "text", theme="monokai"))

    def _print_runs_table(self, runs: List[RunEntry], limit: int = 20) -> None:
        """Drucke Runs als Tabelle."""
        if not runs:
            self._console.print("[yellow]Keine Runs gefunden[/yellow]")
            return

        table = Table(
            title=f"Runs ({len(runs)} total, zeige {min(limit, len(runs))})",
            box=box.ROUNDED,
            show_lines=True,
        )

        table.add_column("Run ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="white")
        table.add_column("BPB", justify="right")
        table.add_column("ΔBPB", justify="right")
        table.add_column("ms/step", justify="right")
        table.add_column("Δms", justify="right")
        table.add_column("Parent", style="dim")
        table.add_column("Tags", style="yellow")

        for run in runs[:limit]:
            # Status-Farbe
            status_style = {
                "completed": "green",
                "failed": "red",
                "killed": "yellow",
                "running": "blue",
                "pending": "dim",
            }.get(run.status, "white")

            # ΔBPB Farbe
            delta_bpb_str = "N/A"
            delta_bpb_style = "white"
            if run.delta_bpb is not None:
                delta_bpb_str = f"{run.delta_bpb:+.4f}"
                delta_bpb_style = "green" if run.delta_bpb < 0 else "red"

            # Δms Farbe
            delta_ms_str = "N/A"
            delta_ms_style = "white"
            if run.delta_ms is not None:
                delta_ms_str = f"{run.delta_ms:+.2f}"
                delta_ms_style = "green" if run.delta_ms < 0 else "red"

            # Tags kürzen
            tags_str = ", ".join(run.tags[:3]) if run.tags else "-"
            if len(run.tags) > 3:
                tags_str += "..."

            table.add_row(
                run.run_id,
                Text(run.status, style=status_style),
                f"{run.val_bpb:.4f}" if run.val_bpb is not None else "N/A",
                Text(delta_bpb_str, style=delta_bpb_style),
                f"{run.ms_per_step:.2f}" if run.ms_per_step is not None else "N/A",
                Text(delta_ms_str, style=delta_ms_style),
                run.parent_run_id or "-",
                tags_str,
            )

        self._console.print(table)

    def _print_run_details(self, run_id: str) -> None:
        """Drucke Details eines Runs."""
        entry = self._registry.get(run_id)

        if not entry:
            self._console.print(f"[red]Run nicht gefunden: {run_id}[/red]")
            return

        # Header
        header = Panel(
            Text(f"📊 {run_id}", style="bold cyan"),
            subtitle=entry.status.upper(),
            box=box.ROUNDED,
        )
        self._console.print(header)
        self._console.print()

        # Metriken
        metrics_table = Table(box=box.SIMPLE, show_header=False)
        metrics_table.add_column("Property", style="cyan")
        metrics_table.add_column("Value")

        metrics_table.add_row("Config Hash", entry.config_hash[:12] + "...")
        metrics_table.add_row("Git Commit", entry.git_commit or "N/A")
        metrics_table.add_row("Parent Run", entry.parent_run_id or "None")
        metrics_table.add_row("Seed", str(entry.seed))
        metrics_table.add_row("Start Time", entry.start_time or "N/A")
        metrics_table.add_row("End Time", entry.end_time or "N/A")
        metrics_table.add_row("", "")  # Leerzeile

        # Metriken
        metrics_table.add_row("BPB", f"{entry.val_bpb:.4f}" if entry.val_bpb else "N/A")
        metrics_table.add_row("ms/step", f"{entry.ms_per_step:.2f}" if entry.ms_per_step else "N/A")
        metrics_table.add_row("Steps", str(entry.steps_completed))
        metrics_table.add_row("Artifact Size", f"{entry.artifact_bytes:,} bytes")
        metrics_table.add_row("", "")  # Leerzeile

        # Deltas
        if entry.delta_bpb is not None:
            delta_style = "green" if entry.delta_bpb < 0 else "red"
            metrics_table.add_row(
                "ΔBPB",
                Text(f"{entry.delta_bpb:+.4f}", style=delta_style),
            )
        if entry.delta_ms is not None:
            delta_style = "green" if entry.delta_ms < 0 else "red"
            metrics_table.add_row(
                "Δms/step",
                Text(f"{entry.delta_ms:+.2f}", style=delta_style),
            )

        metrics_table.add_row("", "")  # Leerzeile
        metrics_table.add_row("Notes", entry.notes or "-")
        metrics_table.add_row("Tags", ", ".join(entry.tags) if entry.tags else "-")

        self._console.print(metrics_table)

        # Lineage
        lineage = self._registry.get_lineage(run_id)
        if lineage:
            self._console.print()
            lineage_panel = Panel(
                " ← ".join([r.run_id for r in lineage]),
                title="Lineage",
                box=box.ROUNDED,
            )
            self._console.print(lineage_panel)

    def _print_comparison(self, run_ids: List[str]) -> None:
        """Drucke Vergleichstabelle."""
        runs = [self._registry.get(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]

        if not runs:
            self._console.print("[yellow]Keine Runs zum Vergleichen gefunden[/yellow]")
            return

        table = Table(
            title=f"Vergleich: {len(runs)} Runs",
            box=box.ROUNDED,
            show_lines=True,
        )

        table.add_column("Metric", style="cyan")
        for run in runs:
            table.add_column(run.run_id, justify="right")

        # Metriken
        metrics = [
            ("Status", lambda r: r.status),
            ("BPB", lambda r: f"{r.val_bpb:.4f}" if r.val_bpb else "N/A"),
            ("ΔBPB", lambda r: f"{r.delta_bpb:+.4f}" if r.delta_bpb else "N/A"),
            ("ms/step", lambda r: f"{r.ms_per_step:.2f}" if r.ms_per_step else "N/A"),
            ("Δms", lambda r: f"{r.delta_ms:+.2f}" if r.delta_ms else "N/A"),
            ("Steps", lambda r: str(r.steps_completed)),
            ("Parent", lambda r: r.parent_run_id or "-"),
            ("Tags", lambda r: ", ".join(r.tags[:3]) if r.tags else "-"),
        ]

        for metric_name, getter in metrics:
            row = [metric_name]
            for run in runs:
                row.append(getter(run))
            table.add_row(*row)

        self._console.print(table)

        # Markdown-Tabelle für Export
        self._console.print()
        self._console.print("[dim]Markdown-Tabelle:[/dim]")
        md_table = self._generate_markdown_comparison(runs)
        self._console.print(Syntax(md_table, "markdown", theme="monokai"))

    def _generate_markdown_comparison(self, runs: List[RunEntry]) -> str:
        """Generiere Markdown-Vergleichstabelle."""
        lines = []

        # Header
        headers = ["Metric"] + [r.run_id for r in runs]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Rows
        metrics = [
            ("Status", lambda r: r.status),
            ("BPB", lambda r: f"{r.val_bpb:.4f}" if r.val_bpb else "N/A"),
            ("ΔBPB", lambda r: f"{r.delta_bpb:+.4f}" if r.delta_bpb else "N/A"),
            ("ms/step", lambda r: f"{r.ms_per_step:.2f}" if r.ms_per_step else "N/A"),
            ("Δms", lambda r: f"{r.delta_ms:+.2f}" if r.delta_ms else "N/A"),
            ("Steps", lambda r: str(r.steps_completed)),
            ("Parent", lambda r: r.parent_run_id or "-"),
        ]

        for metric_name, getter in metrics:
            row = [metric_name] + [getter(r) for r in runs]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def search(self, query: str) -> List[str]:
        """
        Fuzzy Search nach Runs.

        Sucht in:
        - Run-ID
        - Features
        - Notes
        - Status

        Args:
            query: Suchanfrage

        Returns:
            Liste von Run-IDs die匹配en
        """
        all_runs = self._get_all_runs()

        if not FUZZY_AVAILABLE:
            # Fallback: Einfache String-Suche
            query_lower = query.lower()
            matches = [
                r.run_id for r in all_runs
                if (query_lower in r.run_id.lower() or
                    query_lower in r.notes.lower() or
                    query_lower in r.status.lower() or
                    any(query_lower in tag.lower() for tag in r.tags))
            ]
            return matches

        # Fuzzy Search
        # Erstelle Such-Text für jeden Run
        search_texts = {}
        for run in all_runs:
            search_text = f"{run.run_id} {' '.join(run.tags)} {run.notes} {run.status}"
            search_texts[run.run_id] = search_text

        # Fuzzy Match
        results = process.extract(
            query,
            search_texts,
            scorer=fuzz.WRatio,
            limit=20,
            score_cutoff=40,
        )

        return [run_id for run_id, score in results]

    def filter_runs(
        self,
        status: Optional[str] = None,
        features: Optional[List[str]] = None,
        min_bpb: Optional[float] = None,
        max_bpb: Optional[float] = None,
        has_parent: Optional[bool] = None,
    ) -> List[RunEntry]:
        """
        Filtere Runs nach Kriterien.

        Args:
            status: Filter nach Status
            features: Filter nach Features (Tags)
            min_bpb: Minimale BPB
            max_bpb: Maximale BPB
            has_parent: Filter nach Parent-Run

        Returns:
            Gefilterte Liste von Runs
        """
        runs = self._get_all_runs()
        filtered = []

        for run in runs:
            # Status-Filter
            if status and run.status != status:
                continue

            # Feature-Filter
            if features:
                run_tags_lower = [t.lower() for t in run.tags]
                if not any(any(f.lower() in tag for tag in run_tags_lower) for f in features):
                    continue

            # BPB-Filter
            if min_bpb is not None and (run.val_bpb is None or run.val_bpb < min_bpb):
                continue
            if max_bpb is not None and (run.val_bpb is None or run.val_bpb > max_bpb):
                continue

            # Parent-Filter
            if has_parent is not None:
                has_parent_actual = run.parent_run_id is not None
                if has_parent != has_parent_actual:
                    continue

            filtered.append(run)

        return filtered

    def sort_runs(
        self,
        runs: List[RunEntry],
        metric: str,
        reverse: bool = False,
    ) -> List[RunEntry]:
        """
        Sortiere Runs nach Metrik.

        Args:
            runs: Liste von Runs
            metric: Metrik-Name (bpb, delta_bpb, ms_per_step, delta_ms, steps)
            reverse: Umgekehrte Sortierreihenfolge

        Returns:
            Sortierte Liste
        """
        def get_sort_key(run: RunEntry) -> float:
            metric_map = {
                "bpb": run.val_bpb,
                "delta_bpb": run.delta_bpb,
                "ms_per_step": run.ms_per_step,
                "delta_ms": run.delta_ms,
                "steps": float(run.steps_completed),
                "artifact_bytes": float(run.artifact_bytes),
            }
            value = metric_map.get(metric.lower())
            return value if value is not None else float("inf")

        return sorted(runs, key=get_sort_key, reverse=reverse)

    def compare_runs(self, run_ids: List[str]) -> str:
        """
        Vergleichstabelle generieren.

        Args:
            run_ids: Liste von Run-IDs zum Vergleichen

        Returns:
            Markdown-Tabelle mit Metriken-Vergleich
        """
        runs = [self._registry.get(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]

        if not runs:
            return "Keine Runs zum Vergleichen gefunden"

        return self._generate_markdown_comparison(runs)

    def export_runs(
        self,
        runs: List[RunEntry],
        format: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Exportiere Runs in verschiedenen Formaten.

        Args:
            runs: Liste von Runs
            format: Export-Format (csv, json, markdown)
            output_path: Output-Pfad (optional)

        Returns:
            Exportierter String oder Pfad
        """
        if format == "json":
            data = [r.to_dict() for r in runs]
            output = json.dumps(data, indent=2, default=str)

        elif format == "csv":
            if not runs:
                output = ""
            else:
                # Header
                headers = list(runs[0].to_dict().keys())
                lines = [",".join(headers)]

                # Rows
                for run in runs:
                    row_dict = run.to_dict()
                    row = []
                    for h in headers:
                        value = row_dict.get(h, "")
                        if isinstance(value, list):
                            value = ";".join(value)
                        elif isinstance(value, str) and "," in value:
                            value = f'"{value}"'
                        row.append(str(value))
                    lines.append(",".join(row))

                output = "\n".join(lines)

        elif format == "markdown":
            if not runs:
                output = "Keine Runs"
            else:
                lines = []
                # Header
                headers = ["Run ID", "Status", "BPB", "ΔBPB", "ms/step", "Parent"]
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

                # Rows
                for run in runs:
                    row = [
                        run.run_id,
                        run.status,
                        f"{run.val_bpb:.4f}" if run.val_bpb else "N/A",
                        f"{run.delta_bpb:+.4f}" if run.delta_bpb else "N/A",
                        f"{run.ms_per_step:.2f}" if run.ms_per_step else "N/A",
                        run.parent_run_id or "-",
                    ]
                    lines.append("| " + " | ".join(row) + " |")

                output = "\n".join(lines)

        else:
            output = f"Unbekanntes Format: {format}"

        # Speichern wenn Pfad angegeben
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            return output_path

        return output

    def run_interactive(self) -> None:
        """
        Interaktive Session starten.

        Commands:
        /search <query>     - Fuzzy Search
        /filter <feature>   - Nach Feature filtern
        /sort <metric>      - Nach Metrik sortieren
        /compare <ids>      - Runs vergleichen
        /details <id>       - Detail-Ansicht
        /export <format>    - Export (csv, json, markdown)
        /help               - Hilfe
        """
        self._print_header()
        self._console.print("[dim]Tippe /help für Commands, /quit zum Beenden[/dim]\n")

        # Initiale Run-Liste laden
        self._current_runs = self._get_all_runs()
        self._filtered_runs = self._current_runs

        # Zeige initiale Übersicht
        self._print_runs_table(self._filtered_runs)

        # Interaktive Schleife
        while True:
            try:
                # Prompt
                user_input = Prompt.ask(
                    "\n[bold cyan]Explorer[/bold cyan]",
                    default="",
                ).strip()

                if not user_input:
                    continue

                # Command parsen
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:]

                # Commands verarbeiten
                if command in ("/quit", "/exit", "/q"):
                    self._console.print("[green]Auf Wiedersehen![/green]")
                    break

                elif command == "/help":
                    self._print_help()

                elif command == "/search":
                    if not args:
                        self._console.print("[yellow]Usage: /search <query>[/yellow]")
                        continue

                    query = " ".join(args)
                    matches = self.search(query)

                    if matches:
                        self._console.print(f"[green]{len(matches)} Runs gefunden[/green]")
                        matched_runs = [
                            self._registry.get(rid) for rid in matches
                        ]
                        matched_runs = [r for r in matched_runs if r]
                        self._print_runs_table(matched_runs)
                    else:
                        self._console.print("[yellow]Keine Übereinstimmungen[/yellow]")

                elif command == "/filter":
                    if not args:
                        self._console.print("[yellow]Usage: /filter <status|feature>[/yellow]")
                        continue

                    filter_value = " ".join(args).lower()

                    # Versuche nach Status zu filtern
                    valid_statuses = {"pending", "running", "completed", "failed", "killed"}
                    if filter_value in valid_statuses:
                        self._filtered_runs = self.filter_runs(status=filter_value)
                    else:
                        # Filter nach Feature
                        self._filtered_runs = self.filter_runs(features=[filter_value])

                    self._console.print(f"[green]{len(self._filtered_runs)} Runs nach Filter[/green]")
                    self._print_runs_table(self._filtered_runs)

                elif command == "/sort":
                    if not args:
                        self._console.print("[yellow]Usage: /sort <metric>[/yellow]")
                        continue

                    metric = args[0].lower()
                    valid_metrics = {"bpb", "delta_bpb", "ms_per_step", "delta_ms", "steps"}

                    if metric not in valid_metrics:
                        self._console.print(f"[yellow]Ungültige Metrik. Gültig: {', '.join(valid_metrics)}[/yellow]")
                        continue

                    reverse = "--desc" in args or metric in ("bpb", "ms_per_step", "steps")
                    self._filtered_runs = self.sort_runs(self._filtered_runs, metric, reverse)
                    self._print_runs_table(self._filtered_runs)

                elif command == "/compare":
                    if not args:
                        self._console.print("[yellow]Usage: /compare <run_id1> <run_id2> ...[/yellow]")
                        continue

                    self._print_comparison(args)

                elif command == "/details":
                    if not args:
                        self._console.print("[yellow]Usage: /details <run_id>[/yellow]")
                        continue

                    self._print_run_details(args[0])

                elif command == "/list":
                    self._filtered_runs = self._get_all_runs()
                    self._print_runs_table(self._filtered_runs)

                elif command == "/clear":
                    self._filtered_runs = self._get_all_runs()
                    self._console.print("[green]Filter zurückgesetzt[/green]")
                    self._print_runs_table(self._filtered_runs)

                elif command == "/export":
                    if not args:
                        self._console.print("[yellow]Usage: /export <format> [output_path][/yellow]")
                        continue

                    format = args[0].lower()
                    output_path = args[1] if len(args) > 1 else None

                    valid_formats = {"csv", "json", "markdown"}
                    if format not in valid_formats:
                        self._console.print(f"[yellow]Ungültiges Format. Gültig: {', '.join(valid_formats)}[/yellow]")
                        continue

                    result = self.export_runs(self._filtered_runs, format, output_path)

                    if output_path:
                        self._console.print(f"[green]Exportiert nach: {result}[/green]")
                    else:
                        self._console.print(Syntax(result, format, theme="monokai"))

                else:
                    self._console.print(f"[yellow]Unbekannter Command: {command}[/yellow]")
                    self._console.print("[dim]Tippe /help für Hilfe[/dim]")

            except KeyboardInterrupt:
                self._console.print("\n[yellow]Unterbrochen. Tippe /quit zum Beenden.[/yellow]")
            except Exception as e:
                self._console.print(f"[red]Fehler: {e}[/red]")


def cmd_run_explorer(args: argparse.Namespace) -> int:
    """Run Explorer Command."""
    print("🚀 Starte Run Explorer...")

    if not RICH_AVAILABLE:
        print("❌ Fehler: Rich nicht installiert")
        print("   Installiere mit: pip install rich")
        return 1

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results-Verzeichnis nicht gefunden: {results_dir}")
        return 1

    registry = RunRegistry(results_dir=str(results_dir))

    try:
        explorer = RunExplorer(registry)
        explorer.run_interactive()
        return 0
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Run Explorer beendet")
        return 0


def create_parser() -> argparse.ArgumentParser:
    """Erstelle Argument Parser."""
    parser = argparse.ArgumentParser(
        prog="run-explorer",
        description="Interaktive Run-Analyse CLI",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results Verzeichnis",
    )
    parser.set_defaults(func=cmd_run_explorer)
    return parser


def main() -> int:
    """Hauptfunktion."""
    parser = create_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
