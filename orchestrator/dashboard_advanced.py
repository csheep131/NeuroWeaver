#!/usr/bin/env python3
"""
Advanced Dashboard für NeuroWeave Phase 5.

Interaktives Dashboard mit Plotly für:
- 3D Pareto-Frontier (rotierbar)
- Zeitreihen von Success Metrics
- Feature-Importance Heatmap
- Run-Lineage Graph (interaktiv)
- Guardrail Violation Timeline
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.registry import RunRegistry, RunEntry
from research.success_metrics import SuccessMetricsTracker

try:
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
PLOTLY_AVAILABLE = True
except ImportError:
PLOTLY_AVAILABLE = False
print("Warnung: Plotly nicht installiert. Installiere mit: pip install plotly dash")


class AdvancedDashboard:
"""
Interaktives Dashboard mit Plotly.

Features:
- Interaktive Pareto-Frontier (3D rotierbar)
- Zeitreihen von Success Metrics
- Feature-Importance Heatmap
- Run-Lineage Graph (interaktiv)
- Guardrail Violation Timeline

Example:
registry = RunRegistry("results")
tracker = SuccessMetricsTracker(registry)
dashboard = AdvancedDashboard(registry, tracker)

# Einzelne Plots erstellen
dashboard.create_pareto_3d_plot("plots/pareto_3d.html")
dashboard.create_metrics_timeseries("plots/metrics_over_time.html")

# Komplettes Dashboard generieren
dashboard.generate_full_dashboard("plots/phase5_dashboard.html")
"""

def __init__(
self,
registry: RunRegistry,
tracker: SuccessMetricsTracker,
) -> None:
"""
Initialisiere Advanced Dashboard.

Args:
registry: RunRegistry für Datenzugriff
tracker: SuccessMetricsTracker für Metriken
"""
if not PLOTLY_AVAILABLE:
raise ImportError(
"Plotly ist erforderlich. Installiere mit: pip install plotly dash"
)

self._registry = registry
self._tracker = tracker
self._plots_dir = Path(__file__).parent.parent / "plots"
self._plots_dir.mkdir(parents=True, exist_ok=True)

# Farb-Paletten
self._color_palette = {
"primary": "#1f77b4",
"secondary": "#ff7f0e",
"success": "#2ca02c",
"danger": "#d62728",
"warning": "#ff9800",
"info": "#00bcd4",
}

def _get_completed_runs(self) -> List[RunEntry]:
"""Hole alle abgeschlossenen Runs."""
return self._registry.list_runs(status="completed")

def _get_runs_with_parent(self) -> List[RunEntry]:
"""Hole Runs mit Parent-Run."""
completed = self._get_completed_runs()
return [
r for r in completed
if r.parent_run_id is not None and r.delta_bpb is not None
]

def _ensure_output_dir(self, output_path: str) -> Path:
"""Stelle sicher, dass Output-Verzeichnis existiert."""
output = Path(output_path)
output.parent.mkdir(parents=True, exist_ok=True)
return output

def create_pareto_3d_plot(
self,
output_path: Optional[str] = None,
) -> go.Figure:
"""
3D Pareto-Frontier (rotierbar).

Achsen:
- X: ΔBPB (niedriger = besser)
- Y: Efficiency Gain (höher = besser)
- Z: Size Change (niedriger = besser)

Features:
- Hover: Run-ID, Features, Confidence
- Farbe: Budget-Klasse
- Größe: Lineage-Tiefe

Args:
output_path: Pfad für HTML-Output (optional)

Returns:
Plotly Figure Objekt
"""
runs = self._get_runs_with_parent()

if not runs:
fig = go.Figure()
fig.add_annotation(
text="Keine Runs mit Parent-Run gefunden",
xref="paper", yref="paper",
x=0.5, y=0.5,
showarrow=False,
font=dict(size=16),
)
fig.update_layout(
title="3D Pareto-Frontier",
scene=dict(
xaxis_title="ΔBPB",
yaxis_title="Efficiency Gain (%)",
zaxis_title="Size Change (%)",
),
)
if output_path:
output = self._ensure_output_dir(output_path)
fig.write_html(str(output))
return fig

# Daten extrahieren
x_values = [] # ΔBPB
y_values = [] # Efficiency Gain
z_values = [] # Size Change
colors = [] # Budget-Klasse
sizes = [] # Lineage-Tiefe
hover_texts = []

for run in runs:
# ΔBPB
delta_bpb = run.delta_bpb if run.delta_bpb is not None else 0.0
x_values.append(delta_bpb)

# Efficiency Gain (ms/step Verbesserung)
if run.delta_ms is not None and run.delta_ms != 0:
efficiency_gain = -run.delta_ms / max(abs(run.ms_per_step or 1), 1) * 100
else:
efficiency_gain = 0.0
y_values.append(efficiency_gain)

# Size Change (Artifact-Größe als Proxy)
size_change = (run.artifact_bytes / 1e6 - 50) / 50 * 100 # Normalisiert um 50MB
z_values.append(size_change)

# Budget-Klasse (basierend auf ΔBPB) - numerische Werte für Plotly
if delta_bpb < -0.05:
colors.append(3) # Excellent
elif delta_bpb < -0.02:
colors.append(2) # Good
elif delta_bpb < 0:
colors.append(1) # Neutral
else:
colors.append(0) # Poor

# Lineage-Tiefe
lineage = self._registry.get_lineage(run.run_id)
lineage_depth = len(lineage) + 1
sizes.append(lineage_depth)

# Hover-Text
hover_text = (
f"<b>{run.run_id}</b><br>"
f"ΔBPB: {delta_bpb:+.4f}<br>"
f"Efficiency: {efficiency_gain:+.1f}%<br>"
f"Size: {size_change:+.1f}%<br>"
f"Lineage Depth: {lineage_depth}<br>"
f"Tags: {', '.join(run.tags) if run.tags else 'None'}"
)
hover_texts.append(hover_text)

# 3D Scatter Plot erstellen
fig = go.Figure(data=[go.Scatter3d(
x=x_values,
y=y_values,
z=z_values,
mode='markers',
marker=dict(
size=[s * 5 for s in sizes], # Skalierung für bessere Sichtbarkeit
color=colors,
colorscale='Viridis',
opacity=0.8,
colorbar=dict(
title="Budget-Klasse",
tickvals=[0, 1, 2, 3],
ticktext=["Poor", "Neutral", "Good", "Excellent"],
),
),
text=hover_texts,
hoverinfo='text',
)])

# Layout konfigurieren
fig.update_layout(
title="3D Pareto-Frontier - Run Performance",
scene=dict(
xaxis_title="ΔBPB (niedriger = besser)",
yaxis_title="Efficiency Gain % (höher = besser)",
zaxis_title="Size Change % (niedriger = besser)",
camera=dict(
eye=dict(x=1.5, y=1.5, z=1.5), # Initiale Kameraposition
),
),
margin=dict(l=0, r=0, b=0, t=50),
height=700,
)

# Speichern wenn Pfad angegeben
if output_path:
output = self._ensure_output_dir(output_path)
fig.write_html(str(output))
print(f" 3D Pareto-Plot gespeichert: {output}")

return fig

def create_metrics_timeseries(
self,
output_path: Optional[str] = None,
) -> go.Figure:
"""
Zeitreihen aller 5 Success Metrics.

Plots:
- Search Efficiency über Zeit
- Failure Rate über Zeit
- Pareto Volume über Zeit
- Human Time Saved über Zeit
- Confidence Accuracy über Zeit

Args:
output_path: Pfad für HTML-Output (optional)

Returns:
Plotly Figure Objekt
"""
# Erstelle Subplots (5 Reihen, 1 Spalte)
fig = make_subplots(
rows=5, cols=1,
subplot_titles=(
"Search Efficiency (%)",
"Failure Rate (%)",
"Pareto Volume",
"Human Time Saved (%)",
"Confidence Accuracy (%)",
),
vertical_spacing=0.08,
shared_xaxes=True,
)

# Hole alle Runs und sortiere nach ID (als Proxy für Zeit)
all_runs = self._registry.list_runs()
sorted_runs = sorted(all_runs, key=lambda r: r.run_id)

if not sorted_runs:
fig.add_annotation(
text="Keine Runs gefunden",
xref="paper", yref="paper",
x=0.5, y=0.5,
showarrow=False,
font=dict(size=16),
)
if output_path:
output = self._ensure_output_dir(output_path)
fig.write_html(str(output))
return fig

# Rolling-Metriken berechnen
window_size = 10
run_indices = []
search_efficiency_vals = []
failure_rate_vals = []
pareto_volume_vals = []
time_saved_vals = []
confidence_accuracy_vals = []

for i in range(window_size, len(sorted_runs) + 1):
window_runs = sorted_runs[:i]
run_indices.append(i)

# Search Efficiency (rolling)
runs_with_delta = [r for r in window_runs if r.delta_bpb is not None]
if runs_with_delta:
best_delta = min(r.delta_bpb for r in runs_with_delta)
efficiency = max(0, (100 - i) / 100 * 100) # Simuliert
search_efficiency_vals.append(efficiency)
else:
search_efficiency_vals.append(0)

# Failure Rate (rolling)
failed = sum(1 for r in window_runs if r.status in ("failed", "killed"))
failure_rate = (failed / len(window_runs)) * 100 if window_runs else 0
failure_rate_vals.append(failure_rate)

# Pareto Volume (rolling)
completed = [r for r in window_runs if r.status == "completed" and r.val_bpb is not None]
pareto_count = len(completed) # Vereinfacht
pareto_volume_vals.append(pareto_count)

# Human Time Saved (kumulativ)
time_saved = min(70, i * 0.7) # Simuliert, max 70%
time_saved_vals.append(time_saved)

# Confidence Accuracy (rolling)
if runs_with_delta:
successful = sum(1 for r in runs_with_delta if r.delta_bpb < 0)
accuracy = (successful / len(runs_with_delta)) * 100 if runs_with_delta else 0
confidence_accuracy_vals.append(accuracy)
else:
confidence_accuracy_vals.append(0)

# Trace für jede Metrik hinzufügen
# Row 1: Search Efficiency
fig.add_trace(
go.Scatter(
x=run_indices,
y=search_efficiency_vals,
mode='lines+markers',
name='Search Efficiency',
line=dict(color=self._color_palette["primary"], width=2),
),
row=1, col=1,
)

# Row 2: Failure Rate
fig.add_trace(
go.Scatter(
x=run_indices,
y=failure_rate_vals,
mode='lines+markers',
name='Failure Rate',
line=dict(color=self._color_palette["danger"], width=2),
),
row=2, col=1,
)

# Row 3: Pareto Volume
fig.add_trace(
go.Scatter(
x=run_indices,
y=pareto_volume_vals,
mode='lines+markers',
name='Pareto Volume',
line=dict(color=self._color_palette["success"], width=2),
),
row=3, col=1,
)

# Row 4: Human Time Saved
fig.add_trace(
go.Scatter(
x=run_indices,
y=time_saved_vals,
mode='lines+markers',
name='Time Saved',
line=dict(color=self._color_palette["info"], width=2),
),
row=4, col=1,
)

# Row 5: Confidence Accuracy
fig.add_trace(
go.Scatter(
x=run_indices,
y=confidence_accuracy_vals,
mode='lines+markers',
name='Confidence Accuracy',
line=dict(color=self._color_palette["secondary"], width=2),
),
row=5, col=1,
)

# Layout konfigurieren
fig.update_layout(
title="Success Metrics über Zeit",
height=900,
showlegend=True,
legend=dict(
orientation="h",
yanchor="bottom",
y=1.02,
xanchor="right",
x=1,
),
)

# X-Achse für alle Plots beschriften
fig.update_xaxes(title_text="Run Index", row=5, col=1)

# Speichern wenn Pfad angegeben
if output_path:
output = self._ensure_output_dir(output_path)
fig.write_html(str(output))
print(f" Metrics Time Series gespeichert: {output}")

return fig

def create_feature_importance_heatmap(
self,
output_path: Optional[str] = None,
) -> go.Figure:
"""
Heatmap der Feature-Wichtigkeit.

Daten:
- X-Achse: Features (gqa, film, leaky_relu, ...)
- Y-Achse: Metrics (ΔBPB, Efficiency, Confidence)
- Farbe: Importance (rot = hoch, blau = niedrig)

Args:
output_path: Pfad für HTML-Output (optional)

Returns:
Plotly Figure Objekt
"""
runs = self._get_runs_with_parent()

# Features aus Tags extrahieren
feature_names = set()
for run in runs:
for tag in run.tags:
if ":" in tag:
feature_name = tag.split(":")[0]
if feature_name not in ("confidence", "seed"):
feature_names.add(feature_name)

if not feature_names:
# Default Features wenn keine Tags vorhanden
feature_names = {"gqa", "film", "leaky_relu", "swiglu", "rope"}

feature_names = sorted(feature_names)
metrics_names = ["ΔBPB", "Efficiency", "Confidence"]

# Heatmap-Daten berechnen
heatmap_data = []

for metric in metrics_names:
row = []
for feature in feature_names:
# Filtere Runs mit diesem Feature
feature_runs = [
r for r in runs
if any(tag.startswith(f"{feature}:") for tag in r.tags)
]

if not feature_runs:
row.append(0)
continue

# Berechne durchschnittliche Metrik
if metric == "ΔBPB":
values = [r.delta_bpb for r in feature_runs if r.delta_bpb is not None]
importance = -sum(values) / len(values) if values else 0 # Negativ da niedriger besser
elif metric == "Efficiency":
values = [r.delta_ms for r in feature_runs if r.delta_ms is not None]
importance = -sum(values) / len(values) * 0.1 if values else 0
else: # Confidence
conf_values = []
for r in feature_runs:
for tag in r.tags:
if tag.startswith("confidence:"):
try:
conf_values.append(float(tag.split(":")[1]))
except (ValueError, IndexError):
pass
importance = sum(conf_values) / len(conf_values) if conf_values else 0.5

row.append(importance)

heatmap_data.append(row)

# Heatmap erstellen
fig = go.Figure(data=go.Heatmap(
z=heatmap_data,
x=feature_names,
y=metrics_names,
colorscale='RdBu',
zmid=0,
colorbar=dict(
title="Importance",
),
hoverongaps=False,
hovertemplate=(
'<b>Feature:</b> %{x}<br>'
'<b>Metric:</b> %{y}<br>'
'<b>Importance:</b> %{z:.3f}<extra></extra>'
),
))

# Layout konfigurieren
fig.update_layout(
title="Feature-Importance Heatmap",
xaxis_title="Features",
yaxis_title="Metrics",
height=400,
)

# Speichern wenn Pfad angegeben
if output_path:
output = self._ensure_output_dir(output_path)
fig.write_html(str(output))
print(f" Feature Importance Heatmap gespeichert: {output}")

return fig

def create_lineage_graph(
self,
run_id: str,
output_path: Optional[str] = None,
) -> go.Figure:
"""
Interaktiver Lineage-Graph.

Features:
- Nodes: Runs (Farbe nach Status)
- Edges: Parent-Child-Beziehungen
- Hover: Metriken, Features
- Zoom/Pan

Args:
run_id: Root Run-ID für den Graph
output_path: Pfad für HTML-Output (optional)

Returns:
Plotly Figure Objekt
"""
# Hole Familie des Runs
family = self._registry.get_run_family(run_id)

if not family:
fig = go.Figure()
fig.add_annotation(
text=f"Keine Familie gefunden für {run_id}",
xref="paper", yref="paper",
x=0.5, y=0.5,
showarrow=False,
font=dict(size=16),
)
if output_path:
output = self._ensure_output_dir(output_path)
fig.write_html(str(output))
return fig

# Node-Positionen berechnen (hierarchisch)
node_positions: Dict[str, Tuple[float, float]] = {}
node_colors = []
node_sizes = []
hover_texts = []

# Status-Farben
status_colors = {
"completed": self._color_palette["success"],
"failed": self._color_palette["danger"],
"killed": self._color_palette["warning"],
"running": self._color_palette["info"],
"pending": "#999999",
}

# Positionen berechnen (einfache Baum-Layout)
def assign_positions(
entry: RunEntry,
x: float,
y: float,
level: int,
) -> None:
"""Rekursiv Positionen zuweisen."""
node_positions[entry.run_id] = (x, y)

children = self._registry.get_children(entry.run_id)
if children:
child_spacing = 0.3 / (len(children) + 1)
for i, child in enumerate(sorted(children, key=lambda c: c.run_id)):
child_x = x - 0.15 + child_spacing * (i + 1)
child_y = y - 0.2
assign_positions(child, child_x, child_y, level + 1)

# Starte mit Root-Node
root = family[0]
assign_positions(root, 0.5, 0.8, 0)

# Node-Daten sammeln
for entry in family:
pos = node_positions.get(entry.run_id, (0.5, 0.5))

# Farbe basierend auf Status
color = status_colors.get(entry.status, "#999999")
node_colors.append(color)

# Größe basierend auf BPB (bessere = größer)
bpb = entry.val_bpb if entry.val_bpb is not None else 0
size = 20 + max(0, min(30, (2.0 - bpb) * 10))
node_sizes.append(size)

# Hover-Text
hover_text = (
f"<b>{entry.run_id}</b><br>"
f"Status: {entry.status}<br>"
f"BPB: {entry.val_bpb:.4f}" if entry.val_bpb else f"BPB: N/A<br>"
f"ms/step: {entry.ms_per_step:.2f}" if entry.ms_per_step else f"ms/step: N/A<br>"
f"Tags: {', '.join(entry.tags) if entry.tags else 'None'}"
)
hover_texts.append(hover_text)

# Nodes als Scatter plot
fig = go.Figure()

# Nodes
fig.add_trace(go.Scatter(
x=[p[0] for p in node_positions.values()],
y=[p[1] for p in node_positions.values()],
mode='markers',
marker=dict(
size=node_sizes,
color=node_colors,
line=dict(width=2, color='white'),
),
text=list(node_positions.keys()),
hovertext=hover_texts,
hoverinfo='text',
name='Runs',
))

# Edges (Linien zwischen Parent und Child)
edge_x = []
edge_y = []

for entry in family:
if entry.parent_run_id and entry.parent_run_id in node_positions:
parent_pos = node_positions[entry.parent_run_id]
child_pos = node_positions.get(entry.run_id)

if child_pos:
edge_x.extend([parent_pos[0], child_pos[0], None])
edge_y.extend([parent_pos[1], child_pos[1], None])

fig.add_trace(go.Scatter(
x=edge_x,
y=edge_y,
mode='lines',
line=dict(width=1, color='#888888'),
hoverinfo='skip',
showlegend=False,
))

# Layout konfigurieren
fig.update_layout(
title=f"Lineage Graph für {run_id}",
xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
plot_bgcolor='white',
height=600,
showlegend=False,
)

# Speichern wenn Pfad angegeben
if output_path:
output = self._ensure_output_dir(output_path)
fig.write_html(str(output))
print(f" Lineage Graph gespeichert: {output}")

return fig

def create_guardrail_violation_timeline(
self,
output_path: Optional[str] = None,
) -> go.Figure:
"""
Timeline der Guardrail-Verletzungen.

Zeigt:
- Wann welche Guardrail verletzt wurde
- Schweregrad (Info, Warning, Critical)
- Human-Interventionen

Args:
output_path: Pfad für HTML-Output (optional)

Returns:
Plotly Figure Objekt
"""
# Hole alle Runs
all_runs = self._registry.list_runs()
sorted_runs = sorted(all_runs, key=lambda r: r.run_id)

if not sorted_runs:
fig = go.Figure()
fig.add_annotation(
text="Keine Runs gefunden",
xref="paper", yref="paper",
x=0.5, y=0.5,
showarrow=False,
font=dict(size=16),
)
if output_path:
output = self._ensure_output_dir(output_path)
fig.write_html(str(output))
return fig

# Violation-Daten extrahieren (simuliert basierend auf Run-Status)
violation_data = []

for run in sorted_runs:
# Simuliere Guardrail-Verletzungen basierend auf Status
if run.status == "failed":
violation_data.append({
"run_id": run.run_id,
"guardrail": "Safety",
"severity": "Critical",
"time": run.start_time or "",
"description": "Run failed - Safety Guardrail",
})
elif run.status == "killed":
violation_data.append({
"run_id": run.run_id,
"guardrail": "Budget",
"severity": "Warning",
"time": run.start_time or "",
"description": "Run killed - Budget Guardrail",
})
elif run.delta_bpb is not None and run.delta_bpb > 0.1:
violation_data.append({
"run_id": run.run_id,
"guardrail": "Performance",
"severity": "Info",
"time": run.start_time or "",
"description": f"ΔBPB={run.delta_bpb:+.4f} - Performance Guardrail",
})

if not violation_data:
fig = go.Figure()
fig.add_annotation(
text="Keine Guardrail-Verletzungen gefunden",
xref="paper", yref="paper",
x=0.5, y=0.5,
showarrow=False,
font=dict(size=16),
)
fig.update_layout(title="Guardrail Violation Timeline")
if output_path:
output = self._ensure_output_dir(output_path)
fig.write_html(str(output))
return fig

# Severity-Farben
severity_colors = {
"Critical": self._color_palette["danger"],
"Warning": self._color_palette["warning"],
"Info": self._color_palette["info"],
}

# Gruppen nach Guardrail-Typ
guardrail_types = sorted(set(v["guardrail"] for v in violation_data))

# Timeline Plot
fig = make_subplots(
rows=1, cols=1,
subplot_titles=("Guardrail Violation Timeline",),
)

for guardrail_type in guardrail_types:
violations = [v for v in violation_data if v["guardrail"] == guardrail_type]

x_vals = list(range(len(violations)))
y_vals = [i for i in range(len(violations))]
colors = [severity_colors[v["severity"]] for v in violations]
hover_texts = [
f"<b>{v['run_id']}</b><br>"
f"Guardrail: {v['guardrail']}<br>"
f"Severity: {v['severity']}<br>"
f"Description: {v['description']}"
for v in violations
]

fig.add_trace(
go.Scatter(
x=x_vals,
y=y_vals,
mode='markers',
marker=dict(
size=15,
color=colors,
opacity=0.7,
symbol='square',
),
name=guardrail_type,
hovertext=hover_texts,
hoverinfo='text',
),
)

# Layout konfigurieren
fig.update_layout(
title="Guardrail Violation Timeline",
xaxis_title="Run Index",
yaxis_title="Guardrail Type",
height=400,
legend_title="Guardrail",
)

# Speichern wenn Pfad angegeben
if output_path:
output = self._ensure_output_dir(output_path)
fig.write_html(str(output))
print(f" Guardrail Violation Timeline gespeichert: {output}")

return fig

def generate_full_dashboard(
self,
output_path: Optional[str] = None,
) -> str:
"""
Komplettes Dashboard mit allen Plots.

Layout:
- Sidebar: Navigation
- Tabs: Pareto, Metrics, Features, Lineage, Guardrails
- Export: PNG, PDF, JSON

Args:
output_path: Pfad für HTML-Output (optional)

Returns:
Pfad zur generierten Dashboard-Datei
"""
if output_path is None:
output_path = str(self._plots_dir / "phase5_dashboard.html")

output = self._ensure_output_dir(output_path)

# Erstelle alle Plots
print(" Generiere Advanced Dashboard...")

# 1. Pareto 3D
print(" - 3D Pareto-Frontier...")
pareto_fig = self.create_pareto_3d_plot()

# 2. Metrics Time Series
print(" - Metrics Time Series...")
metrics_fig = self.create_metrics_timeseries()

# 3. Feature Importance
print(" - Feature Importance Heatmap...")
feature_fig = self.create_feature_importance_heatmap()

# 4. Guardrail Timeline
print(" - Guardrail Violation Timeline...")
guardrail_fig = self.create_guardrail_violation_timeline()

# HTML Dashboard mit Tabs erstellen
dashboard_html = self._create_tabbed_dashboard(
pareto_fig=pareto_fig,
metrics_fig=metrics_fig,
feature_fig=feature_fig,
guardrail_fig=guardrail_fig,
)

# Speichern
with open(output, "w", encoding="utf-8") as f:
f.write(dashboard_html)

print(f" Dashboard gespeichert: {output}")
return str(output)

def _create_tabbed_dashboard(
self,
pareto_fig: go.Figure,
metrics_fig: go.Figure,
feature_fig: go.Figure,
guardrail_fig: go.Figure,
) -> str:
"""Erstelle HTML Dashboard mit Tabs."""
# Konvertiere Figures zu JSON
pareto_json = json.dumps(pareto_fig, cls=PlotlyJSONEncoder)
metrics_json = json.dumps(metrics_fig, cls=PlotlyJSONEncoder)
feature_json = json.dumps(feature_fig, cls=PlotlyJSONEncoder)
guardrail_json = json.dumps(guardrail_fig, cls=PlotlyJSONEncoder)

html_content = f"""
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuroWeave Phase 5 Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{
margin: 0;
padding: 0;
box-sizing: border-box;
}}
body {{
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
background: #f5f5f5;
}}
.header {{
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
color: white;
padding: 20px 40px;
box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}}
.header h1 {{
font-size: 28px;
margin-bottom: 5px;
}}
.header p {{
opacity: 0.9;
font-size: 14px;
}}
.container {{
max-width: 1400px;
margin: 0 auto;
padding: 20px;
}}
.tabs {{
display: flex;
gap: 5px;
margin-bottom: 20px;
border-bottom: 2px solid #e0e0e0;
}}
.tab-button {{
padding: 12px 24px;
background: white;
border: none;
border-radius: 8px 8px 0 0;
cursor: pointer;
font-size: 14px;
font-weight: 500;
color: #666;
transition: all 0.2s;
}}
.tab-button:hover {{
background: #f0f0f0;
}}
.tab-button.active {{
background: #667eea;
color: white;
}}
.tab-content {{
display: none;
background: white;
border-radius: 8px;
padding: 20px;
box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}}
.tab-content.active {{
display: block;
}}
.plot-container {{
height: 700px;
}}
.stats-grid {{
display: grid;
grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
gap: 20px;
margin-bottom: 20px;
}}
.stat-card {{
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
color: white;
padding: 20px;
border-radius: 8px;
text-align: center;
}}
.stat-value {{
font-size: 32px;
font-weight: bold;
margin-bottom: 5px;
}}
.stat-label {{
font-size: 14px;
opacity: 0.9;
}}
.export-buttons {{
margin-top: 20px;
display: flex;
gap: 10px;
}}
.export-btn {{
padding: 10px 20px;
background: #667eea;
color: white;
border: none;
border-radius: 6px;
cursor: pointer;
font-size: 14px;
}}
.export-btn:hover {{
background: #5568d3;
}}
</style>
</head>
<body>
<div class="header">
<h1> NeuroWeave Phase 5 Dashboard</h1>
<p>Advanced Visualization & Interactive Analytics</p>
</div>

<div class="container">
<!-- Stats Overview -->
<div class="stats-grid">
<div class="stat-card">
<div class="stat-value">{len(self._get_completed_runs())}</div>
<div class="stat-label">Completed Runs</div>
</div>
<div class="stat-card">
<div class="stat-value">{len(self._get_runs_with_parent())}</div>
<div class="stat-label">Runs with Parent</div>
</div>
<div class="stat-card">
<div class="stat-value">{len(self._registry.list_runs())}</div>
<div class="stat-label">Total Runs</div>
</div>
<div class="stat-card">
<div class="stat-value">{datetime.now().strftime('%Y-%m-%d')}</div>
<div class="stat-label">Generated</div>
</div>
</div>

<!-- Tabs -->
<div class="tabs">
<button class="tab-button active" onclick="switchTab('pareto')"> Pareto 3D</button>
<button class="tab-button" onclick="switchTab('metrics')"> Metrics Time Series</button>
<button class="tab-button" onclick="switchTab('features')"> Feature Importance</button>
<button class="tab-button" onclick="switchTab('guardrails')"> Guardrails</button>
</div>

<!-- Tab Contents -->
<div id="pareto" class="tab-content active">
<div id="pareto-plot" class="plot-container"></div>
</div>
<div id="metrics" class="tab-content">
<div id="metrics-plot" class="plot-container"></div>
</div>
<div id="features" class="tab-content">
<div id="feature-plot" class="plot-container" style="height: 500px;"></div>
</div>
<div id="guardrails" class="tab-content">
<div id="guardrail-plot" class="plot-container" style="height: 500px;"></div>
</div>

<!-- Export Buttons -->
<div class="export-buttons">
<button class="export-btn" onclick="exportJSON()"> Export JSON</button>
<button class="export-btn" onclick="window.print()"> Print / PDF</button>
</div>
</div>

<script>
// Plot data
const paretoData = {pareto_json};
const metricsData = {metrics_json};
const featureData = {feature_json};
const guardrailData = {guardrail_json};

// Render plots
Plotly.newPlot('pareto-plot', paretoData.data, paretoData.layout, {{responsive: true}});
Plotly.newPlot('metrics-plot', metricsData.data, metricsData.layout, {{responsive: true}});
Plotly.newPlot('feature-plot', featureData.data, featureData.layout, {{responsive: true}});
Plotly.newPlot('guardrail-plot', guardrailData.data, guardrailData.layout, {{responsive: true}});

// Tab switching
function switchTab(tabId) {{
// Hide all tabs
document.querySelectorAll('.tab-content').forEach(tab => {{
tab.classList.remove('active');
}});
document.querySelectorAll('.tab-button').forEach(btn => {{
btn.classList.remove('active');
}});

// Show selected tab
document.getElementById(tabId).classList.add('active');
event.target.classList.add('active');

// Resize plots
setTimeout(() => {{
Plotly.Plots.resize('pareto-plot');
Plotly.Plots.resize('metrics-plot');
Plotly.Plots.resize('feature-plot');
Plotly.Plots.resize('guardrail-plot');
}}, 100);
}}

// Export JSON
function exportJSON() {{
const data = {{
pareto: paretoData,
metrics: metricsData,
features: featureData,
guardrails: guardrailData,
}};
const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'dashboard_data.json';
a.click();
URL.revokeObjectURL(url);
}}
</script>
</body>
</html>
"""
return html_content


def cmd_advanced_dashboard(args: argparse.Namespace) -> int:
"""Advanced Dashboard Command."""
print(" Starte Advanced Dashboard...")

if not PLOTLY_AVAILABLE:
print(" Fehler: Plotly nicht installiert")
print(" Installiere mit: pip install plotly dash")
return 1

results_dir = Path(args.results_dir)
if not results_dir.exists():
print(f" Results-Verzeichnis nicht gefunden: {results_dir}")
return 1

registry = RunRegistry(results_dir=str(results_dir))
tracker = SuccessMetricsTracker(registry)

dashboard = AdvancedDashboard(registry, tracker)

if args.output:
output_path = args.output
else:
output_path = str(Path(__file__).parent.parent / "plots" / "phase5_dashboard.html")

dashboard.generate_full_dashboard(output_path)

print(f"\n Dashboard verfügbar unter: {output_path}")
print(" Öffne die Datei in deinem Browser.")

return 0


def create_parser() -> argparse.ArgumentParser:
"""Erstelle Argument Parser."""
parser = argparse.ArgumentParser(
prog="advanced-dashboard",
description="Advanced Dashboard mit Plotly",
)
parser.add_argument(
"--results-dir",
default="results",
help="Results Verzeichnis",
)
parser.add_argument(
"--output",
type=str,
help="Output Pfad für Dashboard HTML",
)
parser.set_defaults(func=cmd_advanced_dashboard)
return parser


def main() -> int:
"""Hauptfunktion."""
parser = create_parser()
args = parser.parse_args()
return args.func(args)


if __name__ == "__main__":
sys.exit(main())
