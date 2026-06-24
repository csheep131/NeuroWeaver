#!/usr/bin/env python3
"""
Real-time Monitor für NeuroWeave Phase 5.

Live-Metriken während Training mit WebSocket-Unterstützung.

Features:
- Live Loss-Curve (streaming)
- Gradient Norm Monitoring
- VRAM-Usage (live)
- Step-Time (rolling average)
- Anomaly-Erkennung (live)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Füge Parent-Directory zum Path hinzu für Imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
import websockets
from websockets.server import WebSocketServerProtocol
WEBSOCKETS_AVAILABLE = True
except ImportError:
WEBSOCKETS_AVAILABLE = False
print("Warnung: websockets nicht installiert. Installiere mit: pip install websockets")

try:
import numpy as np
NUMPY_AVAILABLE = True
except ImportError:
NUMPY_AVAILABLE = False
print("Warnung: numpy nicht installiert. Installiere mit: pip install numpy")


@dataclass
class LiveMetrics:
"""Live-Metriken für einen Run."""

run_id: str
step: int = 0
loss: float = 0.0
gradient_norm: float = 0.0
vram_usage_mb: float = 0.0
step_time_ms: float = 0.0
learning_rate: float = 0.0
anomaly_detected: bool = False
anomaly_score: float = 0.0
timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

def to_dict(self) -> Dict[str, Any]:
"""Konvertiere zu Dictionary."""
return {
"run_id": self.run_id,
"step": self.step,
"loss": self.loss,
"gradient_norm": self.gradient_norm,
"vram_usage_mb": self.vram_usage_mb,
"step_time_ms": self.step_time_ms,
"learning_rate": self.learning_rate,
"anomaly_detected": self.anomaly_detected,
"anomaly_score": self.anomaly_score,
"timestamp": self.timestamp,
}

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "LiveMetrics":
"""Erstelle aus Dictionary."""
return cls(
run_id=data.get("run_id", ""),
step=data.get("step", 0),
loss=data.get("loss", 0.0),
gradient_norm=data.get("gradient_norm", 0.0),
vram_usage_mb=data.get("vram_usage_mb", 0.0),
step_time_ms=data.get("step_time_ms", 0.0),
learning_rate=data.get("learning_rate", 0.0),
anomaly_detected=data.get("anomaly_detected", False),
anomaly_score=data.get("anomaly_score", 0.0),
timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
)


@dataclass
class TrainingSession:
"""Training-Session für Monitoring."""

run_id: str
start_time: datetime
metrics_history: List[LiveMetrics] = field(default_factory=list)
connected_clients: List[WebSocketServerProtocol] = field(default_factory=list)
is_running: bool = True
total_steps: int = 0

def add_metrics(self, metrics: LiveMetrics) -> None:
"""Füge Metriken zur Historie hinzu."""
self.metrics_history.append(metrics)
self.total_steps = metrics.step

def get_latest_metrics(self) -> Optional[LiveMetrics]:
"""Hole neueste Metriken."""
if not self.metrics_history:
return None
return self.metrics_history[-1]

def get_metrics_window(self, window_size: int = 100) -> List[LiveMetrics]:
"""Hole Fenster der neuesten Metriken."""
if not self.metrics_history:
return []
return self.metrics_history[-window_size:]


class AnomalyDetector:
"""
Echtzeit-Anomalieerkennung für Training.

Erkennt:
- Loss Divergence (plötzlicher Anstieg)
- Gradient Explosion (norm > threshold)
- Step-Time Anomalien (>50% langsamer)
- VRAM Leak (kontinuierlicher Anstieg)
"""

def __init__(
self,
loss_threshold: float = 3.0,
gradient_threshold: float = 1000.0,
step_time_threshold: float = 1.5,
vram_increase_threshold: float = 100.0,
window_size: int = 20,
) -> None:
"""
Initialisiere Anomaly Detector.

Args:
loss_threshold: Std-Devs für Loss-Anomalie
gradient_threshold: Absoluter Threshold für Gradient Norm
step_time_threshold: Faktor für Step-Time-Anomalie
vram_increase_threshold: MB pro Step für VRAM Leak
window_size: Fenstergröße für Statistik-Berechnung
"""
self._loss_threshold = loss_threshold
self._gradient_threshold = gradient_threshold
self._step_time_threshold = step_time_threshold
self._vram_increase_threshold = vram_increase_threshold
self._window_size = window_size

def detect_anomaly(self, metrics: LiveMetrics, history: List[LiveMetrics]) -> Tuple[bool, float]:
"""
Erkenne Anomalien in Metriken.

Args:
metrics: Aktuelle Metriken
history: Historie der Metriken

Returns:
Tuple aus (anomaly_detected, anomaly_score)
"""
if len(history) < self._window_size:
return False, 0.0

anomaly_scores = []

# 1. Loss Divergence
loss_scores = self._detect_loss_anomaly(metrics, history)
anomaly_scores.append(loss_scores)

# 2. Gradient Explosion
gradient_scores = self._detect_gradient_anomaly(metrics, history)
anomaly_scores.append(gradient_scores)

# 3. Step-Time Anomaly
step_time_scores = self._detect_step_time_anomaly(metrics, history)
anomaly_scores.append(step_time_scores)

# 4. VRAM Leak
vram_scores = self._detect_vram_anomaly(metrics, history)
anomaly_scores.append(vram_scores)

# Maximaler Score
max_score = max(anomaly_scores) if anomaly_scores else 0.0
is_anomaly = max_score > 0.5

return is_anomaly, max_score

def _detect_loss_anomaly(self, metrics: LiveMetrics, history: List[LiveMetrics]) -> float:
"""Erkenne Loss-Anomalien."""
recent_losses = [m.loss for m in history[-self._window_size:]]

if not recent_losses:
return 0.0

mean_loss = sum(recent_losses) / len(recent_losses)
std_loss = (sum((x - mean_loss) ** 2 for x in recent_losses) / len(recent_losses)) ** 0.5

if std_loss == 0:
return 0.0

z_score = abs(metrics.loss - mean_loss) / std_loss
return min(1.0, z_score / self._loss_threshold)

def _detect_gradient_anomaly(self, metrics: LiveMetrics, history: List[LiveMetrics]) -> float:
"""Erkenne Gradient-Anomalien."""
if metrics.gradient_norm > self._gradient_threshold:
return 1.0

if not history:
return 0.0

recent_gradients = [m.gradient_norm for m in history[-self._window_size:]]
if not recent_gradients:
return 0.0

mean_grad = sum(recent_gradients) / len(recent_gradients)
if mean_grad == 0:
return 0.0

ratio = metrics.gradient_norm / mean_grad
return min(1.0, max(0.0, (ratio - 2.0) / 3.0))

def _detect_step_time_anomaly(self, metrics: LiveMetrics, history: List[LiveMetrics]) -> float:
"""Erkenne Step-Time-Anomalien."""
if not history:
return 0.0

recent_times = [m.step_time_ms for m in history[-self._window_size:]]
if not recent_times:
return 0.0

mean_time = sum(recent_times) / len(recent_times)
if mean_time == 0:
return 0.0

ratio = metrics.step_time_ms / mean_time
if ratio > self._step_time_threshold:
return min(1.0, (ratio - self._step_time_threshold) / 2.0)

return 0.0

def _detect_vram_anomaly(self, metrics: LiveMetrics, history: List[LiveMetrics]) -> float:
"""Erkenne VRAM-Anomalien."""
if len(history) < 2:
return 0.0

# Berechne VRAM-Anstieg pro Step
recent_vram = [m.vram_usage_mb for m in history[-self._window_size:]]
if len(recent_vram) < 2:
return 0.0

# Linearer Fit für Trend
n = len(recent_vram)
x_mean = (n - 1) / 2
y_mean = sum(recent_vram) / n

numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent_vram))
denominator = sum((i - x_mean) ** 2 for i in range(n))

if denominator == 0:
return 0.0

slope = numerator / denominator # MB pro Step

if slope > self._vram_increase_threshold:
return min(1.0, slope / (self._vram_increase_threshold * 2))

return 0.0


class RealtimeMonitor:
"""
Live-Monitoring von Runs.

Features:
- Live Loss-Curve (streaming)
- Gradient Norm Monitoring
- VRAM-Usage (live)
- Step-Time (rolling average)
- Anomaly-Erkennung (live)

Example:
monitor = RealtimeMonitor("run001", websocket_port=8765)

# Monitoring starten (im Hintergrund)
monitor.start_monitoring()

# Metriken aktualisieren (vom Training)
monitor.update_metrics(LiveMetrics(
run_id="run001",
step=100,
loss=1.234,
gradient_norm=50.0,
vram_usage_mb=6000,
step_time_ms=150.0,
))

# Live Dashboard öffnen
monitor.create_live_dashboard("plots/live_monitor.html")
"""

def __init__(
self,
run_id: str,
websocket_port: int = 8765,
sampling_interval: int = 10,
) -> None:
"""
Initialisiere Real-time Monitor.

Args:
run_id: Run-ID zu überwachen
websocket_port: Port für WebSocket-Server
sampling_interval: Sampling alle N Steps
"""
self._run_id = run_id
self._websocket_port = websocket_port
self._sampling_interval = sampling_interval

self._session = TrainingSession(
run_id=run_id,
start_time=datetime.utcnow(),
)

self._anomaly_detector = AnomalyDetector()
self._server_thread: Optional[threading.Thread] = None
self._is_running = False

self._plots_dir = Path(__file__).parent.parent / "plots"
self._plots_dir.mkdir(parents=True, exist_ok=True)

# Callbacks für Metriken-Updates
self._update_callbacks: List[Callable[[LiveMetrics], None]] = []

@property
def run_id(self) -> str:
"""Run-ID zurückgeben."""
return self._run_id

@property
def session(self) -> TrainingSession:
"""Training-Session zurückgeben."""
return self._session

def update_metrics(self, metrics: LiveMetrics) -> None:
"""
Aktualisiere Metriken.

Args:
metrics: Neue Live-Metriken
"""
# Anomalie-Erkennung
is_anomaly, anomaly_score = self._anomaly_detector.detect_anomaly(
metrics,
self._session.metrics_history,
)
metrics.anomaly_detected = is_anomaly
metrics.anomaly_score = anomaly_score
metrics.timestamp = datetime.utcnow().isoformat()

# Zur Historie hinzufügen
self._session.add_metrics(metrics)

# Callbacks aufrufen
for callback in self._update_callbacks:
try:
callback(metrics)
except Exception:
pass

# An Clients broadcasten (wenn Server läuft)
if self._is_running:
asyncio.new_event_loop().run_until_complete(
self._broadcast_metrics(metrics)
)

def register_callback(self, callback: Callable[[LiveMetrics], None]) -> None:
"""
Registriere Callback für Metriken-Updates.

Args:
callback: Funktion die bei jedem Update aufgerufen wird
"""
self._update_callbacks.append(callback)

def get_live_metrics(self) -> Dict[str, Any]:
"""
Aktuelle Live-Metriken.

Returns:
Dictionary mit aktuellen Metriken
"""
latest = self._session.get_latest_metrics()
if not latest:
return {
"run_id": self._run_id,
"step": 0,
"loss": 0.0,
"gradient_norm": 0.0,
"vram_usage_mb": 0.0,
"step_time_ms": 0.0,
"anomaly_detected": False,
"status": "no_data",
}

return {
"run_id": latest.run_id,
"step": latest.step,
"loss": latest.loss,
"gradient_norm": latest.gradient_norm,
"vram_usage_mb": latest.vram_usage_mb,
"step_time_ms": latest.step_time_ms,
"learning_rate": latest.learning_rate,
"anomaly_detected": latest.anomaly_detected,
"anomaly_score": latest.anomaly_score,
"status": "running" if self._session.is_running else "stopped",
"timestamp": latest.timestamp,
}

def get_metrics_history(self, window_size: int = 100) -> List[Dict[str, Any]]:
"""
Historie der Metriken.

Args:
window_size: Anzahl der zurückzugebenden Schritte

Returns:
Liste von Metriken-Dictionaries
"""
window = self._session.get_metrics_window(window_size)
return [m.to_dict() for m in window]

async def _broadcast_metrics(self, metrics: LiveMetrics) -> None:
"""Sende Metriken an alle verbundenen Clients."""
if not self._session.connected_clients:
return

message = json.dumps({
"type": "metrics_update",
"data": metrics.to_dict(),
})

# Broadcast an alle Clients
disconnected = []
for client in self._session.connected_clients:
try:
await client.send(message)
except Exception:
disconnected.append(client)

# Getrennte Clients entfernen
for client in disconnected:
self._session.connected_clients.remove(client)

async def _websocket_handler(
self,
websocket: WebSocketServerProtocol,
path: str,
) -> None:
"""WebSocket-Handler für Client-Verbindungen."""
# Client registrieren
self._session.connected_clients.append(websocket)
print(f" Client verbunden: {websocket.remote_address}")

# Sende aktuelle Metriken
latest = self._session.get_latest_metrics()
if latest:
await websocket.send(json.dumps({
"type": "initial_data",
"data": latest.to_dict(),
"history": self.get_metrics_history(100),
}))

try:
# Auf Nachrichten warten
async for message in websocket:
try:
data = json.loads(message)
msg_type = data.get("type")

if msg_type == "get_history":
# Historie anfordern
window_size = data.get("window_size", 100)
history = self.get_metrics_history(window_size)
await websocket.send(json.dumps({
"type": "history",
"data": history,
}))

elif msg_type == "ping":
await websocket.send(json.dumps({"type": "pong"}))

except json.JSONDecodeError:
await websocket.send(json.dumps({
"type": "error",
"message": "Invalid JSON",
}))

except websockets.exceptions.ConnectionClosed:
pass
finally:
# Client entfernen
if websocket in self._session.connected_clients:
self._session.connected_clients.remove(websocket)
print(f" Client getrennt: {websocket.remote_address}")

def _run_websocket_server(self) -> None:
"""Starte WebSocket-Server (im Hintergrund-Thread)."""
if not WEBSOCKETS_AVAILABLE:
print(" websockets nicht verfügbar")
return

async def start_server():
async with websockets.serve(
self._websocket_handler,
"localhost",
self._websocket_port,
):
print(f" WebSocket-Server läuft auf ws://localhost:{self._websocket_port}")
await asyncio.Future() # Für immer laufen

try:
asyncio.run(start_server())
except Exception as e:
print(f" WebSocket-Server Fehler: {e}")

def start_monitoring(self) -> None:
"""
Monitoring starten.

Startet:
- WebSocket Server für Browser-Client
- Metriken-Sampling (alle 10 Steps)
- Anomaly-Detection-Thread
"""
if self._is_running:
print(" Monitoring läuft bereits")
return

self._is_running = True
self._session.is_running = True

# WebSocket-Server in Hintergrund-Thread starten
self._server_thread = threading.Thread(
target=self._run_websocket_server,
daemon=True,
)
self._server_thread.start()

print(f" Monitoring gestartet für Run: {self._run_id}")
print(f" WebSocket: ws://localhost:{self._websocket_port}")

def stop_monitoring(self) -> None:
"""Monitoring stoppen."""
self._is_running = False
self._session.is_running = False
print(" Monitoring gestoppt")

def create_live_dashboard(
self,
output_path: Optional[str] = None,
) -> str:
"""
Live Dashboard (WebSocket-basiert).

Features:
- Auto-refresh (1s)
- Zoom auf Zeitbereich
- Alert bei Anomalien
- Export der Session

Args:
output_path: Pfad für HTML-Output (optional)

Returns:
Pfad zur generierten Dashboard-Datei
"""
if output_path is None:
output_path = str(self._plots_dir / f"live_monitor_{self._run_id}.html")

output = Path(output_path)
output.parent.mkdir(parents=True, exist_ok=True)

# HTML Dashboard erstellen
dashboard_html = self._create_dashboard_html()

with open(output, "w", encoding="utf-8") as f:
f.write(dashboard_html)

print(f" Live Dashboard gespeichert: {output}")
print(f" Öffne die Datei in deinem Browser.")
return str(output)

def _create_dashboard_html(self) -> str:
"""Erstelle HTML für Live Dashboard."""
html_content = f"""
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Monitor - {self._run_id}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{
margin: 0;
padding: 0;
box-sizing: border-box;
}}
body {{
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
background: #1a1a2e;
color: #eee;
padding: 20px;
}}
.header {{
display: flex;
justify-content: space-between;
align-items: center;
margin-bottom: 20px;
padding: 20px;
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
border-radius: 12px;
}}
.header h1 {{
font-size: 24px;
}}
.status {{
display: flex;
align-items: center;
gap: 10px;
}}
.status-indicator {{
width: 12px;
height: 12px;
border-radius: 50%;
background: #2ca02c;
animation: pulse 2s infinite;
}}
.status-indicator.anomaly {{
background: #d62728;
}}
@keyframes pulse {{
0%, 100% {{ opacity: 1; }}
50% {{ opacity: 0.5; }}
}}
.metrics-grid {{
display: grid;
grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
gap: 15px;
margin-bottom: 20px;
}}
.metric-card {{
background: #16213e;
padding: 20px;
border-radius: 8px;
text-align: center;
}}
.metric-value {{
font-size: 28px;
font-weight: bold;
color: #667eea;
margin-bottom: 5px;
}}
.metric-label {{
font-size: 12px;
color: #888;
text-transform: uppercase;
}}
.metric-unit {{
font-size: 14px;
color: #888;
}}
.plots-container {{
display: grid;
grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
gap: 20px;
}}
.plot-card {{
background: #16213e;
padding: 20px;
border-radius: 12px;
}}
.plot-card h3 {{
margin-bottom: 15px;
font-size: 16px;
color: #667eea;
}}
.alert-banner {{
background: #d62728;
color: white;
padding: 15px 20px;
border-radius: 8px;
margin-bottom: 20px;
display: none;
}}
.alert-banner.show {{
display: block;
animation: flash 1s infinite;
}}
@keyframes flash {{
0%, 100% {{ opacity: 1; }}
50% {{ opacity: 0.7; }}
}}
.controls {{
display: flex;
gap: 10px;
margin-bottom: 20px;
}}
.btn {{
padding: 10px 20px;
background: #667eea;
color: white;
border: none;
border-radius: 6px;
cursor: pointer;
font-size: 14px;
}}
.btn:hover {{
background: #5568d3;
}}
.btn.danger {{
background: #d62728;
}}
</style>
</head>
<body>
<div class="header">
<div>
<h1> Live Monitor: {self._run_id}</h1>
<p style="opacity: 0.8; font-size: 14px; margin-top: 5px;">
WebSocket: ws://localhost:{self._websocket_port}
</p>
</div>
<div class="status">
<div class="status-indicator" id="statusIndicator"></div>
<span id="statusText">Verbunden</span>
</div>
</div>

<div class="alert-banner" id="alertBanner">
<strong>Anomalie erkannt!</strong> <span id="alertMessage"></span>
</div>

<div class="metrics-grid">
<div class="metric-card">
<div class="metric-value" id="stepValue">0</div>
<div class="metric-label">Step</div>
</div>
<div class="metric-card">
<div class="metric-value" id="lossValue">0.000</div>
<div class="metric-label">Loss</div>
</div>
<div class="metric-card">
<div class="metric-value" id="gradientValue">0.0</div>
<div class="metric-label">Grad Norm</div>
</div>
<div class="metric-card">
<div class="metric-value" id="vramValue">0</div>
<div class="metric-unit">MB</div>
<div class="metric-label">VRAM</div>
</div>
<div class="metric-card">
<div class="metric-value" id="stepTimeValue">0.0</div>
<div class="metric-unit">ms</div>
<div class="metric-label">Step Time</div>
</div>
<div class="metric-card">
<div class="metric-value" id="anomalyValue">Nein</div>
<div class="metric-label">Anomalie</div>
</div>
</div>

<div class="controls">
<button class="btn" onclick="exportData()"> Export JSON</button>
<button class="btn" onclick="resetZoom()"> Reset Zoom</button>
<button class="btn danger" onclick="stopMonitoring()"> Stop</button>
</div>

<div class="plots-container">
<div class="plot-card">
<h3> Loss Curve</h3>
<div id="lossPlot" style="height: 400px;"></div>
</div>
<div class="plot-card">
<h3> Gradient Norm</h3>
<div id="gradientPlot" style="height: 400px;"></div>
</div>
<div class="plot-card">
<h3> VRAM Usage</h3>
<div id="vramPlot" style="height: 400px;"></div>
</div>
<div class="plot-card">
<h3> Step Time</h3>
<div id="stepTimePlot" style="height: 400px;"></div>
</div>
</div>

<script>
// WebSocket Verbindung
let ws;
let metricsHistory = [];
const wsUrl = 'ws://localhost:{self._websocket_port}';

function connectWebSocket() {{
ws = new WebSocket(wsUrl);

ws.onopen = () => {{
console.log('Verbunden mit WebSocket');
document.getElementById('statusIndicator').classList.remove('anomaly');
document.getElementById('statusText').textContent = 'Verbunden';
}};

ws.onmessage = (event) => {{
const message = JSON.parse(event.data);

if (message.type === 'metrics_update') {{
updateMetrics(message.data);
}} else if (message.type === 'initial_data') {{
metricsHistory = message.history || [];
updateMetrics(message.data);
updatePlots();
}}
}};

ws.onclose = () => {{
console.log('WebSocket getrennt');
document.getElementById('statusIndicator').classList.add('anomaly');
document.getElementById('statusText').textContent = 'Getrennt';
// Automatischer Reconnect nach 3s
setTimeout(connectWebSocket, 3000);
}};

ws.onerror = (error) => {{
console.error('WebSocket Fehler:', error);
}};
}}

function updateMetrics(data) {{
// Metriken aktualisieren
document.getElementById('stepValue').textContent = data.step;
document.getElementById('lossValue').textContent = data.loss.toFixed(4);
document.getElementById('gradientValue').textContent = data.gradient_norm.toFixed(1);
document.getElementById('vramValue').textContent = Math.round(data.vram_usage_mb);
document.getElementById('stepTimeValue').textContent = data.step_time_ms.toFixed(1);

const anomalyText = data.anomaly_detected ? 'JA' : 'Nein';
const anomalyEl = document.getElementById('anomalyValue');
anomalyEl.textContent = anomalyText;
anomalyEl.style.color = data.anomaly_detected ? '#d62728' : '#2ca02c';

// Alert Banner
const alertBanner = document.getElementById('alertBanner');
if (data.anomaly_detected) {{
alertBanner.classList.add('show');
document.getElementById('alertMessage').textContent =
`Score: ${(data.anomaly_score * 100).toFixed(0)}%`;
document.getElementById('statusIndicator').classList.add('anomaly');
}} else {{
alertBanner.classList.remove('show');
document.getElementById('statusIndicator').classList.remove('anomaly');
}}

// Zur Historie hinzufügen
metricsHistory.push(data);
if (metricsHistory.length > 1000) {{
metricsHistory.shift();
}}

updatePlots();
}}

function updatePlots() {{
if (metricsHistory.length === 0) return;

const steps = metricsHistory.map(m => m.step);
const losses = metricsHistory.map(m => m.loss);
const gradients = metricsHistory.map(m => m.gradient_norm);
const vrams = metricsHistory.map(m => m.vram_usage_mb);
const stepTimes = metricsHistory.map(m => m.step_time_ms);

// Loss Plot
Plotly.react('lossPlot', [{{
x: steps,
y: losses,
type: 'scatter',
mode: 'lines',
line: {{color: '#667eea', width: 2}}
}}], {{
margin: {{t: 20, b: 40, l: 50, r: 20}},
xaxis: {{title: 'Step'}},
yaxis: {{title: 'Loss'}},
paper_bgcolor: 'rgba(0,0,0,0)',
plot_bgcolor: 'rgba(0,0,0,0)',
}});

// Gradient Plot
Plotly.react('gradientPlot', [{{
x: steps,
y: gradients,
type: 'scatter',
mode: 'lines',
line: {{color: '#ff7f0e', width: 2}}
}}], {{
margin: {{t: 20, b: 40, l: 50, r: 20}},
xaxis: {{title: 'Step'}},
yaxis: {{title: 'Gradient Norm'}},
paper_bgcolor: 'rgba(0,0,0,0)',
plot_bgcolor: 'rgba(0,0,0,0)',
}});

// VRAM Plot
Plotly.react('vramPlot', [{{
x: steps,
y: vrams,
type: 'scatter',
mode: 'lines',
line: {{color: '#2ca02c', width: 2}}
}}], {{
margin: {{t: 20, b: 40, l: 50, r: 20}},
xaxis: {{title: 'Step'}},
yaxis: {{title: 'VRAM (MB)'}},
paper_bgcolor: 'rgba(0,0,0,0)',
plot_bgcolor: 'rgba(0,0,0,0)',
}});

// Step Time Plot
Plotly.react('stepTimePlot', [{{
x: steps,
y: stepTimes,
type: 'scatter',
mode: 'lines',
line: {{color: '#00bcd4', width: 2}}
}}], {{
margin: {{t: 20, b: 40, l: 50, r: 20}},
xaxis: {{title: 'Step'}},
yaxis: {{title: 'Step Time (ms)'}},
paper_bgcolor: 'rgba(0,0,0,0)',
plot_bgcolor: 'rgba(0,0,0,0)',
}});
}}

function exportData() {{
const blob = new Blob([JSON.stringify(metricsHistory, null, 2)], {{type: 'application/json'}});
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'live_metrics_{self._run_id}.json';
a.click();
URL.revokeObjectURL(url);
}}

function resetZoom() {{
Plotly.relayout('lossPlot', {{'xaxis.autorange': true}});
Plotly.relayout('gradientPlot', {{'xaxis.autorange': true}});
Plotly.relayout('vramPlot', {{'xaxis.autorange': true}});
Plotly.relayout('stepTimePlot', {{'xaxis.autorange': true}});
}}

function stopMonitoring() {{
if (ws) {{
ws.close();
}}
alert('Monitoring gestoppt');
}}

// Verbindung herstellen beim Laden
connectWebSocket();

// Plots initialisieren
updatePlots();
</script>
</body>
</html>
"""
return html_content


def cmd_live_monitor(args: argparse.Namespace) -> int:
"""Live Monitor Command."""
print(" Starte Live Monitor...")

if not WEBSOCKETS_AVAILABLE:
print(" Fehler: websockets nicht installiert")
print(" Installiere mit: pip install websockets")
return 1

run_id = args.run_id
port = args.port

monitor = RealtimeMonitor(run_id, websocket_port=port)

# Dashboard erstellen
monitor.create_live_dashboard()

print(f"\n Live Monitor bereit für: {run_id}")
print(f" WebSocket: ws://localhost:{port}")
print(f" Dashboard: plots/live_monitor_{run_id}.html")
print("\n Warte auf Metriken-Updates...")
print(" (Metriken können via update_metrics() gesendet werden)")

# Monitoring starten
monitor.start_monitoring()

# Warte auf Input
try:
input("\nDrücke Enter zum Beenden...")
except KeyboardInterrupt:
pass
finally:
monitor.stop_monitoring()

return 0


def create_parser() -> argparse.ArgumentParser:
"""Erstelle Argument Parser."""
parser = argparse.ArgumentParser(
prog="live-monitor",
description="Live Monitoring für Training Runs",
)
parser.add_argument(
"run_id",
type=str,
help="Run-ID zu überwachen",
)
parser.add_argument(
"--port",
type=int,
default=8765,
help="WebSocket Port (default: 8765)",
)
parser.set_defaults(func=cmd_live_monitor)
return parser


def main() -> int:
"""Hauptfunktion."""
parser = create_parser()
args = parser.parse_args()
return args.func(args)


if __name__ == "__main__":
sys.exit(main())
