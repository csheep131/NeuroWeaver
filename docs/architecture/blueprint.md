Zielbild der Software

Die Software soll nicht nur ein Modell trainieren, sondern eine Ablation-Maschine sein.

Am Ende brauchst du ein System, das diese fünf Dinge zuverlässig kann:

einen Run deklarativ beschreiben
denselben Run reproduzierbar trainieren
das Modell automatisch quantisieren und packen
BPB, Schrittzeit und Artifact-Größe sauber vergleichen
Kombinationen systematisch freischalten oder verwerfen

Die öffentliche Parameter-Golf-Diskussion zeigt genau, warum das nötig ist: Manche Techniken bringen nur in bestimmten Stacks etwas, manche kippen durch Overhead, und manche Kombinationen wie TTT + XSA können sogar schlechter sein als eine der beiden Techniken allein.

Phase 1 — Experiment Core aufbauen
Ziel

Ein stabiles, kleines Framework, mit dem du Runs definieren, starten, messen und wiederholen kannst.

Das ist die wichtigste Phase. Wenn die nicht sitzt, wird alles danach chaotisch.

Ergebnis von Phase 1

Am Ende von Phase 1 kannst du:

einen Run per Config starten
Backbone und Tokenizer austauschen
Training und Eval automatisiert ausführen
Ergebnisse in einer Run-Datenbank oder JSONL speichern
Baseline und erste Ablationen vergleichen
Architektur Phase 1
1. Config-first Run-System

Jeder Run ist eine eigene Konfiguration, keine händische Codebastelei.

Beispielhafte Run-Achsen:

Modell: depth, width, recurrence, MLP-Ratio
Tokenizer: byte / bigram / trigram / fallback
Aktivierung: GELU / LeakyReLU² / gated
Training: optimizer, warmup, WD, EMA
Eval: sliding window, stride, eval frequency
Quant: int6, int5/int6 mixed, GPTQ-lite an/aus
Extras: XSA, FiLM, TTT
2. Modultrennung

Du solltest die Software in klar getrennte Module schneiden:

configs/
core/
models/
tokenizers/
train/
eval/
quant/
runs/
reports/
3. Run Registry

Jeder Run bekommt:

Run-ID
Parent-Run
Config-Hash
Git-Commit
Start-/Endzeit
Metriken
Artefaktgröße
Status

Das ist wichtig, weil bei der Challenge sehr viele kleine Änderungen gegeneinander getestet werden müssen und öffentliche PRs zeigen, dass schon wenige Millisekunden mehr pro Step den Schrittverlust spürbar erhöhen können.

Empfohlene Dateistruktur Phase 1
project/
├─ configs/
│  ├─ base.yaml
│  ├─ runs/
│  │  ├─ run001_control.yaml
│  │  ├─ run002_hash.yaml
│  │  └─ ...
├─ core/
│  ├─ config.py
│  ├─ registry.py
│  ├─ logging.py
│  ├─ seed.py
│  └─ artifacts.py
├─ models/
│  ├─ backbone.py
│  ├─ recurrent_blocks.py
│  ├─ attention.py
│  ├─ xsa.py
│  ├─ activations.py
│  └─ film.py
├─ tokenizers/
│  ├─ byte.py
│  ├─ bigram_hash.py
│  ├─ trigram_hash.py
│  └─ fallback.py
├─ train/
│  ├─ trainer.py
│  ├─ optimizer_factory.py
│  ├─ scheduler.py
│  └─ ema.py
├─ eval/
│  ├─ bpb_eval.py
│  ├─ sliding_window.py
│  └─ benchmark.py
├─ quant/
│  ├─ quantize.py
│  ├─ mixed_precision.py
│  ├─ gptq_lite.py
│  └─ pack.py
├─ reports/
│  ├─ compare_runs.py
│  └─ leaderboard.py
├─ runs/
│  └─ run.py
└─ results/
Kernfunktionen in Phase 1
A. Run starten

Ein einziger Einstiegspunkt:

python -m runs.run --config configs/runs/run001_control.yaml
B. Standardisierte Outputs

Jeder Run schreibt:

metrics.json
train_log.jsonl
eval.json
artifact_report.json
config_resolved.yaml
C. Vergleichbarkeit

Immer speichern:

val_bpb
ms_per_step
steps_completed
artifact_bytes
quantized_val_bpb
delta_vs_parent

Diese Metriken sind genau deshalb wichtig, weil öffentlich dokumentiert ist, dass der Tradeoff zwischen BPB-Gewinn und Step-Overhead oft über Erfolg oder Misserfolg entscheidet.

Technische Priorität in Phase 1
Muss sofort rein
YAML/JSON Config Loader
Seed-Management
Run Registry
Trainer-Skelett
BPB-Evaluator
Artifact-Size-Check
Vergleichsreport
Noch nicht nötig
TTT
XSA
FiLM
exotische Quantisierung
Multi-Seed-Orchestrierung

Phase 1 soll langweilig stabil sein.

Phase 2 — Research Engine und Feature-Gates
Ziel

Jetzt baust du das System so aus, dass du gezielt Features zuschalten und in Ablationen gegeneinander laufen lassen kannst.

Ergebnis von Phase 2

Am Ende von Phase 2 kannst du:

Backbone-Varianten testen
Tokenizer-Varianten testen
Quantisierung automatisiert evaluieren
XSA, FiLM, gated MLP, recurrence-Varianten als optionale Module aktivieren
Run-Vergleiche halbautomatisch priorisieren
Leitprinzip Phase 2

Nicht „mehr Features“, sondern Feature-Gates.

Ein Gate heißt:

jedes Feature ist optional
jedes Feature hat klare Abhängigkeiten
jedes Feature hat ein Kill-Kriterium

Beispiel:

xsa.enabled = true
nur erlaubt, wenn attention.type = gqa
nur für letzte N Layer
Abbruch, wenn ms_per_step > threshold

Das ist direkt aus der aktuellen Challenge-Realität abgeleitet: Die öffentliche Analyse beschreibt mehrfach, dass manche Techniken nur in bestimmten Regimen funktionieren und dass Overhead-Techniken schnell mehr kosten als sie bringen.

Architektur Phase 2
1. Backbone Factory

Ein Model-Builder, der deklarativ aus der Config baut:

recurrence depth
tied blocks
width
MLP ratio
activation
attention type
optional XSA
optional FiLM
2. Feature Adapter Layer

Neue Techniken nicht direkt in den Hauptcode streuen, sondern als Adapter.

Beispiele:

ActivationAdapter
AttentionAdapter
RecurrenceAdapter
TTTAdapter
QuantAdapter

So kannst du Features isoliert aktivieren und wieder entfernen.

3. Evaluation Rules

Du brauchst ein kleines Regelwerk:

Wenn BPB-Gewinn < Schwelle und Overhead hoch → verwerfen
Wenn Artifact > 16 MB → disqualifiziert
Wenn Quant-Gap zu groß → nicht finalistisch

Das 16-MB-Limit und die H100/10-Minuten-Grenze sind offizielle Challenge-Anforderungen.

Was in Phase 2 konkret reinkommt
A. Backbone-Flexibilität
recurrent tied-depth
loop embeddings / asymmetrische recurrence
MLP ratio variabel
partial RoPE
GQA / KV-sharing
B. Tokenizer-Lab
byte baseline
bigram hash
trigram hash
byte fallback
Tokenizer-Kostenreport
C. Quant-Lab
int6 baseline
int5/int6 mixed
GPTQ-lite
Artifact packer
quantized eval rerun
D. Ablation Reporter

Ein Tool, das automatisch so etwas ausspuckt:

beste BPB
beste BPB pro MB
beste BPB pro ms/step
beste run lineage
Runs zum Killen
Runs für 3-Seed-Finale
Dateierweiterung in Phase 2
models/
├─ factories/
│  ├─ model_factory.py
│  └─ feature_gate.py

research/
├─ ablation_engine.py
├─ kill_rules.py
├─ rank_runs.py
└─ lineage.py
Kill-Regeln in Phase 2

Die würde ich fest in Code gießen:

Kill 1: Artifact > 16,000,000 bytes
Kill 2: ms/step deutlich schlechter ohne klaren BPB-Gewinn
Kill 3: Quant-Gap untragbar
Kill 4: Feature nur in 1 Seed gut, sonst volatil
Kill 5: Kombi macht Debugging unmöglich

Diese Logik passt zu dem, was in der öffentlichen Meta gerade immer wieder auftaucht: Kombinationen brechen, Regime ändern das Verhalten von Optimierungs-Tricks, und nicht jede gute Einzelidee skaliert im ganzen Stack.

Phase 3 — Run Orchestrator und Submission Pipeline
Ziel

Aus deinem Research-Framework wird jetzt eine Produktionspipeline für echte Runs.

Ergebnis von Phase 3

Am Ende von Phase 3 kannst du:

komplette Run-Sets automatisch starten
Ergebnisse automatisch ranken
die besten Kandidaten mit mehreren Seeds nachfahren
Artefakte automatisch packen
Submission-Ordner für GitHub / Repo vorbereiten
Hauptidee Phase 3

Du brauchst jetzt keine Bastelsoftware mehr, sondern einen Orchestrator, der die Ablation-Strategie abarbeitet.

Denn die öffentlichen PRs zeigen, dass sich der Frontier derzeit schnell verschiebt und ständig neue Varianten auftauchen; dein Vorteil ist dann nicht nur ein guter Einfall, sondern dass du schnell und sauber iterieren kannst.

Architektur Phase 3
1. Sweep Runner

Der Sweep Runner erzeugt aus Templates ganze Run-Familien.

Beispiel:

Baseline
+Hash
+LeakyReLU²
+mixed quant
+XSA
+TTT
2. Promotion System

Nicht jeder Run bekommt 3 Seeds.

Pipeline:

Stage 1: 1 Seed Screening
Stage 2: Top-N weiter
Stage 3: 3 Seeds
Stage 4: Final Packing
3. Submission Builder

Automatisch erzeugen:

final config
metrics summary
artifact size report
seed mean report
README skeleton
lineage summary
Module für Phase 3
orchestrator/
├─ sweep.py
├─ promote.py
├─ rerun_topk.py
├─ finalize.py
└─ submit_bundle.py
Stage-Logik
Stage A — Screening

Viele kleine Runs, 1 Seed.

Stage B — Focus

Nur Runs, die entweder:

klar BPB gewinnen
oder Budget freischaufeln
oder ein starkes Einzelmerkmal zeigen
Stage C — Final

3 Seeds, vollständige Quantisierung, vollständiges Packing.

Stage D — Submission

Automatische Bündelung.

Was du in Phase 3 zusätzlich brauchst
A. Dashboard oder CLI-Report

Eine Übersicht wie:

Top 10 Runs nach BPB
Top 10 Runs nach BPB/ms
Top 10 Runs nach BPB/MB
Finalisten
tote Linien
B. Lineage Tracking

Du musst sehen:

Run 8 stammt von Run 6
Run 10 basiert auf Run 8 + XSA
Run 11 basiert auf Run 6 + TTT
C. Rebuild-Fähigkeit

Ein Run muss aus Config + Commit + Seed reproduzierbar sein.

Empfohlene Reihenfolge der Entwicklung
Phase 1 zuerst bauen

Nicht verhandeln. Erst das Grundgerüst.

Dann in Phase 2 genau diese Research-Features hinzufügen
Hash tokenizer
LeakyReLU²
mixed int5/int6
XSA
TTT

Die Reihenfolge passt gut zur öffentlichen Meta: Depth/Architektur-Disziplin, Quantisierungsbudget, XSA als starker Hebel, und TTT eher später und kontrolliert, weil Kombinationen fragil sein können.

Phase 3 zuletzt

Erst wenn die Einzelteile sitzen.

Minimaler MVP pro Phase
Phase 1 MVP
ein Baseline-Run startet
BPB wird berechnet
Artifact-Bytes werden gemessen
Ergebnisse werden gespeichert
Phase 2 MVP
3 Feature-Gates funktionieren:
tokenizer
activation
quant
Ablationsreport erzeugt eine sortierte Tabelle
Phase 3 MVP
Sweep aus 5 Configs
Top-2 automatisch erneut mit 3 Seeds
Submission-Bundle wird erstellt
