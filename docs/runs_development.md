# Runs Development Guide

Nachdem du das System zum Laufen gebracht hast (siehe README.md und SETUP.md), ist es nun
an der Zeit, die Runs zu testen, zu optimieren und den Wettbewerb zu gewinnen.

Hierzu gehst du folgenden Weg — beginne damit, Run 1 sauber zu etablieren, bevor du
irgendetwas anderes anfasst. Kein Feature, kein Trick, keine Optimierung bevor du einen
stabilen Nullpunkt hast.

---

## Voraussetzungen

Bevor du den ersten Run startest, stelle sicher:

```bash
# 1. Python-Abhängigkeiten installiert
pip install -r requirements.txt

# 2. Rust-Core kompiliert (für maximale Performance)
cd rust-core && maturin develop --release && cd ..

# 3. Basis-Imports funktionieren
python3 -c "from core import Config, RunRegistry; from tokenizers import create_tokenizer; print('OK')"

# 4. Ergebnis-Verzeichnis existiert
mkdir -p results
```

---

## Das Prinzip: Eine Änderung pro Run

Die wichtigste Regel des gesamten Ablation-Prozesses:

**Ändere immer nur genau eine Sache pro Run.**

Warum? Weil du sonst nicht weißt, was gewirkt hat. Wenn du in Run 3 gleichzeitig den
Tokenizer tauschst und die Aktivierung änderst und die Quant hochdrehst und dann BPB
besser wird — hast du nichts gelernt. Du kannst in Run 4 nichts gezielt verbessern.

Das System ist gebaut um dieses Prinzip durchzusetzen: Jeder Run hat einen `parent_run_id`,
jede Config ändert exakt einen Parameter gegenüber dem Parent.

---

## Der Ablation-Plan: 10 Runs in 5 Phasen

### 🧱 Phase A — Stabiles Fundament (Runs 1–3)

#### Run 1: Control Baseline

**Das ist dein Nullpunkt. Pflicht.**

Config: `configs/runs/run001_control.yaml`

```bash
python3 -m runs.run --config configs/runs/run001_control.yaml
```

Was dieser Run tut:
- Minimalistisches Backbone: d=512, 12 Layer, Standard-Attention, GELU
- Byte-Tokenizer (simpelste Baseline)
- AdamW, cosine schedule, lr=1e-3, warmup=1000 steps
- Keine Features (XSA, FiLM, TTT alle disabled)

Was du danach weißt:
- Dein BPB-Nullpunkt
- Deine ms/step-Baseline
- Ob das Training überhaupt stabil läuft

Kill-Kriterium: Wenn Training instabil ist oder ms/step bereits zu hoch — reduziere
die Modellgröße jetzt, nicht in Run 7.

Nach Run 1 fragst du:
1. BPB unter 2.0? Wenn nicht — das Backbone ist kaputt.
2. ms/step akzeptabel? Wenn nicht — zu groß für das Budget.
3. Loss-Kurve konvergiert sauber? Wenn nicht — lr oder warmup anpassen.

> Run 1 ist Pflicht und wird nie weggeworfen. Er ist der Anker aller weiteren Entscheidungen.

---

#### Run 2: Hash Tokenizer

**Teste ob Bigram/Trigram-Hashing deinem Backbone hilft.**

Configs: `configs/runs/run002a_bigram_4k.yaml`, `run002b_bigram_8k.yaml`, `run002c_trigram_small.yaml`

```bash
python3 -m runs.run --config configs/runs/run002a_bigram_4k.yaml
```

Was sich ändert gegenüber Run 1:
- Tokenizer: `byte` → `bigram_hash` mit vocab_size=4096
- Alles andere identisch

Was du erwartest: klaren BPB-Drop durch kürzere effektive Sequenzen.

Kill-Kriterium: ΔBPB < 0.002 — unter dieser Schwelle ist es Noise, kein echter Gewinn.

Entscheidungslogik:
- Bigram klar besser → behalten, Trigram optional testen
- Bigram minimal besser (<0.002) → weiter nur mit Byte
- Trigram kaum Mehrwert über Bigram → nur Bigram in Folge-Runs

> Öffentliche Meta zeigt: jenseits von ~10k Vocab werden die Returns klein.
> Mehr Tiefe schlägt meist breiteres Vokabular.

---

#### Run 3: Byte Fallback (optional)

**Teste additives Byte-Fallback-Embedding als Kollisionsschutz.**

Änderung: `byte_fallback: true` im Tokenizer zusätzlich zum Hash-Tokenizer.

Kill-Kriterium: Langsamer + kein BPB-Gain → sofort raus.

> Dieser Run fliegt in den meisten Fällen wieder raus. Nur behalten wenn BPB
> messbar besser wird ohne Schritt-Overhead.

---

### ⚙️ Phase B — MLP und Aktivierung (Runs 4–5)

#### Run 4: LeakyReLU²

**Teste ob deine bevorzugte Aktivierung auf diesem Backbone trägt.**

Config: `configs/runs/run004_leakyrelu.yaml`

```bash
python3 -m runs.run --config configs/runs/run004_leakyrelu.yaml
```

Was sich ändert: `activation: "gelu"` → `activation: "leaky_relu"`

Warum erst jetzt? Weil du erst wissen musst ob dein Backbone stabil ist.
Aktivierungswahl ist stark basisabhängig — auf schwachem Backbone kann SwiGLU
schlechter sein als GELU. Deshalb immer auf starker Basis testen.

Kill-Kriterium: ΔBPB < 0.002

Entscheidungslogik:
- Klar besser → LeakyReLU² wird Standard für alle Folge-Runs
- Nur minimal besser aber quantisiert fragil → lieber bei GELU bleiben

---

#### Run 5: Gated LeakyReLU² (optional)

**Teste ob ein Gate zusätzlich zu LeakyReLU² echten Mehrwert bringt.**

Änderung: `activation: "gated"` (gate on top of LeakyReLU²)

Kill-Kriterium: Nur minimal besser als Run 4 → raus. Der Gate erhöht
Engineering-Kosten bei Quantisierung und Debugging erheblich.

> Faustregel: Nur behalten wenn der Gewinn deutlich sichtbar ist,
> nicht nur in der vierten Nachkommastelle.

---

### 💾 Phase C — Quantisierungsbudget (Runs 6–7)

#### Run 6: Mixed INT5/INT6

**Einer der wichtigsten Runs.**

Config: `configs/runs/run005_mixed_quant.yaml`

```bash
python3 -m runs.run --config configs/runs/run005_mixed_quant.yaml
```

Strategie:
- MLP-Gewichte: INT5
- Attention/kritische Projektionen/Embeddings: INT6

Was du erwartest: Mehr Platz im 16MB-Artifact-Budget bei akzeptablem Quant-Gap.

Kill-Kriterium: `quantized_val_bpb - val_bpb > 0.1` — das System erkennt das automatisch.

```bash
# Artifact-Limit wird automatisch geprüft:
# AblationReporter kill_rule: artifact_bytes > 16_000_000
```

> Das 16MB-Limit und die H100/10-Minuten-Grenze sind harte Challenge-Anforderungen.
> Das System enforced beide automatisch via Kill-Rules.

---

#### Run 7: Größer trainieren, härter quantisieren

**Nutze das freigekämpfte Budget für mehr Kapazität.**

Änderung: Entweder MLP-Ratio 3.0 → 3.5 ODER minimal mehr Tiefe — aber nur eines.

Kill-Kriterium: Kein BPB-Gewinn trotz größerem Modell.

**Wichtig:** Nicht gleichzeitig XSA oder TTT aktivieren. Erst zeigen dass das
Budget besser in Kapazität investiert ist.

---

### 🚀 Phase D — Frontier-Hebel (Runs 8–9)

#### Run 8: XSA-4

**Einer der stärksten realen Hebel. Favorit.**

Config: `configs/runs/run003_xsa.yaml` (als Basis, gegen besten bisherigen Run)

Aktiviere XSA nur auf den letzten 4 Layern:

```yaml
xsa:
  enabled: true
  layers: [8, 9, 10, 11]  # letzte 4 bei 12 Layern
```

Was du beobachtest:
- BPB-Gewinn vs. bisherigem besten Run
- Zusätzlicher ms/step-Overhead
- Schritteverlust durch langsamere Steps

Kill-Kriterium: Overhead zu teuer → XSA raus.

> Öffentlich gut belegt: XSA auf allen Layern ist nicht automatisch besser als XSA-4.
> 4 Layer erscheinen nahe am Sweet Spot.

---

#### Run 9: TTT ohne XSA

**Teste Test-Time Training — aber nur auf dem besten Nicht-XSA-Run.**

**Kritisch: Nicht auf dem XSA-Run aufbauen.**

Konservatives TTT-Setup:
- 1 pass
- Kleine rank-low-rank adaptation
- State reset pro Eval-Fenster
- Keine error-guided Varianten

Kill-Kriterium: Volatil / instabil / schlechter als Run 8.

> Öffentliche Diskussion warnt ausdrücklich: TTT+XSA ist oft schlechter als XSA allein.
> Error-guided TTT-Varianten haben mehrfach negativ abgeschnitten.

---

### 🏁 Phase E — Finale (Run 10)

#### Run 10: 3-Seed-Finale

**Wähle genau einen Pfad und fahre ihn mit 3 Seeds.**

Pfad A: `Backbone + bester Tokenizer + Mixed Quant + XSA-4`

Pfad B: `Backbone + bester Tokenizer + Mixed Quant + konservatives TTT`

Nicht beide. Die Evidenz spricht gegen blindes Stapeln.

```bash
# Multi-Seed via Orchestrator
python3 -c "
from orchestrator import create_multi_seed_orchestrator
orch = create_multi_seed_orchestrator(
    base_config_path='configs/runs/run_finale.yaml',
    num_seeds=3,
    seeds=[42, 123, 456],
)
runs = orch.prepare_runs()
for r in runs:
    print(f'Starte: {r.run_id} (seed={r.seed})')
"
```

---

## Nach jedem Run: Die 4 Fragen

Nach **jedem** Run stellst du exakt diese vier Fragen:

```
1. BPB besser?          (Hauptmetrik — niedrig ist besser)
2. ms/step schlechter?  (Overhead — kostet Schritte)
3. Artifact-Budget?     (freier Platz unter 16MB?)
4. Komplexität ok?      (Engineering-Kosten gerechtfertigt?)
```

Wenn 2 von 4 Punkten negativ → Feature fliegt raus.

Das Ablation-System berechnet das automatisch:

```bash
# Kill-Rules automatisch anwenden
python3 -c "
from core.registry import RunRegistry
from research import AblationReporter

registry = RunRegistry()
reporter = AblationReporter(registry)
kills = reporter.apply_kills()
if kills:
    print('Killed:', kills)
else:
    print('Alle Runs bestehen Kill-Rules')

# Aktuellen Report anzeigen
report = reporter.generate_report()
print(report.print_summary())
"
```

---

## Runs vergleichen und Leaderboard

```bash
# Alle Runs auflisten
python3 -m orchestrator.dashboard list

# Einzelnen Run anzeigen
python3 -m orchestrator.dashboard show run001_control

# Leaderboard
python3 -m orchestrator.dashboard leaderboard --category bpb --top-k 10

# Runs vergleichen
python3 -m orchestrator.dashboard compare run001_control run002a_bigram_4k run004_leakyrelu
```

Oder direkt in Python:

```python
from reports import RunComparator, LeaderboardGenerator

# Vergleich
comp = RunComparator("results")
comparison = comp.compare_runs(["run001_control", "run002a_bigram_4k"])
print(comp.print_summary(comparison))

# Leaderboard
gen = LeaderboardGenerator("results")
lb = gen.generate_by_bpb(top_k=10)
print(lb.print_table())
```

---

## Submission vorbereiten

Wenn Run 10 abgeschlossen und stabil ist:

```python
from orchestrator import SubmissionBuilder

builder = SubmissionBuilder()
bundle = builder.create_bundle(
    run_ids=["run010_finale_seed001", "run010_finale_seed002", "run010_finale_seed003"],
    output_dir="submission/",
    include_configs=True,
)
print(f"Bundle: {bundle.output_dir}")
print(f"Best BPB: {bundle.metrics_summary['best_bpb']}")
print(f"Mean BPB: {bundle.metrics_summary['mean_bpb']:.4f}")
print(f"Artifact within limit: {bundle.artifact_report['within_limit']}")
```

Das Bundle enthält automatisch:
- `final_config.yaml` — die verwendete Konfiguration
- `metrics_summary.json` — BPB, ms/step, Seed-Statistiken
- `artifact_report.json` — Größen-Report (muss unter 16MB bleiben)
- `lineage_summary.json` — welcher Run von welchem abstammt
- `README.md` — Skeleton für die Einreichung

---

## Typische Fehler die Zeit kosten

**Fehler 1: Alles gleichzeitig ändern**
Führt zu: kein Verständnis was wirkt, unstabile Runs, verschwendetes Budget.
Lösung: Eine Änderung pro Run. Immer.

**Fehler 2: +0.001 BPB als Gewinn werten**
Unter diesem Budget gilt: ΔBPB < 0.002 ist oft Noise.
Nur klare Sprünge behalten.

**Fehler 3: TTT zu früh aktivieren**
TTT funktioniert nur auf starkem Backbone mit stabiler Quantisierung.
Erst Run 9 — nicht früher.

**Fehler 4: XSA + TTT kombinieren**
Die öffentliche Evidenz ist eindeutig: diese Kombination ist oft schlechter
als XSA allein. Run 10 wählt einen Pfad — nicht beide.

**Fehler 5: Größeres Vocab = besser**
Falsch für diese Challenge. Depth schlägt Vocab fast immer.

---

## Realistische Zielkonfiguration

Wenn alles nach Plan läuft, sieht das Gewinner-Setup so aus:

```yaml
# Backbone
model:
  d_model: 512
  recurrence:
    enabled: true
    depth: 2
    tied: true

# Tokenizer
tokenizer:
  type: bigram_hash
  vocab_size: 4096  # nicht übertreiben

# Aktivierung
  activation: leaky_relu  # kein Gate wenn nicht klar besser

# Quant
quant:
  type: int5_int6_mixed  # MLP=int5, Attn/Embed=int6

# Frontier
  xsa:
    enabled: true
    layers: [8, 9, 10, 11]  # nur letzte 4
```

---

## Übersicht: Alle 10 Runs auf einen Blick

| Run | Phase | Änderung | Erwartung | Kill-Kriterium | KEEP |
|-----|-------|----------|-----------|----------------|------|
| 1 | A | Baseline Backbone | Stabiler Nullpunkt | Instabil / zu langsam | ✅ Pflicht |
| 2 | A | Bigram+Trigram Hash | Klarer BPB-Drop | ΔBPB < 0.002 | ggf. |
| 3 | A | Byte Fallback | Stabilere BPB | Langsamer + kein Gain | ❌ meistens raus |
| 4 | B | LeakyReLU² | Solider Boost | ΔBPB < 0.002 | wahrscheinlich ✅ |
| 5 | B | Gated LeakyReLU² | Stärker als Run 4 | Minimal besser + komplexer | ❌ wenn nicht klar besser |
| 6 | C | Mixed INT5/INT6 | Mehr Platz | Quant-Gap > 0.1 BPB | ✅ wenn stabil |
| 7 | C | Größer trainieren | Besseres BPB | Kein Gewinn trotz Größe | ✅ nur wenn besser |
| 8 | D | XSA letzte 4 Layer | Starker Boost | Stepzeit explodiert | 🔥 Favorit |
| 9 | D | TTT minimal (kein XSA) | BPB runter | Volatil / instabil | optional |
| 10 | E | 3-Seed-Finale | Stabil | Große Varianz | 🏆 |

---

## Weiterführend

- `development/blueprint.md` — Architektur-Entscheidungen und Begründungen
- `development/roadmap.md` — Detaillierter Run-Plan mit Entscheidungslogik
- `development/run_tabelle.md` — Kompakte Übersichtstabelle
- `development/PHASE_2_REPORT.md` — Implementierter Stand Phase 2
- `configs/runs/` — Alle vorbereiteten Run-Configs
