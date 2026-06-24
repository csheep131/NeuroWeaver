# Weg zum Wettbewerb — Parameter Golf Submission Plan

**Stand:** 2026-03-25
**Deadline:** 30. April 2026
**GitHub:** https://github.com/openai/parameter-golf

---

## Situationsanalyse

### Was passiert ist

Du hast direkt in einem eigenen Verzeichnis angefangen zu arbeiten,
OHNE vorher das Upstream-Repo zu forken/klonen. Das Ergebnis:

- Eigener train_gpt.py (voll funktionsfähig, ~746 Zeilen)
- Eigener train_gpt_mlx.py (MLX-Variante)
- Großes Tooling: orchestrator/, research/, rust_core/, configs/
- Eigene records/-Struktur (baseline_v1, neuroweave_v1)
- Checkpoints (run009_gqa etc.)
- Umfangreiche Dokumentation

### Was das Upstream-Repo erwartet

PRs dürfen NUR einen neuen Ordner unter `records/` hinzufügen:

```
records/
track_10min_16mb/ <-- Record Submissions (SOTA-Anspruch)
2026-03-XX_DeinName/
README.md <-- Pflicht: Ansatz beschreiben
submission.json <-- Pflicht: Metriken + Metadaten
train.log <-- Pflicht: Trainings-Log (3 Seeds)
train_gpt.py <-- Pflicht: Muss standalone laufen
requirements.txt <-- Optional: Extra-Dependencies
track_non_record_16mb/ <-- Non-Record (interessanter Ansatz)
...selbe Struktur...
```

### Was NICHT verloren ist

| Deine Arbeit | Wert fuer Submission |
|---------------------------------|-------------------------------|
| train_gpt.py | KERNSTÜCK — direkt nutzbar |
| Config/Architektur-Entscheidungen| In train_gpt.py eingeflossen |
| GQA, RoPE, LeakyReLU² | Alles drin im Skript |
| INT8 Quant + zlib | Alles drin im Skript |
| DDP 8xH100 Support | Alles drin im Skript |
| orchestrator/, research/ | War fuer deine Experimente |
| rust_core/ | Nicht noetig fuer Submission |
| checkpoints/ | Nicht noetig (zu gross) |
| Doku (SUBMISSION_CHECKLIST etc.)| Guter Leitfaden fuer dich |

FAZIT: Deine Kernarbeit (train_gpt.py) ist 100% wiederverwendbar.
Das ganze Tooling drumherum war fuer deine Entwicklung nuetzlich,
gehoert aber nicht in den PR.

---

## Dein Track: Record oder Non-Record?

Aktuelles Leaderboard (25.03.2026):

| Platz | Score | Ansatz |
|-------|--------|-------------------------------------------|
| #1 | 1.1194 | LeakyReLU² + TTT + Parallel Muon |
| #2 | 1.1228 | 11L EMA + GPTQ-lite + warmdown3500 |
| #3 | 1.1248 | 11L Partial RoPE + LN Scale + EMA + XSA4 |
| ... | ... | ... |
| Base | 1.2244 | Naive Baseline |

Dein submission.json sagt: val_bpb = 1.15

- Fuer Record-Track: Muesste aktuellen SOTA (1.1194) um 0.005 schlagen
=> 1.15 reicht NICHT fuer Record
- Fuer Non-Record-Track: Interessanter Ansatz reicht
=> 1.15 ist DEUTLICH besser als Baseline und qualifiziert sich

EMPFEHLUNG: Einreichen als Non-Record Submission, AUSSER du
optimierst noch weiter und schlaegst SOTA.

---

## Schritt-fuer-Schritt Plan

### Phase 1: Fork + Repo Setup (30 Minuten)

```
1. Auf GitHub: openai/parameter-golf → "Fork" klicken
→ Forkt nach github.com/csheep131/parameter-golf

2. Fork lokal klonen:
git clone https://github.com/csheep131/parameter-golf.git
cd parameter-golf

3. Upstream als Remote hinzufuegen:
git remote add upstream https://github.com/openai/parameter-golf.git
git fetch upstream

4. Branch erstellen:
git checkout -b submission/neuroweave-gqa-leakyrelu2
```

### Phase 2: Deine Arbeit uebertragen (1 Stunde)

```
5. Submission-Ordner anlegen:
mkdir -p records/track_non_record_16mb/2026-03-XX_NeuroWeave_GQA_LeakyReLU2

6. train_gpt.py kopieren:
cp /pfad/zu/deinem/train_gpt.py \
records/track_non_record_16mb/2026-03-XX_NeuroWeave_GQA_LeakyReLU2/

7. KRITISCH — train_gpt.py anpassen:
- Alle Pfade muessen relativ sein (../../data/... etc.)
- Skript muss STANDALONE aus dem records/-Ordner laufen
- Keine Abhaengigkeit auf orchestrator/, research/, rust_core/
- Keine hardcodierten Pfade
- Test:
cd records/track_non_record_16mb/2026-03-XX_NeuroWeave.../
python train_gpt.py --help # Muss ohne Fehler starten
```

### Phase 3: Pflicht-Dateien erstellen (1 Stunde)

```
8. README.md schreiben (Englisch! Upstream ist Englisch):

Inhalt:
- Submission name + author
- Approach summary (GQA, LeakyReLU², RoPE, INT8 quant)
- Key changes vs baseline
- Results table (val_bpb, artifact size, training time)
- Reproduction commands
- Seeds tested

9. submission.json erstellen:
{
"name": "Thomas Speckert",
"github_id": "csheep131",
"run_id": "neuroweave_v1",
"val_bpb": <ECHTER WERT>,
"approach": "GQA + LeakyReLU² + RoPE + INT8 Quant",
"date": "2026-03-XX",
"hardware": "8xH100",
"training_time_minutes": 10,
"artifact_bytes": <ECHTER WERT>
}

10. Train Logs beifuegen:
- Mindestens 1 vollstaendiger Log (3 Logs fuer Credibility)
- Format: step, loss, bpb, ms_per_step pro Zeile
- Logs aus deinen checkpoints/run009_gqa/ extrahieren
ODER neu trainieren auf H100
```

### Phase 4: Validierung (KRITISCH)

```
11. Sicherstellen dass Metriken ECHT sind:
- val_bpb muss von einem echten 8xH100 Run kommen
- submission.json darf KEINE geschaetzten Werte enthalten
- Wenn du noch keinen echten H100-Run hast:
→ RunPod aufsetzen (siehe Phase 5)
→ Erst trainieren, DANN einreichen

12. Artifact Size pruefen:
- Code-Bytes (train_gpt.py) + komprimierte Modell-Bytes < 16 MB
- INT8 + zlib Compression ist schon in deinem Skript

13. Standalone-Test (frischer Clone):
git clone -b submission/neuroweave-gqa-leakyrelu2 \
https://github.com/csheep131/parameter-golf.git /tmp/test-clone
cd /tmp/test-clone/records/track_non_record_16mb/2026-03-XX_.../
# Muss ohne Fehler laufen:
python train_gpt.py --help
```

### Phase 5: H100 Training (falls noch nicht geschehen)

```
14. RunPod Account erstellen (runpod.io)
- SSH Key einrichten
- Compute Grant beantragen:
https://openai.com/form/parameter-golf-compute-grant
(OpenAI sponsort $1M an Credits)

15. 1xH100 Pod starten (zum Testen, ~$2-3/h):
- Parameter Golf Template verwenden
- Fork klonen auf den Pod
- Dataset herunterladen:
python3 data/cached_challenge_fineweb.py --variant sp1024
- Smoke Test:
cd records/track_non_record_16mb/2026-03-XX_.../
RUN_ID=smoke ITERATIONS=200 python train_gpt.py

16. 8xH100 Pod starten (fuer echten Run, ~$20/h):
- 3 Runs mit verschiedenen Seeds (42, 137, 1337)
- Logs speichern
- Bester Run → submission.json aktualisieren
```

### Phase 6: PR einreichen

```
17. Alles committen:
git add records/track_non_record_16mb/2026-03-XX_NeuroWeave.../
git commit -m "Non-Record Submission: NeuroWeave GQA+LeakyReLU² val_bpb X.XX"

18. Pruefen dass NUR der records/-Ordner geaendert ist:
git diff upstream/main --stat
# Sollte NUR Dateien unter records/ zeigen!

19. Push + PR erstellen:
git push origin submission/neuroweave-gqa-leakyrelu2
# Auf GitHub: "Compare & pull request"

20. PR-Titel:
"[Non-Record] NeuroWeave: GQA + LeakyReLU² + INT8 — val_bpb X.XX"

21. PR-Body: Ausfuehrliche Beschreibung (siehe Template unten)
```

---

## PR Body Template

```markdown
## Non-Record Submission: NeuroWeave

**Author:** Thomas Speckert (@csheep131)
**Track:** Non-Record (10min/16MB)

### Approach

Systematic ablation study optimizing architecture choices within
the 16MB constraint:

- **GQA** (6Q, 3KV heads) replacing standard MHA — reduces KV cache
- **LeakyReLU²** (leakiness=0.5) activation
- **RoPE** positional encoding
- **INT8 quantization** + zlib compression for artifact size
- **9L x 384d** architecture (fits 16MB with headroom)

### Results

| Metric | Baseline | Ours | Delta |
|----------------|----------|----------|----------|
| val_bpb | 1.2244 | X.XX | -X.XX |
| artifact_bytes | ~14MB | ~XXMB | -XX% |
| training_time | <10min | <10min | — |

Seeds tested: 3 (42, 137, 1337)
BPB std over seeds: 0.0XX

### Reproduction

```bash
cd records/track_non_record_16mb/2026-03-XX_NeuroWeave_.../
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Files

- `train_gpt.py` — Complete training script
- `submission.json` — Metrics and metadata
- `train.log` — Training log (best seed)
- `README.md` — Detailed description
```

---

## Zeitplan

| Wann | Was | Dauer |
|---------------|--------------------------------------------|---------|
| Tag 1 | Fork + Branch + Dateien uebertragen | 2h |
| Tag 2-3 | RunPod Setup + Compute Grant beantragen | 1h |
| Tag 3-5 | Smoke Tests auf 1xH100 | 2h |
| Tag 5-7 | 3 volle Runs auf 8xH100, Logs sammeln | 3h |
| Tag 7 | submission.json + README mit echten Werten | 1h |
| Tag 7-8 | Standalone-Test, PR einreichen | 1h |
| Puffer | Fixes falls PR Feedback bekommt | ~3 Tage |

Gesamt: ~1 Woche aktive Arbeit, gut im Zeitrahmen bis 30. April.

---

## Haeufige Fehler vermeiden

1. KEINE Dateien ausserhalb records/ aendern
→ PR wird abgelehnt

2. KEINE geschaetzten Metriken einreichen
→ "val_bpb: 1.15" in deiner jetzigen submission.json ist
verdaechtig rund — muss ein echter gemessener Wert sein

3. train_gpt.py MUSS standalone laufen
→ Darf nicht von orchestrator/, research/ etc. abhaengen
→ Alle Imports muessen standard sein (torch, numpy, etc.)

4. README und submission.json auf ENGLISCH
→ Upstream ist komplett Englisch

5. Logs muessen echt sein
→ Kein Fake-Log, OpenAI verifiziert Top-Einreichungen

6. Kein checkpoints/ oder __pycache__/ committen
→ .gitignore des Upstream-Repos beachten

---

## Was mit dem Rest passiert

Dein ganzes lokales Tooling (orchestrator, research, configs, etc.)
ist NICHT verloren — es bleibt in deinem privaten Repo als dein
Experimentier-Framework. Nur train_gpt.py (das Endergebnis)
geht in den PR.

Denk daran wie eine Doktorarbeit: Dein Labor-Equipment bleibt
im Labor. Nur das Paper (train_gpt.py + Ergebnisse) wird
veroeffentlicht.

---

## Entscheidung: Jetzt einreichen oder weiter optimieren?

Option A: JETZT einreichen (Non-Record mit ~1.15 BPB)
+ Sicherer Slot, interessanter Ansatz
+ Fruehe Sichtbarkeit
- Weit von SOTA (1.1194) entfernt

Option B: Weiter optimieren, dann einreichen
+ Potentiell besserer Score
+ Ideen vom Leaderboard einbauen (XSA, EMA, GPTQ-lite, TTT)
- Risiko: Deadline verpasst, nichts eingereicht

EMPFEHLUNG: Option A zuerst (Non-Record PR einreichen),
dann Optional weiter optimieren und zweiten PR mit besserem
Score nachschieben. So ist die Arbeit auf jeden Fall sichtbar.
