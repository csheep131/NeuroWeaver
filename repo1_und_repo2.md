# Repo 1 und Repo 2 — Entwicklung vs. Einreichung

## Ueberblick

Dieses Projekt arbeitet mit ZWEI getrennten Repositories:

```
REPO 1 (Entwicklung)          REPO 2 (Einreichung)
~/projects/NeuroWeave/         Fork von openai/parameter-golf
│                              │
├── train_gpt.py ──kopieren──→ ├── records/
├── train_gpt_mlx.py           │   ├── track_10min_16mb/
├── orchestrator/               │   │   └── YYYY-MM-DD_NeuroWeave_.../
├── research/                   │   │       ├── train_gpt.py  ← NUR DIESE
├── rust_core/                  │   │       ├── README.md
├── configs/                    │   │       ├── submission.json
├── checkpoints/                │   │       └── train.log
├── data/                       │   └── track_non_record_16mb/
├── records/                    │       └── ...
├── plots/                      ├── data/          (upstream, nicht aendern)
├── AGENTS.md                   ├── train_gpt.py   (upstream, nicht aendern)
├── regeln.md                   └── README.md      (upstream, nicht aendern)
└── ...                        
```

---

## Repo 1 — Entwicklung (dieses Repo)

**Pfad:** ~/projects/NeuroWeave/
**Zweck:** Hier wird entwickelt, experimentiert, getestet.
**Sprache:** Deutsch OK

### Was hier lebt

- train_gpt.py — DAS Trainings-Skript (wird staendig verbessert)
- orchestrator/ — Sweep Runner, Promotion, Dashboard
- research/ — Ablation Engine, Hypothesen, Pareto-Tracker
- rust_core/ — Optionale Rust-Beschleunigung
- configs/ — YAML Run-Konfigurationen
- checkpoints/ — Gespeicherte Modell-Checkpoints
- data/ — FineWeb Dataset + Tokenizer
- records/ — Lokale Submission-Entwuerfe

### Regeln fuer Repo 1

- Alles erlaubt: neue Dateien, Experimente, Tooling
- train_gpt.py darf beliebig komplex werden
- Checkpoints, Logs, Plots — alles lokal speichern
- Git-Historie ist privat, muss nicht sauber sein

---

## Repo 2 — Einreichung (GitHub Fork)

**Pfad:** Noch anzulegen (github.com/csheep131/parameter-golf)
**Zweck:** Hier wird NUR der PR erstellt.
**Sprache:** ENGLISCH (Pflicht!)

### Was hier rein darf

AUSSCHLIESSLICH ein neuer Ordner unter records/:

```
records/track_non_record_16mb/YYYY-MM-DD_NeuroWeave_GQA_LeakyReLU2/
  ├── train_gpt.py       Pflicht. Muss standalone laufen.
  ├── README.md           Pflicht. Erklaert den Ansatz (Englisch).
  ├── submission.json     Pflicht. Echte Metriken.
  ├── train.log           Pflicht. Mindestens 1 Log, besser 3.
  └── requirements.txt    Optional. Nur wenn Extra-Pakete noetig.
```

### Regeln fuer Repo 2

- NUR den records/-Ordner aendern — NICHTS anderes
- Keine Aenderungen an train_gpt.py im Root (das ist Upstreams)
- Keine Aenderungen an data/, README.md, requirements.txt im Root
- Kein orchestrator/, research/, rust_core/ etc. committen
- Keine __pycache__/, checkpoints/, .env committen
- Alles auf Englisch

### Was NICHT rein darf

- orchestrator/ (Entwicklungs-Tooling)
- research/ (Entwicklungs-Tooling)
- rust_core/ (Entwicklungs-Tooling)
- configs/ (interne Konfigurationen)
- checkpoints/ (zu gross, nicht relevant)
- plots/ (nicht relevant)
- data/ (Upstream hat eigene)
- Alles auf Deutsch

---

## Workflow: Von Repo 1 nach Repo 2

### Einmalig: Fork anlegen

```bash
# 1. Auf GitHub: openai/parameter-golf → Fork
# 2. Fork klonen:
git clone https://github.com/csheep131/parameter-golf.git ~/projects/parameter-golf-fork
cd ~/projects/parameter-golf-fork

# 3. Upstream hinzufuegen:
git remote add upstream https://github.com/openai/parameter-golf.git

# 4. Branch erstellen:
git checkout -b submission/neuroweave-v1
```

### Bei jeder Einreichung

```bash
# 1. In Repo 1: Sicherstellen dass train_gpt.py standalone laeuft
cd ~/projects/NeuroWeave
python train_gpt.py --help  # Kein ImportError?

# 2. Submission-Ordner in Repo 2 anlegen (einmalig)
SUBMISSION_DIR=~/projects/parameter-golf-fork/records/track_non_record_16mb/2026-03-XX_NeuroWeave_GQA_LeakyReLU2
mkdir -p $SUBMISSION_DIR

# 3. train_gpt.py kopieren
cp ~/projects/NeuroWeave/train_gpt.py $SUBMISSION_DIR/

# 4. Pfade in der kopierten Datei pruefen!
#    DATA_PATH muss relativ sein: ../../data/datasets/...
#    TOKENIZER_PATH muss relativ sein: ../../data/tokenizers/...

# 5. README.md, submission.json, train.log erstellen/kopieren
#    (siehe AGENTS.md fuer Inhalt)

# 6. Standalone-Test
cd $SUBMISSION_DIR
python train_gpt.py --help  # Muss OHNE Repo-1-Code laufen

# 7. Commit + Push
cd ~/projects/parameter-golf-fork
git add records/
git diff --cached --stat  # Nur records/ Dateien? Gut.
git commit -m "Non-Record Submission: NeuroWeave GQA+LeakyReLU2 val_bpb X.XX"
git push origin submission/neuroweave-v1

# 8. Auf GitHub: Pull Request erstellen
```

### Vor dem PR nochmal pruefen

```bash
# Frischer Clone — simuliert was OpenAI sieht
git clone -b submission/neuroweave-v1 \
  https://github.com/csheep131/parameter-golf.git /tmp/test-submission
cd /tmp/test-submission

# Dataset vorhanden?
ls data/datasets/fineweb10B_sp1024/

# Submission laeuft?
cd records/track_non_record_16mb/2026-03-XX_.../
python train_gpt.py --help

# Artifact < 16MB?
python -c "
import os
code_size = os.path.getsize('train_gpt.py')
print(f'Code: {code_size:,} bytes')
print(f'Budget fuer Modell: {16_000_000 - code_size:,} bytes')
"
```

---

## Zusammenfassung

| Frage                        | Repo 1 (NeuroWeave)    | Repo 2 (Fork)         |
|------------------------------|------------------------|-----------------------|
| Wo?                          | ~/projects/NeuroWeave  | ~/projects/parameter-golf-fork |
| Zweck?                       | Entwickeln             | Einreichen            |
| Was aendern?                 | Alles                  | NUR records/          |
| Sprache?                     | Deutsch OK             | Englisch Pflicht      |
| Tooling committen?           | Ja                     | NEIN                  |
| Checkpoints committen?       | Lokal OK               | NEIN                  |
| train_gpt.py bearbeiten?     | Ja, staendig           | Nur Kopie in records/ |
