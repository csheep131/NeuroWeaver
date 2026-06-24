# RunPod Setup Guide — Parameter Golf Challenge

Dieser Guide beschreibt das Setup von RunPod GPUs für das Training von Parameter Golf Challenge Modellen.

## Übersicht

OpenAI partnered mit RunPod für einfaches GPU-Setup. Challenge-Training erfordert:
- **Minimum:** 1xH100 (für Testing)
- **Maximum:** 8xH100 SXM (für finale Submission)

### Kosten (ca.)

| GPU | Preis/Stunde | Empfohlen für |
|-----|--------------|---------------|
| 1xH100 | ~$2-3 | Testing, Smoke Tests |
| 4xH100 | ~$8-12 | Intermediate Runs |
| 8xH100 SXM | ~$20-25 | Finale Submissions |

---

## Schritt 1: Account erstellen

### 1.1 RunPod Account

1. Besuche [runpod.io](https://www.runpod.io/)
2. Klicke auf "Sign Up"
3. Registriere mit Email oder Google/GitHub
4. Verifiziere deine Email

### 1.2 SSH Key erstellen (wichtig!)

Ohne SSH Key kannst du dich nicht mit der Pod verbinden:

```bash
# SSH Key erstellen (falls nicht vorhanden)
ssh-keygen -t ed25519 -C "runpod"

# Key anzeigen (für RunPod Dashboard)
cat ~/.ssh/id_ed25519.pub
```

**Im RunPod Dashboard:**
1. Gehe zu "Settings" → "SSH Keys"
2. Klicke "Add SSH Key"
3. Füge den Inhalt von `id_ed25519.pub` ein
4. Speichern

---

## Schritt 2: GPU Cloud Pod erstellen

### 2.1 Pod konfigurieren

1. **Dashboard** → "GPU Cloud Pods" → "Deploy"
2. **Template wählen:** "Parameter Golf Challenge" (offizielles Template)
- Alternativ: "PyTorch 2.1.0-py3.10" oder ähnliches
3. **GPU wählen:**
- Testing: 1xH100 PCIe (~$2-3/h)
- Training: 8xH100 SXM (~$20-25/h)
4. **GPU Count:** 1 (für Testing) oder 8 (für finale Runs)
5. **Container Disk:** 50GB (minimum), 100GB+ empfohlen
6. **SSH Access:** Aktivieren (wichtig!)
7. **Deploy Pod**

### 2.2 Pod starten

- Pod startet in 2-5 Minuten
- Status wechselt von "Initializing" → "Running"
- Notiere die **Pod IP** und **SSH Port**

---

## Schritt 3: Verbindung herstellen

### 3.1 SSH Verbindung

```bash
# Mit SSH Key verbinden
ssh -i ~/.ssh/id_ed25519 root@<POD_IP> -p <SSH_PORT>

# Beispiel
ssh -i ~/.ssh/id_ed25519 root@192.168.1.100 -p 22
```

### 3.2 Repository klonen

```bash
# In der Pod
cd /workspace
git clone https://github.com/neuro-weave/NeuroWeave.git
cd NeuroWeave
```

### 3.3 Dependencies installieren

```bash
# Python Environment
python3 -m venv .venv
source .venv/bin/activate

# Dependencies
pip install --upgrade pip
pip install torch numpy pyyaml datasets tqdm sentencepiece

# Optional: MLX für lokales Testing (nicht auf RunPod)
# pip install mlx
```

---

## Schritt 4: Dataset herunterladen

```bash
# FineWeb Dataset mit 1024-token Vocabulary
python3 data/cached_challenge_fineweb.py \
--variant sp1024 \
--train-shards 80
```

**Dauer:** ~10-30 Minuten (abhängig von Netzwerk)

**Speicher:** ~8-16 GB für 80 Shards (8B Tokens)

---

## Schritt 5: Training starten

### 5.1 Smoke Test (1xH100)

```bash
# Kurzer Test (200 Iterationen)
RUN_ID=smoke_test \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/train \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
python3 train_gpt.py
```

### 5.2 Full Training (8xH100)

```bash
# Mit torchrun für verteiltes Training
RUN_ID=baseline_v1 \
ITERATIONS=2000 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=200 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=600 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/train \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### 5.3 Training im Hintergrund

Für längere Runs (empfohlen):

```bash
# Mit tmux (Session bleibt aktiv)
tmux new -s training

# Dann Training starten
RUN_ID=baseline_v1 ... python3 train_gpt.py

# Detachen: Strg+B, dann D
# Wieder verbinden: tmux attach -t training
```

Oder mit `nohup`:

```bash
nohup python3 train_gpt.py > training.log 2>&1 &

# Logs ansehen
tail -f training.log
```

---

## Schritt 6: Ergebnisse speichern

### 6.1 Logs herunterladen

```bash
# Von RunPod zu lokal
scp -i ~/.ssh/id_ed25519 -P <SSH_PORT> \
root@<POD_IP>:/workspace/NeuroWeave/records/baseline_v1/logs/*.log \
./local_logs/
```

### 6.2 Modell-Gewichte speichern

```bash
# Modell exportieren
python3 -c "
import torch
from train_gpt import GPT, Config

config = Config.from_env()
model = GPT(config)

# State dict speichern
torch.save(model.state_dict(), 'model_weights.pt')
print(f'Gespeichert: model_weights.pt')
"

# Herunterladen
scp -i ~/.ssh/id_ed25519 -P <SSH_PORT> \
root@<POD_IP>:/workspace/NeuroWeave/model_weights.pt \
./
```

---

## Schritt 7: Pod stoppen (wichtig!)

**Achtung:** RunPod berechnet nach Zeit, nicht nach Nutzung!

```bash
# Pod im Dashboard stoppen oder terminieren
# Dashboard → GPU Cloud Pods → Pod → Stop/Terminate
```

### Kosten sparen:

1. **Pod stoppen** wenn nicht in Verwendung
2. **Snapshot erstellen** für späteres Fortsetzen
3. **Niedrigere GPU** für Testing (1xH100 statt 8xH100)

---

## Troubleshooting

### Problem: SSH Verbindung schlägt fehl

**Lösung:**
```bash
# SSH Key Berechtigungen prüfen
chmod 600 ~/.ssh/id_ed25519

# Pod IP und Port prüfen (im Dashboard)
# Firewall/Netzwerk prüfen
```

### Problem: Out of Memory (OOM)

**Lösung:**
```bash
# Batch-Größe reduzieren
TRAIN_BATCH_TOKENS=4092 python3 train_gpt.py

# Oder Gradient Accumulation verwenden
```

### Problem: Training zu langsam

**Lösung:**
- Auf 8xH100 SXM wechseln (nicht PCIe)
- `torchrun` mit `--nproc_per_node=8` verwenden
- Mixed Precision aktivieren (AMP)

### Problem: Dataset Download zu langsam

**Lösung:**
```bash
# HuggingFace Cache verwenden
export HF_DATASETS_CACHE=/workspace/hf_cache

# Oder manuell von schnellerer Quelle
```

---

## Alternative Cloud-Anbieter

Falls RunPod nicht verfügbar:

### Lambda Labs
- [lambdalabs.com](https://lambdalabs.com/)
- Ähnliche Preise wie RunPod
- 1-8xH100 verfügbar

### Vast.ai
- [vast.ai](https://vast.ai/)
- Günstiger, aber weniger zuverlässig
- Marketplace-Modell

### GCP / AWS
- Teurer (~$3-4/h pro H100)
- Enterprise-grade Infrastruktur
- Längere Setup-Zeit

---

## Compute Grant (OpenAI)

OpenAI bietet $1.000.000 in Compute Credits:

### Beantragung

1. Formular ausfüllen: [OpenAI Compute Grant](https://forms.openai.com/compute-grant)
2. Projektbeschreibung (Parameter Golf Challenge)
3. Erwarteter Compute-Bedarf
4. GitHub Account verknüpfen

### Berechtigung

- Early-career researchers
- Undergraduate students
- Olympiad medalists
- Exceptional participants

---

## Kostenkalkulator

### Beispiel: Baseline Submission

| Aktivität | Dauer | GPU | Kosten |
|-----------|-------|-----|--------|
| Setup & Testing | 2h | 1xH100 | $6 |
| Smoke Tests | 1h | 1xH100 | $3 |
| Training Run 1 | 10min | 8xH100 | $4 |
| Training Run 2 | 10min | 8xH100 | $4 |
| Training Run 3 | 10min | 8xH100 | $4 |
| **Total** | | | **~$21** |

### Tipps zum Sparen

1. **Lokal testen** (Apple Silicon MLX)
2. **Smoke Tests** auf 1xH100
3. **Nur finale Runs** auf 8xH100
4. **Pod sofort stoppen** nach Training

---

## Ressourcen

- [RunPod Docs](https://docs.runpod.io/)
- [Challenge Regeln](../../regeln.md)
- [Submission Guide](submission_guide.md)
- [train_gpt.py](../../train_gpt.py)

---

## Support

Bei Problemen:
- RunPod Discord: [discord.gg/runpod](https://discord.gg/runpod)
- OpenAI Parameter Golf: `#parameter-golf-discussions`
- NeuroWeave Issues: [GitHub](https://github.com/neuro-weave/NeuroWeave/issues)
