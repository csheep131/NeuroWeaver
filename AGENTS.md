# AGENTS.md — Verbindliche Regeln fuer alle AI-Agenten

Dieses Dokument ist PFLICHTLEKTUERE fuer jeden Agenten der in diesem
Repository arbeitet (Hermes, Qwen, Cline, Codex, etc.).

---

## ⚠️ REGEL 0: NIEMALS train_gpt.py VON NULL SCHREIBEN! ⚠️

**Die aktuelle `train_gpt.py` IST bereits der SOTA (val_bpb = 1.1194).**

Sie enthält ~15 Features die ALLE zusammenwirken. Wenn du sie komplett 
neu schreibst, landest du bei ~4.0 BPB statt ~1.12 — das ist SCHLECHTER 
als die Naive Baseline (1.2244).

**→ Lies SOTA_REFERENCE.md BEVOR du irgendetwas änderst!**
**→ Mache nur GEZIELTE, INKREMENTELLE Verbesserungen!**
**→ Teste nach jeder Änderung ob BPB sich verbessert hat!**

### Was passiert wenn du von Null schreibst:

| Feature das fehlt           | BPB-Impact |
|----------------------------|------------|
| Muon Optimizer fehlt       | +0.5-1.0   |
| Parameter Banking fehlt    | +0.3-0.5   |
| INT6 Quantisierung fehlt   | +0.05-0.1  |
| U-Net Skips fehlen         | +0.02-0.05 |
| XSA fehlt                  | +0.02-0.03 |
| BigramHash fehlt           | +0.01-0.02 |
| SmearGate fehlt            | +0.01-0.02 |
| EMA fehlt                  | +0.01-0.02 |
| Sliding Window Eval fehlt  | +0.01-0.02 |
| Kumulativ                  | ~+3.0 BPB! |

---

## Regel 1: Nur train_gpt.py zaehlt

Die EINZIGE Datei die am Ende eingereicht wird ist `train_gpt.py`.

Alles andere in diesem Repo (orchestrator/, research/, rust_core/,
configs/, checkpoints/, plots/, etc.) ist Entwicklungs-Tooling.
Es hilft beim Experimentieren, geht aber NIEMALS in den PR.

Wenn du an train_gpt.py arbeitest, beachte:

- Die Datei MUSS standalone lauffaehig sein
- Keine Imports aus orchestrator/, research/, rust_core/ etc.
- Erlaubte Abhaengigkeiten: torch, numpy, sentencepiece, tqdm,
  und alles was im RunPod-Template vorinstalliert ist
- Artifact (Code + komprimiertes Modell) MUSS < 16.000.000 Bytes sein
- Training MUSS in < 10 Minuten auf 8xH100 durchlaufen
- **MUSS auf dem bestehenden SOTA aufbauen, nicht von Grund auf neu!**

## Regel 2: Zwei Repos, klare Trennung

Es gibt ZWEI Repos. Siehe repo1_und_repo2.md fuer Details.

- REPO 1 (hier): Entwicklung. Alles erlaubt.
- REPO 2 (Fork): Submission. NUR records/-Ordner aendern.

Kein Agent darf Code aus Repo 1 in Repo 2 committen der nicht
unter records/track_*/... liegt.

## Regel 3: Keine falschen Metriken

val_bpb, artifact_bytes und andere Metriken in submission.json
muessen von ECHTEN Runs auf H100 GPUs stammen.

Niemals geschaetzte, gerundete oder Placeholder-Werte einreichen.

## Regel 4: Wettbewerbs-Constraints

| Constraint          | Limit              |
|---------------------|--------------------|
| Artifact Size       | < 16.000.000 Bytes |
| Training Time       | < 10 min 8xH100   |
| Eval Time           | < 10 min 8xH100   |
| Metrik              | val_bpb (bits/byte)|
| SOTA schlagen um    | >= 0.005 nats      |
| Seeds fuer Beweis   | min. 3 Runs        |
| Statistik           | p < 0.01           |

## Regel 5: Was Agenten hier tun sollen

Wenn du an diesem Projekt arbeitest, ist dein Ziel:

1. **SOTA_REFERENCE.md lesen** um die aktuelle Architektur zu verstehen
2. **train_gpt.py GEZIELT verbessern** (niedrigerer val_bpb)
3. Dabei die Constraints einhalten (16MB, 10min)
4. Aenderungen testen und dokumentieren
5. Das Tooling (orchestrator etc.) nur als Hilfsmittel nutzen

### Aktueller SOTA Stack (NICHT ENTFERNEN):

- **Parallel Muon Optimizer** mit Newton-Schulz Orthogonalisierung
- **Parameter Banking** (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank)
- **U-Net Skip Connections** (Encoder-Decoder mit Skip-Weights)
- **11 Layer, 512 dim, 8 Heads (GQA: 4 KV Heads)**
- **LeakyReLU²** Activation (leaky_relu(x, 0.5).square())
- **SmearGate + BigramHash + XSA (last 4)**
- **Partial RoPE** (16 dims), **RMSNorm**, **Logit Softcapping**
- **Value Embeddings** auf Layer 9,10
- **EMA Weight Averaging** (decay=0.997)
- **Late QAT** + **INT6+LZMA Komprimierung**
- **Sliding Window Evaluation** (stride=64)
- **Optional: Test-Time Training** (Score-First Protokoll)

### Verbesserungs-Ideen (priorisiert):

1. **Bessere Quantisierung** → mehr Platz für Layer oder Features
2. **12 Layer** wenn Quant-Savings es erlauben  
3. **Verbessertes TTT** Protokoll
4. **Depth Recurrence** für Parameter-Sharing
5. **Gated Attention** mit Sigmoid-Gate pro Head
6. **Bessere LR-Schedules** (Warmdown-Tuning)
7. **Custom CUDA Kernels** für fused ops

## Regel 6: Englisch im Submission-Repo

Alles was in Repo 2 (Fork/PR) landet muss auf Englisch sein.
README.md, submission.json, Kommentare im Code — alles Englisch.

Hier in Repo 1 ist Deutsch OK.

---

## Workflow für Verbesserungen

```bash
# 1. SOTA_REFERENCE.md lesen
# 2. Gezielte Änderung in train_gpt.py machen
# 3. Lokal testen (RTX 3050, SDPA fallback)
cd /home/schaf/projects/NeuroWeave
source .venv/bin/activate

# Quick smoke test (wenige Steps):
RUN_ID=test_change \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=100 \
TRAIN_BATCH_TOKENS=32768 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=50 \
MAX_WALLCLOCK_SECONDS=300 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 4. val_bpb vergleichen mit Baseline
# 5. Wenn besser: In run.sh kopieren und auf H100 testen
```

---

Letzte Aktualisierung: 2026-03-28
