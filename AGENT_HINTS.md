# AGENT HINTS - NeuroWeave Train GPT

## ⚠️ WICHTIGSTER HINT: train_gpt.py ist BEREITS der SOTA!

Die aktuelle `train_gpt.py` (1920 Zeilen) IST das Ergebnis von 2 Wochen
Wettbewerb-Optimierung. Sie erreicht **val_bpb = 1.1194**.

**NIEMALS komplett neu schreiben!** Nur gezielte Verbesserungen machen.
→ Lies `SOTA_REFERENCE.md` für die vollständige Architektur-Dokumentation.

---

## Kritische Bugs die vermieden werden müssen

### 1. FATAL: train_gpt.py komplett neu schreiben

**Problem:** Agent schreibt 844-Zeilen Basic-GPT statt auf dem 1920-Zeilen SOTA aufzubauen.

**Symptom:** val_bpb ~4.0 statt ~1.12

**Ursache:** Dem Agent fehlt das Wissen über die 15+ Features die den SOTA ausmachen.

**Fix:** IMMER auf der bestehenden train_gpt.py aufbauen. SOTA_REFERENCE.md lesen.

**Status:** ✅ Gefixt am 2026-03-28 (SOTA train_gpt.py als Basis, SOTA_REFERENCE.md erstellt)

---

### 2. compute_bpb() Endlosschleife (CRITICAL)

**Problem:** Die `compute_bpb()` Funktion läuft ENDLOS wenn der DataLoader `while True` in `__iter__` hat.

**Ort:** Betrifft nur selbst-geschriebene train_gpt.py Versionen (der SOTA hat dieses Problem nicht).

**Status:** ✅ Nicht relevant im SOTA (verwendet eval_val() mit vorberechneten val_tokens)

---

### 3. Flash Attention 3 nur auf H100 verfügbar

**Problem:** Der SOTA importiert `flash_attn_interface` — nur auf H100 installiert.

**Fix:** SDPA-Fallback ist eingebaut (seit 2026-03-28):
```python
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _USE_FA3 = True
except ImportError:
    _USE_FA3 = False
```

**Status:** ✅ Gefixt (SDPA fallback für RTX 3050)

---

### 4. Daten nicht gefunden

**Problem:** Wenn `DATA_PATH` nicht existiert, crasht der SOTA train_gpt.py.

**Fix in run.sh:**
```bash
export DATA_PATH=/home/schaf/projects/NeuroWeave/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/home/schaf/projects/NeuroWeave/data/tokenizers/fineweb_1024_bpe.model
```

**Verifikation:** 80 Train-Shards + 1 Val-Shard vorhanden.

**Status:** ✅ Automatisch in run.sh integriert (2026-03-28)

---

## Wichtige Dateien

| Datei | Zweck |
|-------|-------|
| `train_gpt.py` | **DER SOTA** — 1920 Zeilen, val_bpb=1.1194 |
| `train_gpt.py.bak` | Alte Basic-Version (844 Zeilen, val_bpb~4.0) — NUR als Warnung |
| `SOTA_REFERENCE.md` | **Vollständige Architektur-Doku des SOTA** |
| `AGENTS.md` | Regeln für alle Agenten |
| `configs/base.yaml` | SOTA-Defaults als YAML |
| `regeln.md` | Wettbewerbs-Regeln + Leaderboard |

## Workflow für Verbesserungen

```bash
# 1. SOTA_REFERENCE.md lesen
# 2. EINE gezielte Änderung in train_gpt.py machen
# 3. Smoke-Test lokal:
cd /home/schaf/projects/NeuroWeave
source .venv/bin/activate
RUN_ID=test \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=100 \
TRAIN_BATCH_TOKENS=32768 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=50 \
MAX_WALLCLOCK_SECONDS=300 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
# 4. Vergleiche val_bpb mit Baseline
# 5. Wenn besser → auf H100 testen
```

## Debug-Checkliste

Vor dem Training prüfen:
- [ ] `train_gpt.py` ist ~1920 Zeilen (nicht ~844!)
- [ ] Enthält `class Muon` Optimizer
- [ ] Enthält `qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`
- [ ] Enthält `SmearGate`, `BigramHashEmbedding`
- [ ] Enthält `mixed_quantize_int6()` + `lzma.compress()`
- [ ] `_USE_FA3` Fallback ist vorhanden
- [ ] DATA_PATH zeigt auf existierendes Verzeichnis mit .bin Dateien
- [ ] TOKENIZER_PATH zeigt auf existierende .model Datei

## Aktuelles Leaderboard (Top 5)

| Rang | val_bpb | Methode |
|------|---------|---------|
| 1 | 1.1194 | LeakyReLU² + TTT + Parallel Muon |
| 2 | 1.1228 | GPTQ-lite + EMA + warmdown3500 |
| 3 | 1.1248 | Partial RoPE + LN Scale + EMA + XSA4 |
| 4 | 1.1271 | XSA4 + EMA + Int6 MLP3x |
| 5 | 1.1307 | Efficient Partial XSA |

**Ziel: < 1.1194 (mindestens 0.005 nats besser = < 1.1144)**

---

Letzte Aktualisierung: 2026-03-28
