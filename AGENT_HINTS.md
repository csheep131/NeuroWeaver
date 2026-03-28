# AGENT HINTS - NeuroWeave Train GPT

## Kritische Bugs die vermieden werden müssen

### 1. compute_bpb() Endlosschleife (CRITICAL)

**Problem:** Die `compute_bpb()` Funktion läuft ENDLOS wenn der DataLoader `while True` in `__iter__` hat.

**Ort:** `train_gpt.py`, Funktion `compute_bpb()`

**Fix:** max_batches Parameter hinzufügen:
```python
def compute_bpb(model: nn.Module, data_loader, device: str = "cuda", max_batches: int = 100) -> float:
    # ... im Loop:
    if num_batches >= max_batches:
        break
```

**Status:** ✅ Gefixt am 2026-03-27

---

### 2. Config.from_env() liest nicht alle ENV-Variablen (CRITICAL)

**Problem:** `Config.from_env()` hat nur grundlegende ENV-Variablen gelesen (ITERATIONS, DATA_PATH, etc.) aber NICHT die SOTA-Architektur-Parameter.

**Folge:** Alle ENV-Einstellungen aus run.sh werden ignoriert!
- NUM_LAYERS=11 → wird ignoriert, Default: 9
- USE_XSA=1 → wird ignoriert, Default: False
- ATTENTION_TYPE=mha → wird ignoriert, Default: "gqa"
- etc.

**Fix:** Config.from_env() erweitern um alle Parameter:
```python
num_layers=int(os.getenv("NUM_LAYERS", "9")),
d_model=int(os.getenv("D_MODEL", "384")),
use_xsa=os.getenv("USE_XSA", "0") == "1",
# ... etc
```

**Status:** ✅ Gefixt am 2026-03-27

**WICHTIG:** Nach dem Fix erneut testen - BPB sollte von ~5.0 auf <1.5 fallen!

---

### 3. DataLoader Iteration

**Problem:** `FineWebDataset.__iter__()` hat `while True:` - endloser Iterator für Evaluation.

**Lösung:** Alle Funktionen die den DataLoader iterieren MÜSSEN begrenzt werden.

---

## Wichtige Dateien

- `train_gpt.py` - Haupttrainingsskript (kopiert nach parameter-golf Repos)
- `run.sh` - Kopiert train_gpt.py in frisches Repo, setzt ENV-Variablen

## Workflow

1. Änderungen hier in NeuroWeave machen
2. run.sh kopiert automatisch nach parameter-golf-mein-sota-*/
3. Dort dann Training starten

## Debug-Checkliste

Vor dem Training prüfen:
- [ ] Config.from_env() liest alle benötigten ENV-Variablen
- [ ] compute_bpb() hat max_batches Limit
- [ ] run.sh exportiert alle Variablen korrekt
- [ ] **DATA_PATH existiert im Zielverzeichnis** (sonst Zufallsdaten!)
- [ ] **Tokenizer existiert im Zielverzeichnis**

## Kritischer Fehler: Daten nicht gefunden

Wenn `DATA_PATH` nicht existiert, generiert `FineWebDataset.get_batch()` **Zufallsdaten** (Zeile 479)!

**Symptom:** BPB ~5.0 (statt <1.5)

**Fix in run.sh:**
```bash
# Nach dem git clone und cd in das Repo:
mkdir -p data
ln -sf /home/schaf/projects/NeuroWeave/data/datasets data/datasets
ln -sf /home/schaf/projects/NeuroWeave/data/tokenizers data/tokenizers
export DATA_PATH=/home/schaf/projects/NeuroWeave/data/datasets/fineweb10B_sp1024
```

**Status:** ✅ Automatisch in run.sh integriert (2026-03-27)

---

Letzte Aktualisierung: 2026-03-27
