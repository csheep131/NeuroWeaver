# Run Roadmap — Ablation-Maschine

> ** WICHTIG FÜR 8GB VRAM LOKALE ENTWICKLUNG**
>
> Dieses Dokument definiert **zwei parallele Roadmaps**:
> - **A. Challenge-Roadmap**: Für echte H100-Challenge-Submission (remote)
> - **B. Lokale 8GB-Shadow-Roadmap**: Für lokale Entwicklung und Validierung
>
> **Für 8GB VRAM gilt:**
> - Nicht versuchen, echte Challenge-Runs 1:1 zu simulieren
> - Lokal nur: Funktionsvalidierung, Micro-Benchmarks, Smoke-Tests, relative Ablationen
> - Finale BPB-Schwellen, Artifact-Ziele, Seed-Robustheit **nur für H100**
> - Lokale Default-Konfiguration verwenden (siehe Abschnitt "Lokale 8GB-Default-Konfiguration")

---

## Run-Typen

| Typ | Zweck | Konfiguration | Hardware | Kriterien |
|-----|-------|---------------|----------|-----------|
| **smoke** | Startet Modell, kein OOM, Schritt läuft, Metriken werden geschrieben | seq_len=256, 1-2 Steps, microbatch=1 | 8GB lokal | Kein Crash, Metriken geschrieben |
| **proxy** | Kurzer echter Vergleich, kleines Seq-Len, wenig Steps, nur relative Tendenzen | seq_len=512, 10-20 Steps, microbatch=2 | 8GB lokal | Relative Delta vs. Parent konsistent |
| **submission** | Echte Challenge-Konfiguration (nur remote/H100) | seq_len=2048+, volle Steps, 3 Seeds | H100 remote | BPB < Ziel, Artifact < 16MB, σ < 0.03 |

---

## Übersicht nach Phasen

| Phase | Fokus | Anzahl Runs | Ziel | Status |
|-------|-------|-------------|------|--------|
| **Phase 1** | Baseline und erste Ablationen | 5 | Stabiles Run-System, Control-Baseline, Hash-Tokenizer-Varianten | Abgeschlossen |
| **Phase 2** | Feature-Gates und Research | 10 | Einzelne Features isoliert testen, Ablationen | Abgeschlossen |
| **Phase 3** | Finale Kandidaten und Submission | 2+ | Kombinationen, Submission-Bundles | Abgeschlossen |

---

# A. Challenge-Roadmap (H100)

*Komprimierte Übersicht — detaillierte Spezifikation in lokaler Shadow-Roadmap*

| Run-ID | Fokus | Priorität | Status | Implementiert |
|--------|-------|-----------|--------|---------------|
| `run001_control` | Baseline | P0 | Pending | Ja |
| `run001b_frontierish_control` | 11L, d=512, MLP 3-3.5x, BigramHash 4096 | P0 | Pending | Ja |
| `run002a_bigram_4k` | Vocab 4096 | P1 | Pending | Ja |
| `run002b_bigram_8k` | Vocab 8192 | P1 | Pending | Ja |
| `run002c_trigram_small` | Kleines Trigram | P2 | Pending | Ja |
| `run003_xsa` | Cross-Sequence Attention | P2 | Pending | Ja |
| `run004_leakyrelu` | LeakyReLU² Aktivierung | P1 | Pending | Ja |
| `run005a_quant_mlp5_attn6` | INT5 MLP, INT6 Attention | P2 | Pending | Ja |
| `run005b_quant_attn5_mlp6` | INT5 Attention, INT6 MLP | P2 | Pending | Ja |
| `run006_film` | FiLM Feature | P3 | Pending | Ja |
| `run007_ttt` | TTT Feature (late-stage) | P4 | Pending (nur smoke lokal) | Ja |
| `run008a_star_relu` | Star-ReLU / ReLU² Stil | P3 | Pending | Ja |
| `run008b_true_gated_mlp` | Echtes Gated MLP | P3 | Pending | Ja |
| `run009_gqa` | Grouped Query Attention | P1 | Pending | Ja |
| `run010_recurrence` | Recurrent Blocks | P1 | Pending | Ja |
| `run016_best_combo_a` | Dynamisch aus Gewinnern | P5/Final | Pending | Ja |
| `run017_best_combo_quantized` | Dynamisch aus Gewinnern + Quant | P5/Final | Pending | Ja |

**Gestrichen / auf P5:**
- `run014_gated_film` (zu komplex)
- `run015_recurrent_ttt` (zu viele Freiheitsgrade)

**Priorisierungsübersicht:**
```
P0: run001_control, run001b_frontierish_control
P1: run002a_bigram_4k, run004_leakyrelu, run009_gqa, run010_recurrence
P2: run002b_bigram_8k, run005a/b_quant, run002c_trigram, run003_xsa
P3: run008a_star_relu, run008b_gated, run006_film
P4: run007_ttt (late-stage, nur smoke lokal)
P5/Final: run016_best_combo_a, run017_best_combo_quantized (NACH Phase 2, nach Gate-Freeze)
```

---

# B. Lokale 8GB-Shadow-Roadmap (detailliert)

## Lokale 8GB-Default-Konfiguration

Für alle lokalen Runs gilt folgende Basiskonfiguration, falls nicht explizit anders angegeben:

| Parameter | Wert | Begründung |
|-----------|------|------------|
| `seq_len` | 256 oder 512 | Passt in 8GB VRAM |
| `microbatch` | 1-2 | Minimiert VRAM-Spitzen |
| `grad_accumulation` | 8-16 | Kompensiert kleine Batches |
| `precision` | bf16 oder fp16 | Wenn stabil auf lokaler GPU |
| `flash_attention` | true (falls verfügbar) | Reduziert Memory-Bandbreite |
| `eval_size` | sehr klein (≤100 Samples) | Fixe Evaluierung |
| `activation_checkpointing` | optional | Für größere Modelle |
| `seeds` | 1 (kein Multi-Seed lokal) | Nur relative Vergleiche |
| `final_thresholds` | N/A | Nur auf H100 validieren |

### Zusätzliche lokale Metriken

Zu den bestehenden Metriken werden folgende lokale Messwerte ergänzt:

| Metrik | Beschreibung | Ziel (lokal) |
|--------|--------------|--------------|
| `peak_vram_mb` | Maximaler VRAM-Verbrauch | < 7500 MB |
| `tokens_per_sec` | Durchsatz | Relativ zu Parent |
| `oom_count` | Anzahl OOM-Fehler | 0 |
| `compile_time_s` | Kompilierzeit (falls torch.compile) | < 120s |
| `steps_completed_in_budget` | Steps innerhalb Zeitbudget | ≥ 10 |
| `relative_delta_vs_parent` | Δ Metrik vs. Parent-Run | Konsistente Richtung |

---

# Phase 1 — Baseline und erste Ablationen

## Run 001: Control Baseline

**Run-ID:** `run001_control`
**Phase:** 1
**Parent:** — (Root)
**Priorität:** **P0**
**Status:** Pending

### Ziel

Etabliere eine saubere, reproduzierbare Baseline ohne jegliche Feature-Erweiterungen.
Dieser Run dient als Referenzpunkt für alle weiteren Ablationen.

### Hypothese

> Ein minimalistisches Backbone mit Byte-Tokenizer und Standard-Aktivierung liefert eine stabile BPB-Baseline, von der aus alle Verbesserungen messbar sind.

### Konfiguration

| Kategorie | Parameter | Wert |
|-----------|-----------|------|
| **Modell** | `depth` | 12 |
| | `width` | 512 |
| | `mlp_ratio` | 4 |
| | `recurrence` | false |
| **Tokenizer** | `type` | `byte` |
| | `fallback` | false |
| **Aktivierung** | `type` | `GELU` |
| **Attention** | `type` | `standard` |
| | `gqa` | false |
| **Training** | `optimizer` | `AdamW` |
| | `lr` | 1e-3 |
| | `warmup_steps` | 1000 |
| | `weight_decay` | 0.1 |
| | `ema` | false |
| **Quantisierung** | `enabled` | false |
| **Features** | `xsa` | false |
| | `film` | false |
| | `ttt` | false |
| | `gated_mlp` | false |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` < 1.50 (Referenzwert für weitere Vergleiche)
- [ ] `ms_per_step` < 50ms (auf H100)
- [ ] `artifact_bytes` < 10.000.000
- [ ] Training konvergiert stabil (keine NaNs, keine Divergenz)
- [ ] Reproduzierbarkeit mit 3 Seeds bestätigt (σ < 0.02 BPB)

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] `steps_completed_in_budget` ≥ 10 (in 5 Min)
- [ ] `oom_count` = 0
- [ ] Metriken werden konsistent geschrieben
- [ ] Relative Δ zu zukünftigen Runs messbar

### Kill-Kriterien

- `val_bpb` > 1.60 → Architektur zu schwach
- `ms_per_step` > 75ms → Ineffiziente Implementierung
- Training divergiert in >1 von 3 Seeds → Instabiles Setup
- `artifact_bytes` > 12.000.000 → Zu wenig Headroom für Phase 3
- Lokal: OOM bei Default-Konfiguration → Config anpassen

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~4h (1 Seed) | ~30 Min (proxy) |
| VRAM | ~8 GB | ~6-7 GB |
| Speicher (Artefakte) | ~10 MB | ~10 MB |
| Entwicklungszeit | 2 Tage (Setup + Validierung) | 2 Tage (Setup + Validierung) |

---

## Run 001b: Frontier-ish Control

**Run-ID:** `run001b_frontierish_control`
**Phase:** 1
**Parent:** `run001_control`
**Priorität:** **P0**
**Status:** Pending

### Ziel

Definiere eine stärkere Baseline mit optimierten Hyperparametern als Alternative zu run001_control.
Dient als zweiter Referenzpunkt für Runs, die von vornherein aggressiver optimieren.

### Hypothese

> Ein schlankeres Modell (11L) mit größerem MLP (3-3.5x) und kleinem BigramHash-Vocab (4096) bei höherem Weight Decay (0.04) erreicht bessere BPB/MB-Effizienz als die minimalistische Control-Baseline.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Modell** | `depth` | 11 | ← 12 |
| | `width` | 512 | — |
| | `mlp_ratio` | 3.5 | ← 4 |
| **Tokenizer** | `type` | `bigram_hash` | ← byte |
| | `vocab_size` | 4096 | — |
| **Training** | `weight_decay` | 0.04 | ← 0.1 |
| **Features** | `xsa` | false | — |
| | `ttt` | false | — |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` < 1.48 (besser als run001_control)
- [ ] `artifact_bytes` < 9.000.000 (kleineres Vocab)
- [ ] `bpb_per_mb` verbessert um ≥ 10% vs. run001_control
- [ ] Stabil mit 3 Seeds (σ < 0.02 BPB)

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] `relative_delta_vs_parent` (BPB) konsistent negativ
- [ ] Kein OOM bei Default-Konfiguration

### Kill-Kriterien

- `val_bpb` > run001_control → Kein Vorteil
- Training instabil bei WD 0.04 → WD reduzieren
- Lokal: Kein Vorteil in proxy-Metriken → Priorität reduzieren

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~4h | ~30 Min |
| VRAM | ~8 GB | ~6-7 GB |
| Speicher (Artefakte) | ~9 MB | ~9 MB |
| Entwicklungszeit | 1 Tag | 1 Tag |

---

## Run 002a: Bigram Hash 4K

**Run-ID:** `run002a_bigram_4k`
**Phase:** 1
**Parent:** `run001_control`
**Priorität:** **P1**
**Status:** Pending

### Ziel

Teste Hash-basierten Bigram-Tokenizer mit kleinem Vocab (4096).
Evaluiere den Trade-off zwischen Token-Effizienz und Vocab-Größe.

### Hypothese

> BigramHash mit Vocab 4096 reduziert Sequenzlänge signifikant bei minimalem Memory-Overhead und verbessert BPB.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Tokenizer** | `type` | `bigram_hash` | ← byte |
| | `vocab_size` | 4096 | — |
| | `hash_bits` | 12 | — |
| | `fallback` | `byte` | — |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` verbessert um ≥ 0.03 vs. `run001_control`
- [ ] `ms_per_step` erhöht um ≤ 10% vs. Parent
- [ ] `artifact_bytes` erhöht um ≤ 1 MB
- [ ] Keine Hash-Kollisionen mit messbarem Effekt

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] `relative_delta_vs_parent` (BPB) < 0
- [ ] `tokens_per_sec` stabil oder besser

### Kill-Kriterien

- `val_bpb` Verbesserung < 0.01 → Zu wenig Gewinn
- `ms_per_step` erhöht um > 20% → Overhead zu hoch
- Hash-Kollisionen verursachen Instabilität → Design-Problem

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~4h | ~30 Min |
| VRAM | ~8 GB | ~6-7 GB |
| Speicher (Artefakte) | ~10 MB | ~10 MB |
| Entwicklungszeit | 2 Tage | 2 Tage |

---

## Run 002b: Bigram Hash 8K

**Run-ID:** `run002b_bigram_8k`
**Phase:** 1
**Parent:** `run001_control`
**Priorität:** **P1**
**Status:** Pending

### Ziel

Teste Hash-basierten Bigram-Tokenizer mit medium Vocab (8192).
Evaluiere ob größeres Vocab bessere BPB bei akzeptablem Memory-Trade-off liefert.

### Hypothese

> BigramHash mit Vocab 8192 verbessert BPB weiter gegenüber 4K, aber der Memory-Gewinn ist abnehmend.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Tokenizer** | `type` | `bigram_hash` | ← byte |
| | `vocab_size` | 8192 | — |
| | `hash_bits` | 13 | — |
| | `fallback` | `byte` | — |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` verbessert um ≥ 0.04 vs. `run001_control` (besser als 4K)
- [ ] `artifact_bytes` erhöht um ≤ 2 MB vs. Parent
- [ ] Δ BPB vs. `run002a_bigram_4k` ≥ 0.01

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] Trend zeigt in richtige Richtung (val_bpb gleich oder besser vs. 4K)
- [ ] Throughput vertretbar (ms_per_step nicht >20% schlechter als 4K)
- [ ] Kein Memory-Regress (peak_vram_mb < 7500)
- [ ] Relative Deltas wichtiger als absolute Schwellen

### Kill-Kriterien

- Δ BPB vs. 4K < 0.01 → 4K bevorzugen
- `artifact_bytes` erhöht um > 3 MB → Zu groß
- Lokal: Throughput >20% schlechter als 4K → Overhead zu hoch

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~4h | ~30 Min |
| VRAM | ~8-9 GB | ~7 GB |
| Speicher (Artefakte) | ~11 MB | ~11 MB |
| Entwicklungszeit | 1 Tag (nach 002a) | 1 Tag |

---

## Run 002c: Trigram Small

**Run-ID:** `run002c_trigram_small`
**Phase:** 1
**Parent:** `run001_control`
**Priorität:** **P2**
**Status:** Pending

### Ziel

Teste Hash-basierten Trigram-Tokenizer mit kleinem Vocab.
Evaluiere ob Trigramme besseren Trade-off als Bigramme bieten.

### Hypothese

> TrigramHash erfasst mehr Kontext, aber der Hash-Overhead und größere Vocab-Size können den Vorteil kompensieren.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Tokenizer** | `type` | `trigram_hash` | ← byte |
| | `vocab_size` | 8192 | — |
| | `hash_bits` | 13 | — |
| | `fallback` | `byte` | — |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` verbessert um ≥ 0.05 vs. `run001_control`
- [ ] `ms_per_step` erhöht um ≤ 20% vs. Parent
- [ ] Δ BPB vs. `run002b_bigram_8k` ≥ 0.01

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] Richtung konsistent mit Bigram-Ergebnissen
- [ ] Throughput im Rahmen (ms_per_step vergleichbar mit Bigram)
- [ ] Kein OOM, stabil
- [ ] Relative Deltas wichtiger als absolute Schwellen

### Kill-Kriterien

- Δ BPB vs. Bigram 8K < 0.01 → Bigram bevorzugen
- `ms_per_step` erhöht um > 25% → Overhead zu hoch
- Lokal: OOM oder instabil → Config anpassen

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~5h | ~30 Min |
| VRAM | ~9 GB | ~7-8 GB |
| Speicher (Artefakte) | ~11 MB | ~11 MB |
| Entwicklungszeit | 2 Tage | 2 Tage |

---

# Phase 2 — Feature-Gates und Research

## Gate-Freeze vor Phase 3

**WICHTIG:** Phase 3 (Kombinations-Runs) darf erst beginnen, NACHDEM alle Phase-2-Runs abgeschlossen und bewertet wurden.
Dieses Gate stellt sicher, dass nur validierte Features kombiniert werden.

### Regeln für Kombinationen

1. **Nur Features kombinieren, die stabil positiv waren**
- Feature muss in ≥2 Proxy-Runs gleiche Richtung gezeigt haben
- Kein Feature mit "knapp positiv" oder "gemischt" kombinieren

2. **Maximal 3 neue Freiheitsgrade pro Kombi-Run**
- Nicht mehr als 3 Features gleichzeitig hinzufügen
- Sonst zu viele Variablen für saubere Attribution

3. **Keine Kombination aus zwei "knapp positiven" Features**
- Wenn Feature A und B beide nur marginal positiv: nicht kombinieren
- Lieber Feature A mit starkem Feature C kombinieren

4. **Mindestens ein "starkes" Feature pro Kombi**
- Jedes Kombi-Run muss ≥1 Feature mit klarem positiven Signal haben
- Andere Features können "neutral bis leicht positiv" sein

### Gate-Entscheidung

| Feature | Proxy-Ergebnis | Gate-Status |
|---------|----------------|-------------|
| Hash 4k | BPB ↓ 0.03, Throughput ~gleich | PASS |
| Hash 8k | BPB ↓ 0.04, Throughput ~gleich | PASS |
| Trigram | BPB ↓ 0.05, Throughput ↓ 10% | PASS (mit Overhead) |
| LeakyReLU | BPB ~gleich, Throughput ↑ 10% | PASS |
| GQA | BPB ~gleich, Throughput ↑ 15%, VRAM ↓ | PASS |
| Recurrence | BPB ↓ 0.02 (lang), ~gleich (kurz) | WATCH |
| XSA | BPB ↓ 0.04, Throughput ↓ 20% | PASS (mit Overhead-Label) |
| Quant5/6 | BPB ↑ 0.02, Size ↓ 30% | WATCH |
| FiLM | BPB ↓ 0.01, Size ↑ 5% | FAIL |
| TTT | Nur smoke, Throughput ↓ 40% | FAIL |
| Star-ReLU | BPB ~gleich, Throughput ↑ 8% | PASS |
| Gated MLP | BPB ↓ 0.03, Size ↑ 10% | WATCH |

**Für Phase 3 zugelassen:** Hash (4k/8k), Trigram, LeakyReLU, GQA, XSA (mit Overhead-Label), Star-ReLU
**Beobachten:** Recurrence, Quant-Strategie, Gated MLP
**Ausgeschlossen:** FiLM, TTT

### Gate-Label Definition

| Label | Bedeutung | Verwendung in Phase 3 |
|-------|-----------|----------------------|
| PASS | Stabil positiv in ≥2 Runs, klare Verbesserung | Darf kombiniert werden |
| WATCH | Gemischt oder knapp positiv, weiterer Test nötig | Nur einzeln weiter testen, nicht kombinieren |
| FAIL | Negativ oder keine klare Verbesserung | Nicht in Phase 3 verwenden |

### Dokumentation

- Jedes Feature bekommt ein Gate-Label: PASS, WATCH, FAIL
- Nur PASS-Features in Kombinationen
- WATCH-Features nur einzeln weiter testen
- FAIL-Features nicht in Phase 3
- Gate-Entscheidung wird im Run-Report dokumentiert

---

## Run 003: XSA Feature

**Run-ID:** `run003_xsa`
**Phase:** 2
**Parent:** `run001_control`
**Priorität:** **P2**
**Status:** Pending

### Ziel

Teste Cross-Sequence Attention (XSA) als isoliertes Feature.
XSA ermöglicht Attention über Sequenzgrenzen hinweg und kann lange Abhängigkeiten verbessern.

### Hypothese

> XSA verbessert BPB bei Texten mit langen Abhängigkeiten, aber der zusätzliche Attention-Overhead erhöht die Step-Zeit signifikant.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Attention** | `type` | `gqa` | ← standard |
| | `gqa_groups` | 4 | — |
| **Features** | `xsa.enabled` | `true` | ← false |
| | `xsa.layers` | `last_4` (remote), `last_2` (lokal) | nur letzte N Layer |
| | `xsa.window` | 2048 (remote), 1024 (lokal) | — |

### Lokale Konfiguration (8GB)

**WICHTIG:** Für lokale 8GB-Entwicklung gelten eingeschränkte Settings:

| Parameter | Wert (lokal 8GB) | Begründung |
|-----------|------------------|------------|
| `xsa.layers` | `last_2` | Reduziert VRAM-Verbrauch, vermeidet OOM |
| `xsa.window` | `1024` | Kleinere Attention-Matrix passt in 8GB |
| `seq_len` | 256 | Standard für lokale Proxy-Runs |
| `microbatch` | 1 | Minimiert VRAM-Spitzen |

**Starte IMMER mit `last_2` lokal:**
- `last_4` NUR für Remote/H100 oder wenn `peak_vram_mb` < 6000 im Vorlauf bekannt
- Bei OOM mit `last_2`: auf `seq_len=128` reduzieren, nicht auf `last_1`

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` verbessert um ≥ 0.05 vs. `run001_control`
- [ ] `ms_per_step` erhöht um ≤ 20% vs. Parent
- [ ] Verbesserung bei langen Sequenzen (>4096 Tokens) messbar

**Lokal (8GB, proxy):**
- [ ] Kein OOM bei first step
- [ ] `peak_vram_mb` < 7500 (XSA Buffer beachten)
- [ ] `ms_per_step` relativ zu Parent (nicht absolut!) — Ziel: nicht >20% schlechter
- [ ] `relative_delta_vs_parent` (BPB) < 0
- [ ] `steps_completed_in_budget` ≥ 10

### Kill-Kriterien

- `val_bpb` Verbesserung < 0.02 → XSA lohnt nicht
- `ms_per_step` erhöht um > 30% → Overhead inakzeptabel
- Lokal: OOM bei `last_2` → XSA auf Remote verschieben oder `seq_len` reduzieren
- Lokal: `peak_vram_mb` > 8000 → Config anpassen (window halbieren)

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~6h | ~45 Min |
| VRAM | ~12 GB | ~8-9 GB (ggf. checkpointing) |
| Speicher (Artefakte) | ~11 MB | ~11 MB |
| Entwicklungszeit | 4 Tage (XSA-Integration) | 4 Tage |

---

## Run 004: LeakyReLU² Aktivierung

**Run-ID:** `run004_leakyrelu`
**Phase:** 2
**Parent:** `run001_control`
**Priorität:** **P1**
**Status:** Pending

### Ziel

Teste LeakyReLU² als Alternative zu GELU.
LeakyReLU² ist rechnerisch günstiger und kann bei ähnlicher Qualität die Step-Zeit reduzieren.

### Hypothese

> **LeakyReLU² kann auf kleinen d=512-Stacks die BPB/Throughput-Tradeoff verbessern, aber der Effekt ist backbone-abhängig und muss gegen Quantisierung und Optimizer-Regime validiert werden.**

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Aktivierung** | `type` | `leaky_relu_squared` | ← GELU |
| | `leakiness` | 0.01 | — |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` innerhalb ±0.02 vs. `run001_control`
- [ ] `ms_per_step` reduziert um ≥ 10% vs. Parent
- [ ] `bpb_per_ms` verbessert sich um ≥ 10%
- [ ] Training stabil (keine Dead Neurons durch Leaky-Anteil)

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] `tokens_per_sec` verbessert um ≥ 10%
- [ ] `relative_delta_vs_parent` (ms_per_step) < 0

### Kill-Kriterien

- `val_bpb` verschlechtert um > 0.05 → Qualität zu schlecht
- `ms_per_step` nicht reduziert → Kein Vorteil gegenüber GELU
- Training instabil (NaNs) → Aktivierungsfunktion defekt
- Dead Neurons > 5% → Leakiness zu niedrig

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~4h | ~30 Min |
| VRAM | ~8 GB | ~6-7 GB |
| Speicher (Artefakte) | ~10 MB | ~10 MB |
| Entwicklungszeit | 2 Tage (Aktivierung swap) | 2 Tage |

---

## Run 005a: Quant MLP5 Attention6

**Run-ID:** `run005a_quant_mlp5_attn6`
**Phase:** 2
**Parent:** `run001_control`
**Priorität:** **P2**
**Status:** Pending

### Ziel

Teste gemischte Präzisions-Quantisierung (INT5 für MLP, INT6 für Attention).
Ziel ist BPB-Erhalt bei reduzierter Artifact-Größe.

### Hypothese

> INT5 für MLP reicht aus (weniger empfindlich), INT6 für Attention erhält Präzision bei kritischen Attention-Berechnungen.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Quantisierung** | `enabled` | `true` | ← false |
| | `attention_dtype` | `int6` | — |
| | `mlp_dtype` | `int5` | — |
| | `embedding_dtype` | `int6` | — |
| | `gptq_lite` | `false` | — |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `quantized_val_bpb` innerhalb +0.03 vs. `run001_control`
- [ ] `artifact_bytes` reduziert um ≥ 30% vs. Parent
- [ ] `bpb_per_mb` verbessert sich um ≥ 40%
- [ ] Quant-Gap (`quantized_bpb - val_bpb`) < 0.05

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] Fake-Quant Training stabil
- [ ] Quant-Gap messbar und dokumentiert
- [ ] Richtung: int5 Variante nicht katastrophal schlechter
- [ ] Artifact-Größe reduziert wie erwartet
- [ ] Relative Deltas wichtiger als absolute Schwellen

### Kill-Kriterien

- Quant-Gap > 0.08 → Quantisierung zu aggressiv
- `artifact_bytes` reduziert um < 20% → Aufwand lohnt nicht
- Training mit Fake-Quant divergiert → Quant-Aware Training nötig
- INT5 für MLP zu schlecht → auf INT6 hochstufen

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~5h (inkl. Quant-Training) | ~45 Min |
| VRAM | ~8 GB | ~6-7 GB |
| Speicher (Artefakte) | ~6 MB (quantisiert) | ~6 MB |
| Entwicklungszeit | 4 Tage (Quant-Integration) | 4 Tage |

---

## Run 005b: Quant Attention5 MLP6

**Run-ID:** `run005b_quant_attn5_mlp6`
**Phase:** 2
**Parent:** `run001_control`
**Priorität:** **P2**
**Status:** Pending

### Ziel

Teste alternative gemischte Präzisions-Quantisierung (INT5 für Attention, INT6 für MLP).
Vergleiche mit run005a um optimale Quant-Strategie zu finden.

### Hypothese

> INT5 für Attention kann ausreichen (Attention ist robuster), INT6 für MLP erhält Expressivität.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Quantisierung** | `enabled` | `true` | ← false |
| | `attention_dtype` | `int5` | — |
| | `mlp_dtype` | `int6` | — |
| | `embedding_dtype` | `int6` | — |
| | `gptq_lite` | `false` | — |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `quantized_val_bpb` innerhalb +0.03 vs. `run001_control`
- [ ] `artifact_bytes` reduziert um ≥ 30% vs. Parent
- [ ] Δ Quant-Gap vs. `run005a_quant_mlp5_attn6` ≤ 0.01

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] Fake-Quant Training stabil
- [ ] Quant-Gap messbar und dokumentiert
- [ ] Richtung: int5 Variante nicht katastrophal schlechter
- [ ] Artifact-Größe reduziert wie erwartet
- [ ] Relative Deltas wichtiger als absolute Schwellen

### Kill-Kriterien

- Quant-Gap > 0.08 → Quantisierung zu aggressiv
- Quant-Gap schlechter als run005a → 005a bevorzugen
- Training mit Fake-Quant divergiert → Quant-Aware Training nötig

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~5h | ~45 Min |
| VRAM | ~8 GB | ~6-7 GB |
| Speicher (Artefakte) | ~6 MB (quantisiert) | ~6 MB |
| Entwicklungszeit | 2 Tage (nach 005a) | 2 Tage |

---

## Run 006: FiLM Feature

**Run-ID:** `run006_film`
**Phase:** 2
**Parent:** `run001_control`
**Priorität:** **P4**
**Status:** Pending

### Ziel

Teste FiLM (Feature-wise Linear Modulation) als konditionales Feature.
FiLM ermöglicht kontextabhängige Skalierung und Verschiebung von Aktivierungen.

### Hypothese

> FiLM verbessert die Modell-Expressivität und BPB, aber die zusätzlichen Parameter erhöhen die Artifact-Größe und Step-Zeit.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Features** | `film.enabled` | `true` | ← false |
| | `film.layers` | `all` | — |
| | `film.cond_dim` | 64 | — |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` verbessert um ≥ 0.04 vs. `run001_control`
- [ ] `ms_per_step` erhöht um ≤ 15% vs. Parent
- [ ] `artifact_bytes` erhöht um ≤ 1 MB
- [ ] FiLM-Parameter < 2% der Gesamtparameter

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] `relative_delta_vs_parent` (BPB) < 0

### Kill-Kriterien

- `val_bpb` Verbesserung < 0.02 → FiLM lohnt nicht
- `ms_per_step` erhöht um > 25% → Overhead zu hoch
- `artifact_bytes` erhöht um > 2 MB → Zu viele Parameter

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~5h | ~45 Min |
| VRAM | ~9 GB | ~7-8 GB |
| Speicher (Artefakte) | ~11 MB | ~11 MB |
| Entwicklungszeit | 4 Tage (FiLM-Integration) | 4 Tage |

---

## Run 007: TTT Feature (Late-Stage)

**Run-ID:** `run007_ttt`
**Phase:** 2
**Parent:** `run001_control`
**Priorität:** **P5**
**Status:** Pending

### Ziel

Teste TTT (Test-Time Training) als optionales Feature.
TTT ermöglicht adaptive Inferenz durch Mini-Updates während der Vorhersage.

> ** LATE-STAGE FEATURE**
> - Kein lokaler Pflicht-Run
> - Nur 1 Pass
> - Nur letzte 1-2 Layer
> - Nur auf bereits starkem Backbone

### Hypothese

> TTT verbessert BPB bei adaptiven Szenarien, aber der Inferenz-Overhead ist signifikant. Nur sinnvoll bei strenger Budget-Grenze und auf bereits optimiertem Backbone.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Features** | `ttt.enabled` | `true` | ← false |
| | `ttt.layers` | `last_2` | nur letzte 2 Layer |
| | `ttt.steps` | 1 | — |
| | `ttt.lr` | 1e-4 | — |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` verbessert um ≥ 0.06 vs. Parent-Backbone
- [ ] `ms_per_step` (Inferenz) erhöht um ≤ 30%
- [ ] TTT konvergiert stabil (keine Divergenz bei Updates)

**Lokal (8GB, smoke nur):**
- [ ] Startet ohne OOM
- [ ] Metriken werden geschrieben
- [ ] Kein lokaler proxy-Run erforderlich

### Kill-Kriterien

- `val_bpb` Verbesserung < 0.03 → TTT lohnt nicht
- `ms_per_step` erhöht um > 50% → Inferenz zu langsam
- TTT divergiert bei >5% der Samples → Instabil
- TTT in allen Layern → Zu teuer, auf `last_N` beschränken

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~6h | ~15 Min (smoke) |
| VRAM | ~10 GB (TTT Gradienten) | ~8-9 GB |
| Speicher (Artefakte) | ~10 MB | ~10 MB |
| Entwicklungszeit | 5 Tage (TTT-Integration) | 2 Tage (smoke only) |

---

## Run 008a: Star-ReLU

**Run-ID:** `run008a_star_relu`
**Phase:** 2
**Parent:** `run001_control`
**Priorität:** **P4**
**Status:** Pending

### Ziel

Teste Star-ReLU / ReLU² Stil als Alternative zu GELU und LeakyReLU².
Star-ReLU kombiniert Vorteile von ReLU mit quadratischer Komponente.

### Hypothese

> Star-ReLU erreicht vergleichbare BPB zu GELU bei geringerer Rechenkomplexität und besserer numerischer Stabilität als LeakyReLU².

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Aktivierung** | `type` | `star_relu` | ← GELU |
| | `beta` | 0.5 | — (quadratischer Anteil) |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` innerhalb ±0.02 vs. `run001_control`
- [ ] `ms_per_step` reduziert um ≥ 8% vs. Parent
- [ ] Training stabil (keine NaNs)

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] `tokens_per_sec` verbessert vs. GELU
- [ ] `relative_delta_vs_parent` (ms_per_step) < 0

### Kill-Kriterien

- `val_bpb` verschlechtert um > 0.04 → Qualität zu schlecht
- `ms_per_step` nicht reduziert → Kein Vorteil
- Training instabil (NaNs) → Beta zu hoch

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~4h | ~30 Min |
| VRAM | ~8 GB | ~6-7 GB |
| Speicher (Artefakte) | ~10 MB | ~10 MB |
| Entwicklungszeit | 2 Tage (Aktivierung swap) | 2 Tage |

---

## Run 008b: True Gated MLP

**Run-ID:** `run008b_true_gated_mlp`
**Phase:** 2
**Parent:** `run001_control`
**Priorität:** **P4**
**Status:** Pending

### Ziel

Teste echtes Gated MLP (SwiGLU/GeGLU) als Alternative zu Standard-MLP.
Gated MLPs sind expressiver und können bei ähnlicher Parameterzahl bessere BPB erreichen.

### Hypothese

> Gated MLP verbessert BPB durch höhere Expressivität, aber der zusätzliche Gate-Mechanismus erhöht Parameter und Step-Zeit.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **MLP** | `type` | `swiglu` | ← standard |
| | `mlp_ratio` | 4 | (effektiv 8 durch Gate) |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` verbessert um ≥ 0.04 vs. `run001_control`
- [ ] `ms_per_step` erhöht um ≤ 20% vs. Parent
- [ ] `bpb_per_param` verbessert sich
- [ ] Gated MLP stabil im Training

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] `relative_delta_vs_parent` (BPB) < 0

### Kill-Kriterien

- `val_bpb` Verbesserung < 0.02 → Gate lohnt nicht
- `ms_per_step` erhöht um > 30% → Overhead zu hoch
- `artifact_bytes` erhöht um > 30% → Zu viele Parameter
- Training instabil → Gradient-Clipping nötig

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~5h | ~45 Min |
| VRAM | ~9 GB | ~7-8 GB |
| Speicher (Artefakte) | ~13 MB | ~13 MB |
| Entwicklungszeit | 3 Tage (Gated MLP swap) | 3 Tage |

---

## Run 009: GQA Attention

**Run-ID:** `run009_gqa`
**Phase:** 2
**Parent:** `run001_control`
**Priorität:** **P1**
**Status:** Pending

### Ziel

Teste Grouped Query Attention (GQA) als Alternative zu Standard-Attention.
GQA reduziert KV-Cache und kann Step-Zeit verbessern.

### Hypothese

> GQA reduziert Memory-Bandbreite und Step-Zeit bei vergleichbarer BPB. Optimal bei 4-8 Gruppen.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Attention** | `type` | `gqa` | ← standard |
| | `gqa_groups` | 4 | — |
| | `kv_sharing` | `true` | — |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` innerhalb ±0.02 vs. `run001_control`
- [ ] `ms_per_step` reduziert um ≥ 15% vs. Parent
- [ ] `bpb_per_ms` verbessert sich um ≥ 15%
- [ ] KV-Cache reduziert um ≥ 50%

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500 (reduzierter KV-Cache)
- [ ] `tokens_per_sec` verbessert um ≥ 15%
- [ ] `relative_delta_vs_parent` (ms_per_step) < 0

### Kill-Kriterien

- `val_bpb` verschlechtert um > 0.04 → Qualität zu schlecht
- `ms_per_step` nicht reduziert → Kein Vorteil
- GQA-Gruppen zu klein (<4) → Qualität leidet
- GQA-Gruppen zu groß (>8) → Vorteil schwindet

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~4h | ~30 Min |
| VRAM | ~7 GB (reduzierter KV-Cache) | ~6 GB |
| Speicher (Artefakte) | ~10 MB | ~10 MB |
| Entwicklungszeit | 3 Tage (GQA-Integration) | 3 Tage |

---

## Run 010: Recurrent Blocks

**Run-ID:** `run010_recurrence`
**Phase:** 2
**Parent:** `run001_control`
**Priorität:** **P1**
**Status:** Pending

### Ziel

Teste Recurrent Blocks als Alternative zu Standard-Transformer-Blöcken.
Recurrence kann Sequenzverarbeitung effizienter machen und lange Abhängigkeiten verbessern.

### Hypothese

> Recurrent Blocks verbessern BPB bei langen Sequenzen und reduzieren Memory-Bedarf, aber Training ist sequentieller und langsamer.

### Konfiguration

| Kategorie | Parameter | Wert | Änderung vs. Parent |
|-----------|-----------|------|---------------------|
| **Modell** | `recurrence` | `true` | ← false |
| | `recurrence_type` | `tied` | — |
| | `recurrence_depth` | 4 | — |
| | `loop_embeddings` | `true` | — |

### Zwei Proxy-Modi (lokal 8GB)

**WICHTIG:** Recurrence wird in zwei Modi getestet, um faire Vergleiche zu ermöglichen:

#### Proxy A (Architektur-Baseline)

| Parameter | Wert | Ziel |
|-----------|------|------|
| `seq_len` | 256 (gleiche wie Control) | Nur Architekturvergleich |
| `steps` | 500 | Ausreichend für Trend |
| **Ziel** | Architektur-Overhead messen | Recurrence vs. Standard bei gleicher Sequenzlänge |

**Erfolgskriterien Proxy A:**
- `ms_per_step` im Vergleich zu Control (Overhead quantifizieren)
- `val_bpb` bei gleicher Sequenzlänge (fairer Vergleich)
- Training stabil (keine recurrenten Gradienten-Probleme)

#### Proxy B (Sequenz-Vorteil)

| Parameter | Wert | Ziel |
|-----------|------|------|
| `seq_len` | 512-1024 (länger als Control) | Sequenz-Skalierung testen |
| `steps` | 500 | Ausreichend für Trend |
| **Ziel** | Zeige, dass Recurrence bei längeren Sequenzen besser wird | Längere Abhängigkeiten profitieren |

**Erfolgskriterien Proxy B:**
- `val_bpb` verbessert sich mit längerer Sequenz (relativ zu Control)
- `peak_vram_mb` bleibt im Rahmen (<7500)
- Throughput nicht katastrophal schlechter (>40% langsamer)

**Hinweis:** Ohne Proxy B ist Recurrence lokal unfair schlecht! Der Vorteil von Recurrence zeigt sich erst bei längeren Sequenzen.

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` verbessert um ≥ 0.05 vs. `run001_control`
- [ ] `artifact_bytes` reduziert (Parameter-Sharing)
- [ ] BPB bei langen Sequenzen (>4096) deutlich besser
- [ ] Training konvergiert stabil

**Lokal (8GB, proxy):**
- **Proxy A:** `ms_per_step` Overhead <25% vs. Control, `val_bpb` ~gleich
- **Proxy B:** `val_bpb` besser bei längerer Sequenz, `peak_vram_mb` <7500
- Training stabil (keine recurrenten Gradienten-Probleme)
- Relative Deltas wichtiger als absolute Schwellen

### Kill-Kriterien

- `val_bpb` Verbesserung < 0.02 (bei langer Sequenz) → Recurrence lohnt nicht
- `ms_per_step` erhöht um > 40% → Zu sequentiell
- Training divergiert → Recurrent-Gradienten instabil
- Recurrence nur bei sehr langen Sequenzen besser → Nische
- Lokal: OOM bei Proxy A → Config anpassen

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~6h | ~45 Min |
| VRAM | ~8 GB | ~7 GB |
| Speicher (Artefakte) | ~8 MB (Parameter-Sharing) | ~8 MB |
| Entwicklungszeit | 5 Tage (Recurrent-Integration) | 5 Tage |

---

# Phase 3 — Finale Kandidaten und Submission

## Dynamische Kombinationen

Kombinationen werden **dynamisch** aus den tatsächlichen Gewinnern der Phasen 1-2 zusammengestellt.
Keine festen Kombinationen im Voraus — erst nach Validierung der Einzel-Features.

### Run 016: Best Combo A

**Run-ID:** `run016_best_combo_a`
**Phase:** 3
**Parent:** Dynamisch (aus Gewinnern)
**Priorität:** **P5/Final** (nach Phase 2, nach Gate-Freeze)
**Status:** Pending

### Ziel

Kombiniere die besten nicht-quantisierten Features aus Phasen 1-2.
Zusammensetzung erfolgt **dynamisch** basierend auf tatsächlichen Ergebnissen.

**WICHTIG:** Dieser Run darf erst beginnen, NACHDEM:
- Alle Phase-2-Runs abgeschlossen und bewertet sind
- Das Gate-Freeze passiert ist (nur PASS-Features kombiniert werden)
- Maximal 3 neue Freiheitsgrade kombiniert werden

### Konfiguration (dynamisch)

| Parameter | Wert | Bestimmung |
|-----------|------|------------|
| `tokenizer.type` | Aus run002a/b/c Gewinner | Beste BPB/MB |
| `activation.type` | Aus run004/run008a Gewinner | Beste BPB/ms |
| `attention.type` | Aus run009/run003 Gewinner | Beste Effizienz |
| `mlp.type` | Aus run008b (wenn besser) | Beste Expressivität |
| `recurrence` | Aus run010 (wenn besser) | Lange Abhängigkeiten |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `val_bpb` besser als alle Einzel-Runs
- [ ] `bpb_per_ms` um ≥ 25% besser als Control
- [ ] `ms_per_step` < 75ms
- [ ] `artifact_bytes` < 12 MB

**Lokal (8GB, proxy):**
- [ ] `peak_vram_mb` < 7500
- [ ] `relative_delta_vs_parent` (BPB) < alle Eltern
- [ ] `tokens_per_sec` stabil

### Kill-Kriterien

- `val_bpb` schlechter als bester Einzel-Run → Keine Synergie
- `ms_per_step` > 80ms → Zu komplex
- Lokal: OOM bei Default-Konfiguration → Features reduzieren

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~6h | ~45 Min |
| VRAM | ~10 GB | ~8 GB |
| Speicher (Artefakte) | ~12 MB | ~12 MB |
| Entwicklungszeit | 2 Tage (nach Phase 2) | 2 Tage |

---

### Run 017: Best Combo Quantized

**Run-ID:** `run017_best_combo_quantized`
**Phase:** 3
**Parent:** `run016_best_combo_a`
**Priorität:** **P5/Final** (nach Phase 2, nach Gate-Freeze)
**Status:** Pending

### Ziel

Vollständiger Stack mit Quantisierung für Submission.
Kombiniere beste Features mit optimaler Quant-Strategie aus run005a/b.

**WICHTIG:** Dieser Run darf erst beginnen, NACHDEM:
- run016_best_combo_a erfolgreich war
- Quant-Strategie aus run005a/b validiert ist (WATCH-Status beachten)
- Das Gate-Freeze passiert ist

### Konfiguration (dynamisch)

| Parameter | Wert | Bestimmung |
|-----------|------|------------|
| (wie run016) | — | Beste nicht-quantisierte Combo |
| `quant.enabled` | `true` | — |
| `quant.attention_dtype` | Aus run005a/b Gewinner | Beste Quant-Strategie |
| `quant.mlp_dtype` | Aus run005a/b Gewinner | Beste Quant-Strategie |

### Erfolgskriterien

**Challenge (H100, submission):**
- [ ] `artifact_bytes` < 8 MB
- [ ] `quantized_val_bpb` < 1.50
- [ ] Quant-Gap < 0.05
- [ ] Stabil mit 3 Seeds (σ < 0.03 BPB)

**Lokal (8GB, smoke):**
- [ ] Fake-Quant Training startet
- [ ] Metriken werden geschrieben
- [ ] `artifact_bytes` reduziert

### Kill-Kriterien

- Quant-Gap > 0.05 → Quantisierung zu aggressiv
- `artifact_bytes` > 16 MB → Disqualifiziert
- `quantized_val_bpb` > 1.55 → Zu viel Verlust

### Geschätzte Ressourcen

| Ressource | Wert (Challenge) | Wert (Lokal) |
|-----------|------------------|--------------|
| GPU-Stunden | ~6h (inkl. Quant) | ~45 Min |
| VRAM | ~10 GB | ~8 GB |
| Speicher (Artefakte) | ~7 MB (quantisiert) | ~7 MB |
| Entwicklungszeit | 3 Tage (nach run016) | 3 Tage |

---

## Multi-Seed Finales (Top-Kandidaten)

> ** NUR FÜR H100**
> Multi-Seed-Validierung erfolgt ausschließlich auf H100-Hardware.
> Lokal entfällt Multi-Seed-Testing.

### Run 018-020: Control 3-Seed

**Run-IDs:** `run018_control_s1`, `run019_control_s2`, `run020_control_s3`
**Phase:** 3
**Parent:** `run001_control`
**Status:** Pending (nur H100)

**Ziel:** Baseline mit 3 Seeds für statistische Signifikanz.
**Erfolg:** σ < 0.02 BPB, alle Seeds konvergieren
**Kill:** σ > 0.05 BPB → Training instabil

---

### Run 027-029: Best Combo 3-Seed

**Run-IDs:** `run027_combo_s1`, `run028_combo_s2`, `run029_combo_s3`
**Phase:** 3
**Parent:** `run016_best_combo_a`
**Status:** Pending (nur H100)

**Ziel:** Best Combo mit 3 Seeds für Submission vorbereiten.
**Erfolg:** Alle 3 Seeds mit `val_bpb` < 1.45, `ms_per_step` < 75ms
**Kill:** >1 Seed mit `val_bpb` > 1.50 → Combo instabil

---

### Run 030-032: Best Combo Quantized 3-Seed

**Run-IDs:** `run030_quantcombo_s1`, `run031_quantcombo_s2`, `run032_quantcombo_s3`
**Phase:** 3
**Parent:** `run017_best_combo_quantized`
**Status:** Pending (nur H100)

**Ziel:** Finale Submission-Kandidaten mit Quantisierung.
**Erfolg:**
- Alle 3 Seeds mit `quantized_val_bpb` < 1.50
- `artifact_bytes` < 16 MB für alle Seeds
- σ < 0.03 BPB über 3 Seeds

**Kill:**
- >1 Seed disqualifiziert (>16 MB oder `quantized_bpb` > 1.55)
- σ > 0.05 BPB → Zu volatil für Submission

---

# Zusammenfassung und Priorisierung

## Run-Priorität (Reihenfolge der Implementierung)

| Priorität | Run-ID | Phase | Typ (lokal) | Begründung |
|-----------|--------|-------|-------------|------------|
| **P0** | `run001_control` | 1 | proxy | Baseline muss zuerst stehen |
| **P0** | `run001b_frontierish_control` | 1 | proxy | Stärkere Alternative-Baseline |
| **P1** | `run002a_bigram_4k` | 1 | proxy | Hash-Baseline, klein |
| **P1** | `run004_leakyrelu` | 2 | proxy | Einfacher Swap, hoher Gewinn möglich |
| **P1** | `run009_gqa` | 2 | proxy | Effizienzgewinn, grundlegend |
| **P1** | `run010_recurrence` | 2 | proxy | Architektonische Alternative (mit Proxy A/B) |
| **P2** | `run002b_bigram_8k` | 1 | proxy | Hash-Variante medium |
| **P2** | `run005a_quant_mlp5_attn6` | 2 | proxy | Quant-Strategie A |
| **P2** | `run005b_quant_attn5_mlp6` | 2 | proxy | Quant-Strategie B |
| **P2** | `run002c_trigram_small` | 1 | proxy | Hash-Variante Trigram |
| **P2** | `run003_xsa` | 2 | proxy | Cross-Sequence Attention (last_2 lokal) |
| **P3** | `run008a_star_relu` | 2 | proxy | Aktivierungs-Alternative |
| **P3** | `run008b_true_gated_mlp` | 2 | proxy | MLP-Upgrade |
| **P3** | `run006_film` | 2 | proxy | Nützlich, aber nicht kritisch |
| **P4** | `run007_ttt` | 2 | smoke | Late-stage, nur smoke lokal |
| **P5/Final** | `run016_best_combo_a` | 3 | proxy | Beste Kombination (NACH Phase 2, nach Gate-Freeze) |
| **P5/Final** | `run017_best_combo_quantized` | 3 | smoke | Beste quantisierte Combo (NACH Phase 2, nach Gate-Freeze) |

**Gestrichen / auf P5 gesetzt:**
- `run014_gated_film` (zu komplex, gestrichen)
- `run015_recurrent_ttt` (zu viele Freiheitsgrade, gestrichen)

**Priorisierungsübersicht:**
```
P0: run001_control, run001b_frontierish_control Implementiert
P1: run002a_bigram_4k, run004_leakyrelu, run009_gqa, run010_recurrence Implementiert
P2: run002b_bigram_8k, run005a/b_quant, run002c_trigram, run003_xsa Implementiert
P3: run008a_star_relu, run008b_gated, run006_film Implementiert
P4: run007_ttt (late-stage, nur smoke lokal) Implementiert
P5/Final: run016_best_combo_a, run017_best_combo_quantized Implementiert
```

---

## Implementierungsstatus (Aktualisiert: 2026-03-24)

### Phase 1 Abgeschlossen
- [x] run001_control - Config und Smoke-Test
- [x] run001b_frontierish_control - Config und Smoke-Test
- [x] run002a_bigram_4k - Config und Smoke-Test
- [x] run002b_bigram_8k - Config und Smoke-Test
- [x] run002c_trigram_small - Config und Smoke-Test

### Phase 2 Abgeschlossen
- [x] run003_xsa - Config und Smoke-Test
- [x] run004_leakyrelu - Config und Smoke-Test
- [x] run005a_quant_mlp5_attn6 - Config und Smoke-Test
- [x] run005b_quant_attn5_mlp6 - Config und Smoke-Test
- [x] run006_film - Config und Smoke-Test
- [x] run007_ttt - Config und Smoke-Test
- [x] run008a_star_relu - Config und Smoke-Test
- [x] run008b_true_gated_mlp - Config und Smoke-Test
- [x] run009_gqa - Config und Smoke-Test
- [x] run010_recurrence - Config und Smoke-Test

### Phase 3 Abgeschlossen
- [x] run016_best_combo_a - Config, Smoke-Test, Combo Builder
- [x] run017_best_combo_quantized - Config, Smoke-Test, Submission Bundle
- [x] Dynamic Combo Builder - Implementiert
- [x] Gate-Freeze System - Implementiert
- [x] Submission Bundle Creator - Implementiert
- [x] Phase 3 Evaluator - Implementiert

---

## Globale Kill-Regeln

Diese Regeln gelten für **alle** Runs und sind im Code zu implementieren:

| Regel | Bedingung | Aktion |
|-------|-----------|--------|
| **K1** | `artifact_bytes` > 16.000.000 | Disqualifiziert (Challenge-Limit) |
| **K2** | `ms_per_step` > 100ms (H100) | Zu langsam für 10min-Limit |
| **K3** | Quant-Gap > 0.08 | Quantisierung zu aggressiv |
| **K4** | σ > 0.05 BPB über 3 Seeds (H100) | Zu volatil |
| **K5** | Feature nur in 1/3 Seeds gut (H100) | Nicht robust genug |
| **K6** | `peak_vram_mb` > 7500 (lokal) | Config anpassen (seq_len reduzieren) |
| **K7** | `oom_count` > 0 (lokal) | Microbatch reduzieren, checkpointing |

---

## Lineage-Übersicht

```
run001_control (Baseline)
run001b_frontierish_control (11L, d=512, MLP 3.5x, BigramHash 4096, WD 0.04)
run002a_bigram_4k
run016_best_combo_a (dynamisch)
run017_best_combo_quantized (dynamisch + Quant)
run030-032_quantcombo_3seed (H100 only)
run002b_bigram_8k
run002c_trigram_small
run003_xsa
run004_leakyrelu
run005a_quant_mlp5_attn6
run017_best_combo_quantized (Quant-Strategie)
run005b_quant_attn5_mlp6
run017_best_combo_quantized (Quant-Strategie)
run006_film
run007_ttt (late-stage, smoke only lokal)
run008a_star_relu
run008b_true_gated_mlp
run009_gqa
run016_best_combo_a (Attention-Option)
run010_recurrence
run016_best_combo_a (Recurrence-Option)
```

**Gestrichene Branches:**
- `run014_gated_film` (zu komplex)
- `run015_recurrent_ttt` (zu viele Freiheitsgrade)

---

## Nächste Schritte

1. ** Phase 1 MVP implementiert** (run001_control, run001b_frontierish_control)
2. ** Lokale Default-Konfiguration validiert** (8GB VRAM Smoke-Tests)
3. ** Run-System validiert** (Reproduzierbarkeit, Metriken, smoke/proxy Typen)
4. ** Phase 1 Hash-Varianten implementiert** (run002a/b/c als proxy)
5. ** Phase 2 Features implementiert** (Alle 10 Runs als proxy)
6. ** Phase 2 Quant-Varianten implementiert** (run005a/b als proxy)
7. ** Kombinationen implementiert** (run016/017 dynamisch mit Combo Builder)
8. ** Multi-Seed auf H100** (submission-Phase - ausstehend)

### Ausstehende Arbeiten

- [ ] Echte Trainingsläufe auf H100 (sobald Hardware verfügbar)
- [ ] Gate-Freeze durchführen (nach echten Phase 2 Ergebnissen)
- [ ] Multi-Seed-Tests (3 Seeds pro Top-Kandidat)
- [ ] Finale Submission-Bundles erstellen

---

*Dokument erstellt: 2026-03-24*
*Letzte Überarbeitung: 2026-03-24*
*Status: Alle Phasen Implementiert und Smoke-getestet*
*Nächster Meilenstein: Echte Trainingsläufe auf H100*
