# Run-Tabelle — Ablationsplan (10 Runs)

**Stand:** 2026-03-24  
**Status:** Implementiert (Phase 1–3 abgeschlossen)  
**Verwandte Dokumente:** [runs_guide.md](runs_guide.md), [runs_development.md](runs_development.md), [roadmap_runs.md](roadmap_runs.md)

---

## Übersicht

Dieses Dokument bietet eine komprimierte Übersicht des 10-Run-Ablationsplans zur systematischen Optimierung des Modell-Setups für die Challenge (16 MB Artifact-Limit, H100 GPU). Jeder Run ändert genau **eine** Variable gegenüber dem Parent-Run.

---

## Legende

| Begriff | Beschreibung |
|---------|--------------|
| **ΔBPB** | Verbesserung der Bits Per Byte vs. Parent-Run (negativ = besser) |
| **Kill-Kriterium** | Bedingung bei der der Run sofort verworfen wird |
| **KEEP** | Feature darf in nachfolgende Runs übernommen werden |
| **BPB** | Bits Per Byte — Hauptmetrik für Modellqualität (niedriger = besser) |
| **ms/step** | Millisekunden pro Trainingsschritt — Performance-Metrik |
| **Artifact-Budget** | Maximale Größe der Modell-Artefakte (16 MB Limit) |

### Akronyme

| Abkürzung | Bedeutung |
|-----------|-----------|
| **XSA** | Cross-Sequence Attention |
| **TTT** | Test-Time Training |
| **FiLM** | Feature-wise Linear Modulation |
| **GQA** | Grouped Query Attention |
| **MLP** | Multi-Layer Perceptron |
| **Attn** | Attention-Layer |

---

## Run-Übersicht

| Run | Phase | Änderung | Erwartung | Kill-Kriterium | KEEP |
|-----|-------|----------|-----------|----------------|------|
| 1 | A | Baseline Backbone | Stabile Referenz | Instabil / zu langsam | ✅ Pflicht |
| 2 | A | Bigram + Trigram Hash | Klarer BPB-Drop | ΔBPB < 0,002 | ggf. |
| 3 | A | Byte Fallback Embedding | Stabilere BPB | Langsamer + kein Gain | ❌ meistens raus |
| 4 | B | LeakyReLU² Aktivierung | Solider Boost | ΔBPB < 0,002 | wahrscheinlich ✅ |
| 5 | B | Gated LeakyReLU² | Stärker als Run 4 | Minimal besser + komplexer | ❌ wenn nicht klar besser |
| 6 | C | Mixed INT5/INT6 Quantisierung | Mehr Platz im Budget | Quant-Gap > 0,1 BPB | ✅ wenn stabil |
| 7 | C | Größere Kapazität | Besseres BPB | Kein Gewinn trotz Größe | ✅ nur wenn besser |
| 8 | D | XSA-4 (letzte 4 Layer) | Starker Boost | Stepzeit explodiert | 🔥 Favorit |
| 9 | D | TTT minimal (ohne XSA) | BPB reduziert | Volatil / instabil | optional |
| 10 | E | 3-Seed-Finale | Stabiles Ergebnis | Große Varianz | 🏆 |

---

## Phasen im Detail

### 🧱 Phase A — Backbone + Tokenizer (Runs 1–3)

**Ziel:** Etablierung einer stabilen Baseline und Evaluation von Tokenizer-Ansätzen.

#### Run 1 — Control Baseline
- **Änderung:** Baseline Backbone ohne Features
- **Erwartung:** Stabile Referenz für alle weiteren Vergleiche
- **Kill-Kriterium:** Training instabil oder ms/step zu hoch
- **KEEP:** ✅ Pflicht — wird nie verworfen
- **Notiz:** Dient als Nullpunkt für alle ΔBPB-Messungen

#### Run 2 — Hash Tokenizer
- **Änderung:** Bigram + Trigram Hash-Tokenizer
- **Erwartung:** Klarer BPB-Drop durch kürzere effektive Sequenzen
- **Kill-Kriterium:** ΔBPB < 0,002 (unter Noise-Schwelle)
- **KEEP:** ggf. — bei minimalem Gewinn nur Bigram beibehalten
- **Notiz:** Wenn kaum Gewinn gegenüber Byte → nur Bigram in Folge-Runs

#### Run 3 — Byte Fallback
- **Änderung:** Byte Fallback Embedding zusätzlich zu Hash
- **Erwartung:** Stabilere BPB bei Hash-Kollisionen
- **Kill-Kriterium:** Langsamer + kein messbarer BPB-Gewinn
- **KEEP:** ❌ meistens raus — nur bei klarem Vorteil behalten
- **Notiz:** Sehr wahrscheinlich wird dieses Feature verworfen

---

### ⚙️ Phase B — MLP / Aktivierung (Runs 4–5)

**Ziel:** Evaluation von Aktivierungsfunktionen und MLP-Architekturen.

#### Run 4 — LeakyReLU²
- **Änderung:** LeakyReLU² statt GELU
- **Erwartung:** Solider BPB-Boost bei ähnlicher Rechenkosten
- **Kill-Kriterium:** ΔBPB < 0,002
- **KEEP:** wahrscheinlich ✅ — bei klarem Gewinn Standard für Folge-Runs
- **Notiz:** Aktivierungswahl ist backbone-abhängig

#### Run 5 — Gated LeakyReLU²
- **Änderung:** Gate zusätzlich zu LeakyReLU²
- **Erwartung:** Stärker als Run 4
- **Kill-Kriterium:** Nur minimal besser + höhere Komplexität
- **KEEP:** ❌ wenn nicht klar besser — Gate erhöht Engineering-Kosten
- **Notiz:** Faustregel: Nur behalten wenn Gewinn deutlich sichtbar

---

### 💾 Phase C — Budget / Quantisierung (Runs 6–7)

**Ziel:** Optimierung des Artifact-Budgets durch Quantisierung und Kapazitätsanpassung.

#### Run 6 — Mixed INT5/INT6
- **Änderung:** INT5 für MLP, INT6 für Attention/Embeddings
- **Erwartung:** Mehr Platz im 16 MB Budget bei akzeptablem Quant-Gap
- **Kill-Kriterium:** Quant-Gap > 0,1 BPB (`quantized_val_bpb - val_bpb`)
- **KEEP:** ✅ wenn stabil — einer der wichtigsten Runs
- **Notiz:** 16 MB Limit und H100/10-Minuten-Grenze sind Challenge-Anforderungen

#### Run 7 — Größere Kapazität
- **Änderung:** Mehr Kapazität (MLP-Ratio 3,0 → 3,5 oder minimale Tiefe)
- **Erwartung:** Besseres BPB durch mehr Parameter
- **Kill-Kriterium:** Kein BPB-Gewinn trotz größerem Modell
- **KEEP:** ✅ nur wenn besser — Budget muss genutzt werden
- **Notiz:** Nicht gleichzeitig XSA oder TTT aktivieren

---

### 🚀 Phase D — Frontier Hebel (Runs 8–9)

**Ziel:** Evaluation fortgeschrittener Features mit hohem Potenzial.

#### Run 8 — XSA-4
- **Änderung:** Cross-Sequence Attention auf letzten 4 Layern
- **Erwartung:** Starker BPB-Boost für lange Abhängigkeiten
- **Kill-Kriterium:** Stepzeit explodiert (>30% Overhead)
- **KEEP:** 🔥 Favorit — einer der stärksten realen Hebel
- **Notiz:** XSA auf allen Layern nicht automatisch besser als XSA-4

#### Run 9 — TTT (ohne XSA!)
- **Änderung:** Test-Time Training minimal (1 pass, state reset)
- **Erwartung:** BPB reduziert
- **Kill-Kriterium:** Volatil / instabil / schlechter als Run 8
- **KEEP:** optional — nur wenn klar besser als XSA
- **Notiz:** 
  - KEIN XSA gleichzeitig mit TTT
  - Nur 1 pass
  - State reset pro Eval-Fenster

---

### 🏁 Phase E — Finale (Run 10)

**Ziel:** Finales Setup mit 3 Seeds für Robustheits-Validierung.

#### Run 10 — 3-Seed-Finale
- **Änderung:** Bestes Setup aus vorherigen Runs
- **Erwartung:** Stabiles Ergebnis über 3 Seeds
- **Kill-Kriterium:** Große Varianz zwischen Seeds (σ > 0,03 BPB)
- **KEEP:** 🏆 — finales Submission-Setup
- **Notiz:** Wähle genau einen Pfad:
  - **Variante A:** Backbone + Tokenizer + Quant + XSA
  - **Variante B:** Backbone + Tokenizer + Quant + TTT
  - ❗ Nicht mischen (erst später evaluieren)

---

## Entscheidungslogik

Nach **jedem** Run werden folgende 4 Fragen gestellt:

| # | Frage | Bewertung |
|---|-------|-----------|
| 1 | BPB besser? | Hauptmetrik — niedrig ist besser |
| 2 | ms/step schlechter? | Overhead — kostet Trainingsschritte |
| 3 | Artifact-Budget besser/schlechter? | Freier Platz unter 16 MB? |
| 4 | Komplexität gerechtfertigt? | Engineering-Kosten im Verhältnis zum Gewinn? |

**Entscheidungsregel:** Wenn 2 von 4 Punkten negativ → Feature wird verworfen.

---

## Best Practices & Anti-Patterns

### ❌ Anti-Pattern 1: Features stacken

**Beschreibung:** Mehrere Änderungen gleichzeitig in einem Run.

**Konsequenz:**
- Keine Ahnung was wirkt
- Instabile Runs
- Verschwendetes Budget

**Gegenmaßnahme:** Immer genau eine Änderung pro Run.

---

### ❌ Anti-Pattern 2: +0,001 BPB als Gewinn werten

**Beschreibung:** Minimale BPB-Verbesserungen als echten Fortschritt interpretieren.

**Konsequenz:**
- Noise wird als Signal behandelt
- Falsche Entscheidungen bei Feature-Übernahme

**Gegenmaßnahme:** ΔBPB < 0,002 gilt als Noise — nur klare Sprünge behalten.

---

### ❌ Anti-Pattern 3: TTT zu früh aktivieren

**Beschreibung:** TTT auf schwachem Backbone oder ohne stabile Quantisierung.

**Konsequenz:**
- Instabiles Training
- Kein messbarer Gewinn

**Gegenmaßnahme:** TTT erst in Phase D auf starkem Backbone mit stabiler Quant.

---

### ❌ Anti-Pattern 4: Größeres Vocab = besser

**Beschreibung:** Annahme dass größeres Vokabular automatisch bessere BPB liefert.

**Konsequenz:**
- Verschwendetes Artifact-Budget
- Kein entsprechender BPB-Gewinn

**Gegenmaßnahme:** Depth schlägt Vocab fast immer — Vocab-Size nicht übertreiben.

---

## Zielkonfiguration

Basierend auf dem Ablationsplan ergibt sich folgende realistische Zielkonfiguration:

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
  vocab_size: 4096  # Nicht übertreiben — Depth gewinnt meist

# Aktivierung
  activation: leaky_relu_squared  # Kein Gate wenn nicht klar besser

# Quantisierung
quantization:
  enabled: true
  type: mixed
  mlp_dtype: int5
  attention_dtype: int6
  embedding_dtype: int6

# Frontier Features
  xsa:
    enabled: true
    layers: [8, 9, 10, 11]  # Nur letzte 4 Layer

# Optional (nur wenn klar besser als XSA)
# ttt:
#   enabled: true
#   passes: 1
#   state_reset: true
```

---

## Nächste Schritte

Nach Abschluss des Ablationsplans:

1. **Submission Bundle erstellen** — Siehe [runs_guide.md](runs_guide.md#submission-vorbereiten)
2. **Multi-Seed Validierung** — 3 Seeds für finales Setup
3. **Artifact-Check** — Sicherstellen dass < 16 MB
4. **Dokumentation** — Lineage und Metriken im Submission-README

---

## Verwandte Dokumente

- [runs_guide.md](runs_guide.md) — Ausführlicher Guide zum Starten von Runs
- [runs_development.md](runs_development.md) — Development-Guide mit Prinzipien
- [roadmap_runs.md](roadmap_runs.md) — Detaillierte Roadmap mit Challenge- und Lokal-Pfad
- [configuration.md](configuration.md) — Konfigurations-Handbuch
