# Phase 4: Autonome Selbstverbesserung - Auto-Approve System

**Letztes Update:** 2026-03-24
**Status:** ✅ IMPLEMENTIERT (Woche 1-8 abgeschlossen)
**Ziel:** Systematische Verbesserung der Such- und Ausführungsfähigkeiten bis zum optimalen Performance-Plateau

---

## Vision

Ein System, das nicht nur Runs ausführt, sondern seinen eigenen Suchprozess kontinuierlich verbessert – basierend auf historischen Erfolgen, erkannten Mustern und adaptiven Entscheidungen.

**Grundprinzipien:**
1. **Human-on-the-loop statt full autonomy** – Definierte Freigabegrenzen, Auto-Betrieb nur innerhalb harter Guardrails
2. **Search & Execution self-improvement (A+B), nicht Code self-improvement (C)** – Keine autonome Codegenerierung, aber Code-Verbesserungsvorschläge sind erwünscht
3. **Inkrementeller Ansatz** – Von einfachen Ranking-Modellen zu komplexeren Systemen
4. **Operational statt akademisch** – Praktische Tools, keine überkomplizierten Wissensgraphen

---

## Die 3-Stufen-Autonomie-Pyramide

### Stufe 1: Phase 4A — Adaptive Search Engine
**Ziel:** Das System wird besser darin, welche Runs als Nächstes sinnvoll sind.

#### Kernmodule:

1. **Meta-Feature Extractor**
   - Extrahiert pro Run strukturierte Meta-Daten:
     ```
     - Aktivierte Features (Bitmask oder Feature-Vektor)
     - Parent-Lineage (Vererbungshistorie)
     - Budgetklasse (low/medium/high basierend auf Ressourcen)
     - Sequence Length (local vs remote Kontext)
     - Quant-Status (unquantized / int6 / int5 / mixed / gptq_lite)
     - Step-Time (ms/step, normalisiert)
     - Delta vs Parent (BPB-Veränderung)
     - Stabilität über Seeds (Varianz, min/max)
     - Erfolgsquote in ähnlichen Kontexten
     ```

2. **Surrogate Scorer**
   - **Algorithmus:** Random Forest / Gradient Boosting Regressor
   - **Input:** Meta-Features
   - **Output:** Vorhergesagter Erfolg (BPB-Gewinn, Effizienz-Score)
   - **Training:** Auf historischen Runs mit bekannten Outcomes
   - **Tuning:** Bayesian Optimization für Hyperparameter
   - **Nicht:** Komplexes Reinforcement Learning (zu früh, zu teuer)

3. **Hypothesis Generator**
   - **Co-occurrence Analysis:** "Wenn Feature X gut mit Y lief, probiere auch Z"
   - **Contextual Bandits:** Exploration vs Exploitation basierend auf Kontext
   - **Pattern Mining:** Erfolgreiche Feature-Sequenzen identifizieren
   - **Diversity Preservation:** Sicherstellen, dass Suchraum nicht zu eng wird

4. **Pareto Tracker**
   - **Multi-Objective:** BPB vs Efficiency vs Model Size vs Training Speed
   - **Frontier Monitoring:** Aktueller Pareto-Frontier in Echtzeit
   - **Gap Detection:** Identifiziert Lücken im Suchraum
   - **Progress Measurement:** Fortschritt über Zeit quantifizieren

5. **Adaptive Kill Thresholds**
   - **Kontext-sensitive Grenzen:** Strenger bei knappem Budget, explorativer bei viel Budget
   - **Dynamische Anpassung:** Basierend auf Erfolgsrate und Fortschritt
   - **Feature-spezifisch:** Unterschiedliche Thresholds für verschiedene Feature-Klassen

#### Konkreter Output (Dashboard):
```
Top 5 vielversprechende nächste Runs:
1. Combo: [X, Y, Z] - Predicted ΔBPB: -0.02 (Confidence: 85%)
2. Combo: [A, B]    - Predicted Efficiency Gain: 15% (Confidence: 78%)
3. Single: [F]      - Exploration in new context (Confidence: 65%)

Top 3 riskante Kombis (high variance, high potential):
1. [X, Y, W] - Instability score: high, but breakthrough potential

Features mit sinkendem Grenznutzen:
- Feature G: Diminishing returns after 5 successful combos
- Feature H: Only effective in "high-budget" context

Kandidaten für Freeze/Prune:
- Feature J: No wins in last 10 attempts across contexts
```

---

### Stufe 2: Phase 4B — Self-Diagnosis & Recovery
**Ziel:** Das System erkennt und reagiert auf Probleme, bevor sie Ressourcen verschwenden.

#### Kernmodule:

1. **Anomaly Detector**
   - **Statistische Tests:** Shapiro-Wilk auf Seed-Varianz, Grubbs' Test für Ausreißer
   - **Instabilitäts-Metriken:** Coefficient of Variation > 20% → Warnung
   - **OOM-Regression:** Memory usage spike detection
   - **Noisy Feature Identification:** Features mit hoher Outcome-Varianz

2. **Failure Classifier**
   - **Fehlerkategorien:**
     - OOM (Out of Memory)
     - NaN-Gradients
     - Training-Divergence (loss explosion)
     - Quant-Explosion (post-quantization degradation)
     - Performance-Regression
   - **ML-basiert:** Entscheidungsbaum auf Meta-Features + Error-Signature
   - **Root Cause Analysis:** "Feature X + Y führt in Kontext Z zu OOM"

3. **Rollback Manager**
   - **Automatisches Recovery:** Bei kritischen Fehlern zurück zu letzter stabiler Konfiguration
   - **Inkrementelles Rollback:** Behält erfolgreiche Teile bei, verwirft nur problematische
   - **Learning from Rollbacks:** Dokumentiert, welche Änderungen welche Probleme verursachen

4. **Run Quarantine**
   - **Automatische Blockierung:** Features/Kombis, die in 3+ Lineages Probleme machen
   - **Context-aware Block:** "Local Block" – nur in bestimmten Kontexten verboten
   - **Time-limited:** Quarantäne läuft nach N erfolgreichen Runs anderer Features aus
   - **Human Override:** Möglichkeit zur manuellen Freigabe mit Begründung

5. **Drift Monitor**
   - **Performance-Drift:** BPB/Effizienz ändert sich ohne erkennbaren Grund
   - **Umwelt-Änderungen:** Dataset-Updates, Hardware-Änderungen, Dependency-Updates
   - **Concept Drift:** Erfolgreiche Patterns werden plötzlich weniger erfolgreich
   - **Alerting:** Frühwarnung bei signifikantem Drift

#### Praktische Regeln (Beispiele):
```
RULE: Feature-Instability
IF: Feature A zeigt Varianz > 30% über 3+ Seeds in 3+ Lineages
THEN: Quarantine für 5 Runs, nur in isolierten Tests erlauben

RULE: OOM-Prevention  
IF: Kombination B+C führt in 2 verschiedenen Runs zu OOM
THEN: Local Block für B+C in ähnlichen Kontexten

RULE: Context-Specific Success
IF: Feature C gewinnt nur in Budgetklasse "high" (keine Wins in "medium"/"low")
THEN: Automatische Einschränkung auf "high"-Budget Kontexte

RULE: Quant-Gap Explosion
IF: Quant-Gap (post-quant BPB delta) > 0.1 für Kombination D
THEN: Rule-Update-Vorschlag an Human + Auto-Block für weitere Quant-Versuche
```

---

### Stufe 3: Phase 4C — Guided Autonomy
**Ziel:** Das System darf innerhalb klarer Grenzen selbst Entscheidungen treffen.

#### Erlaubte Autonomie (innerhalb Guardrails):

✅ **Automatisch erlaubt:**
1. **Neue Run-Kombinationen vorschlagen & testen**
   - Aber nur im "Smoke-Test"-Modus (max 1000 Steps, kleines Budget)
   - Nur wenn Confidence Score > 70%

2. **Lokale Promotion automatisieren**
   - Candidate → Promoted basierend auf klar definierten Thresholds:
     ```
     - ΔBPB < -0.01 (stat. signifikant über 3 Seeds)
     - Efficiency Gain > 10%  
     - Model Size < Parent Size oder Size Increase < 10%
     - Quant-Gap < 0.05 (falls quantisiert)
     ```
   - **Promoted → Submitted NUR mit menschlicher Freigabe**

3. **Exploration Rate dynamisch anpassen**
   - Bei Stagnation (kein Fortschritt in 20 Runs): Exploration Rate +20%
   - Bei schnellem Fortschritt (>0.02 BPB Gewinn in 5 Runs): Exploitation Rate +20%
   - **Cap:** Min 10% Exploration, Max 50% Exploration

4. **Pareto-Updates automatisch generieren**
   - Regelmäßige Pareto-Frontier-Snapshots (alle 10 Runs)
   - Visualisierung von Fortschritt über Zeit
   - Automatische Reports bei signifikantem Frontier-Expansion

#### ❌ **Verboten ohne menschliche Freigabe:**
1. **Neue Codepfade aktivieren** – Nur nach Code-Review
2. **Zentrale Reward-Formel ändern** – Kern-Metriken sind tabu
3. **Submission bauen** – Finale Entscheidung bleibt beim Human
4. **Neue teure Remote-Runs auslösen** – Budget > X benötigt Approval
5. **Kill-Regeln fundamental umschreiben** – Grundlegende Filter-Logik darf nicht autonom geändert werden
6. **Neue Feature-Klassen einführen** – Neue Architektur-Patterns benötigen Design-Review

---

## Implementierungs-Roadmap

### Woche 1-2: Foundation & Meta-Features
1. **Meta-Feature Schema** – Datenmodell für Run-Metadata
2. **Historical Run Enrichment** – Bestehende Runs mit Meta-Features anreichern
3. **Co-occurrence Statistics** – Einfache Feature-Interaktionsanalyse
4. **Basic Dashboard** – Visualisierung der Meta-Features

### Woche 3-4: Phase 4A Core
1. **Surrogate Scorer** – Random Forest Implementierung
2. **Hypothesis Generator** – Contextual Bandits für Run-Vorschläge
3. **Pareto Tracker** – Multi-Objective Frontier Monitoring
4. **Adaptive Kill Thresholds** – Kontext-sensitive Grenzen

### Woche 5-6: Phase 4B Core  
1. **Anomaly Detector** – Statistische Tests für Instabilität
2. **Failure Classifier** – ML-basierte Fehlerkategorisierung
3. **Rollback Manager** – Automatisches Recovery-System
4. **Run Quarantine** – Self-protection gegen bekannte Probleme

### Woche 7-8: Phase 4C & Integration
1. **Guardrail System** – Definition der Autonomie-Grenzen
2. **Autonomy Level Configuration** – Stufenweise Freischaltung
3. **Human-on-the-loop Interface** – Dashboard mit Approval-Workflow
4. **Alerting & Notification** – Wann muss Human eingreifen?

### Woche 9-10: Evaluation & Refinement
1. **A/B-Testing** – Autonome vs. manuelle Run-Auswahl vergleichen
2. **Success Metrics** – Definieren, wie "Selbstverbesserung" gemessen wird
3. **Iterative Refinement** – Basierend auf Performance Feedback-Schleife
4. **Documentation & Knowledge Transfer** – System erklärt seine eigenen Entscheidungen

---

## Erfolgsmetriken & Evaluation

### Quantitative Metriken:
| Metrik | Ziel | Messung |
|--------|------|---------|
| **Search Efficiency** | 30% weniger Runs für gleichen BPB-Gewinn | Runs benötigt für ΔBPB = -0.05 |
| **Failure Rate Reduction** | 50% weniger OOMs/NaN/Divergence | Fehler/Run Rate über Zeit |
| **Pareto Frontier Expansion** | 20% mehr Pareto-optimale Punkte | Frontier Area Growth |
| **Human Time Saved** | 70% weniger manuelle Run-Auswahl | Stunden/Woche für Run-Planning |
| **Average Confidence Score** | >75% für erfolgreiche Runs | Accuracy der Vorhersagen |

### Qualitative Verbesserungen:
1. **Explainable Decisions** – System kann begründen, warum es etwas vorschlägt/blockiert
2. **Early Problem Detection** – Probleme werden erkannt, bevor sie katastrophal werden
3. **Reproducible Success Patterns** – Erfolgreiche Kombinationen werden systematisch wiederverwendet
4. **Adaptive Learning** – System passt sich an veränderte Bedingungen an (Hardware, Datasets)
5. **Human-AI Collaboration** – Mensch und System ergänzen sich sinnvoll

---

## Risikomanagement & Guardrails

### Technische Guardrails:
1. **Budget Caps** – Max X Runs/Tag, Max Y GPU-hours/Woche
2. **Exploration Limits** – Max 50% der Runs dürfen explorativ sein
3. **Rollback Safety Net** – Immer letzte stabile Konfiguration als Fallback
4. **Quarantine Escalation** – Bei wiederholten Problemen → Human Review
5. **Confidence Thresholds** – Nur Vorschläge mit >60% Confidence automatisch ausführen

### Prozess-Guardrails:
1. **Weekly Human Review** – Alle autonomen Entscheidungen der Woche werden reviewed
2. **Approval Workflow** – Bestimmte Aktionen benötigen explizite Freigabe
3. **Transparency Log** – Jede autonome Entscheidung wird protokolliert + begründet
4. **Override Mechanism** – Human kann jederzeit autonome Entscheidungen überschreiben
5. **Learning from Overrides** – Wenn Human häufig überschreibt, passt System seine Thresholds an

---

## Nächste Schritte

### ✅ Abgeschlossen (Woche 1-8)

#### Woche 1-2: Foundation & Meta-Features ✅
1. [x] Meta-Feature Schema finalisieren
2. [x] Historische Runs analysieren und bereichern
3. [x] Einfache Co-occurrence Statistics implementieren
4. [x] Basic Dashboard für Meta-Features bauen

**Implementiert:**
- `core/meta_features.py` (RunMetaFeatures, MetaFeatureExtractor)
- `scripts/enrich_historical_runs.py` (Enrichment-Skript)
- `orchestrator/meta_dashboard.py` (Dashboard CLI)
- `tests/test_meta_features.py` (30 Tests)

#### Woche 3-4: Phase 4A Core ✅
1. [x] Surrogate Scorer mit Random Forest implementieren
2. [x] Hypothesis Generator mit Contextual Bandits
3. [x] Pareto Tracker Dashboard
4. [x] Erste A/B-Tests: Manuelle vs. vorgeschlagene Runs

**Implementiert:**
- `research/surrogate_scorer.py` (Random Forest, Gradient Boosting)
- `research/hypothesis_generator.py` (Exploitation, Exploration, Pattern-based)
- `research/pareto_tracker.py` (Multi-Objective Frontier)
- `research/adaptive_kill_thresholds.py` (Kontext-sensitive Grenzen)
- `tests/test_*.py` (46 Tests)

#### Woche 5-6: Phase 4B Core ✅
1. [x] Vollständige Phase 4A Implementierung
2. [x] Phase 4B Self-Diagnosis Module
3. [x] Phase 4C Guardrail System
4. [x] Umfassende Evaluation und Feinabstimmung

**Implementiert:**
- `research/anomaly_detector.py` (Shapiro-Wilk, Grubbs' Test)
- `research/failure_classifier.py` (ML-basierte Fehlerkategorisierung)
- `orchestrator/rollback_manager.py` (Automatisches Recovery)
- `orchestrator/run_quarantine.py` (Self-protection)
- `research/drift_monitor.py` (CUSUM, ADWIN)
- `tests/test_*.py` (94 Tests)

#### Woche 7-8: Phase 4C & Integration ✅
1. [x] Guardrail System implementieren
2. [x] Autonomy Orchestrator implementieren
3. [x] Approval Interface implementieren
4. [x] Alerting & Notification System
5. [x] Override Learner implementieren
6. [x] Phase 4 Orchestrator als Entry-Point

**Implementiert:**
- `orchestrator/guardrails.py` (Autonomie-Grenzen)
- `orchestrator/autonomy_orchestrator.py` (Zentrale Steuerung)
- `orchestrator/approval_interface.py` (Human-Freigaben)
- `core/alerting.py` (Alert-System)
- `research/override_learner.py` (Learning from Overrides)
- `orchestrator/phase4_orchestrator.py` (Entry-Point)
- `tests/test_*.py` (80 Tests)

### 📊 Gesamt-Status

| Komponente | Status | Tests | Zeilen |
|------------|--------|-------|--------|
| **Meta-Features** | ✅ Fertig | 30 | ~450 |
| **Phase 4A Core** | ✅ Fertig | 46 | ~1,883 |
| **Phase 4B Self-Diagnosis** | ✅ Fertig | 94 | ~2,100 |
| **Phase 4C Guardrails** | ✅ Fertig | 80 | ~2,200 |
| **Gesamt** | **✅ 100%** | **250** | **~6,633** |

---

## Anhang: Meta-Feature Schema (Vorschlag)

```yaml
run_meta_features:
  # Feature-Vektor (One-hot oder Bitmask)
  features_active: List[str]  # ["gqa", "film", "leaky_relu"]
  
  # Lineage & History
  parent_run_id: str
  lineage_depth: int  # Wie viele Generationen von Parent
  siblings_count: int  # Wie viele Runs mit gleichem Parent
  
  # Context
  budget_class: "low" | "medium" | "high"  # basierend auf d_model, num_layers
  sequence_length: "local" | "remote"  # short vs long context
  quantization_type: "none" | "int6" | "int5" | "mixed" | "gptq_lite"
  
  # Performance Characteristics
  step_time_ms: float  # Normalisiert pro Parameter
  memory_usage_mb: float
  training_stability: float  # loss curve smoothness
  
  # Outcomes (für Training)
  delta_bpb_vs_parent: float
  efficiency_gain_percent: float  # (parent_step_time / current_step_time - 1) * 100
  model_size_change_percent: float
  quant_gap: float  # post-quantization degradation
  
  # Statistical Properties
  seed_variance: float  # Varianz über 3 Seeds
  confidence_interval_width: float  # Breite des 95% CI für BPB
  
  # Temporal Features
  days_since_first_feature_introduction: int
  runs_since_feature_last_successful: int
  
  # Interaction Features
  co_occurrence_with: Dict[str, int]  # Count mit anderen Features
  previous_success_rate_in_similar_context: float
```

---

## Fazit

Dieser Plan verwandelt NeuroWeave von einem "dummen" Run-Executor in ein lernendes System, das seinen eigenen Suchprozess kontinuierlich optimiert. Der Fokus liegt auf **praktischer, inkrementeller Autonomie** mit klaren Guardrails – nicht auf akademischer KI-Forschung.

Das System lernt aus seinen eigenen Erfolgen und Fehlern, wird effizienter im Finden guter Kombinationen, erkennt Probleme früher und arbeitet sinnvoll mit menschlicher Expertise zusammen. Das Ziel ist nicht "Zero Human Intervention", sondern **maximale Human-AI Synergie** – wo jeder das tut, was er am besten kann.