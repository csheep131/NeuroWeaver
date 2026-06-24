# Phase 2 Implementierungsbericht

## Zusammenfassung

Phase 2 implementiert die **Research Engine** mit Feature-Gates, erweiterter Tokenizer- und Quantizer-Unterstützung, sowie automatisierten Ablation-Reports mit Kill-Regeln.

## Implementierte Komponenten

### 2.1 Backbone Factory (`models/factories/`)

**backbone_factory.py:**
- `ArchitectureConfig`: Deklarative Konfiguration aller Architektur-Parameter
- `ModelSpec`: Spezifikation für Modellerstellung mit Parameter-Schätzung
- `BackboneFactory`: Factory für Modellerstellung (Rust oder Python-Fallback)

**feature_gate.py:**
- `FeatureGate`: Definition von Feature-Gates mit Dependencies und Kill-Rules
- `FeatureGateManager`: Verwaltung aller Feature-Gates
- Vordefinierte Gates für: XSA, FiLM, TTT, Hash-Tokenizer, LeakyReLU, Gated-MLP, Mixed-Quant

**Verwendung:**
```python
from models.factories import BackboneFactory, create_feature_manager

factory = BackboneFactory()
model = factory.create(config)

feature_mgr = create_feature_manager()
feature_mgr.enable("xsa")
feature_mgr.validate_all(config.to_dict())
```

### 2.2 Tokenizer-Lab (`tokenizers/`)

**tokenizers.py:**
- `ByteTokenizer`: Byte-level Tokenisierung (Baseline)
- `BigramHashTokenizer`: Hash-basierte Bigram-Tokenisierung
- `TrigramHashTokenizer`: Hash-basierte Trigram-Tokenisierung
- `FallbackTokenizer`: Kombiniert Tokenizer mit Fallback
- `TokenizerFactory`: Factory für Tokenizer-Erstellung

**Verwendung:**
```python
from tokenizers import create_tokenizer

byte_tok = create_tokenizer("byte")
bigram_tok = create_tokenizer("bigram_hash", vocab_size=4096)
trigram_tok = create_tokenizer("trigram_hash", vocab_size=8192)
```

### 2.3 Quant-Lab (`quant/`)

**quantizers.py:**
- `Int6Quantizer`: 6-Bit-Quantisierung (0-63 Range)
- `Int5Quantizer`: 5-Bit-Quantisierung (0-31 Range)
- `MixedQuantizer`: Kombiniert INT5 und INT6 basierend auf Gewicht-Magnitude
- `GPTQLiteQuantizer`: Gruppenweise Quantisierung nach GPTQ-Art
- `QuantizerFactory`: Factory für Quantizer

**Verwendung:**
```python
from quant import create_quantizer

int6 = create_quantizer("int6")
mixed = create_quantizer("int5_int6_mixed")
gptq = create_quantizer("gptq_lite", group_size=128)
```

### 2.4 Ablation Reporter (`research/`)

**ablation_engine.py:**
- `KillRule`: Definition von Kill-Regeln mit Priorität
- `KillReason`: Enum für Kill-Gründe
- `AblationReport`: Umfassender Report mit Statistiken
- `AblationReporter`: Hauptklasse mit Kill-Rule-Engine

**Kill-Regeln (aus Blueprint):**
1. **Artifact > 16MB**: `MAX_ARTIFACT_BYTES = 16_000_000`
2. **Slow without gain**: `delta_ms > 2.0` ohne `delta_bpb < -0.05`
3. **Quant gap**: `quantized_bpb - bpb > 0.1`
4. **BPB regression**: `delta_bpb > 0.1` (schlechter als Parent)
5. **Volatile across seeds**: Wird in Registry erkannt

**Verwendung:**
```python
from research import AblationReporter

reporter = AblationReporter()
report = reporter.generate_report()
print(report.print_summary())

# Kill-Regeln anwenden
kills = reporter.apply_kills()
```

### 2.5 Lineage Tracking (`core/registry.py`)

**Erweiterte Methoden:**
- `get_lineage_tree(run_id)`: Vollständiger Stammbaum als Tree
- `get_all_lineages()`: Alle Parent-Child-Beziehungen
- `get_run_family(run_id)`: Alle Runs mit gleichem Vorfahren
- `get_config_family(config_hash)`: Runs mit gleicher Config (verschiedene Seeds)
- `get_seed_statistics(config_hash)`: Statistiken über Seeds (mean, std, min, max)
- `find_volatile_configs()`: Erkennt volatile Konfigurationen

## Neue Config-Dateien

- `configs/runs/run003_xsa.yaml`: XSA-Experiment
- `configs/runs/run004_leakyrelu.yaml`: LeakyReLU-Aktivierung
- `configs/runs/run005_mixed_quant.yaml`: Mixed-Precision-Quantisierung

## Getestete Funktionalität

### Backbone Factory Test
```
ModelSpec: 512d x 6L x 8H | 17.43M params | Activation: gelu | Attention: gqa | Features: None
```

### Tokenizer Test
```
Byte Tokenizer: [72, 101, 108, 108, 111, ...] (47 tokens)
Bigram Tokenizer: [72, 2385, 3543, ...] (47 tokens)
Trigram Tokenizer: [72, 101, 2717, ...] (47 tokens)
```

### Quantizer Test
```
INT6 Quantizer: avg error = 0.010329, compression = 5.33x
Mixed Quantizer: avg error = 0.464906, compression = 6.40x
```

### Ablation Reporter Test
```
RUN STATISTICS
Total runs: 4
Active: 4
Killed: 0
Failed: 0

BEST RESULTS
Best BPB: run003_xsa (1.4000)
Best BPB/MB: run003_xsa
Best BPB/ms: run003_xsa

[KILL] run004_tired
Reason: Artifact size 17,000,000 bytes exceeds 16,000,000 byte limit
```

## Architektur-Übersicht

```
Phase 2 Architektur:

configs/
runs/
run001_control.yaml # Baseline
run002_hash.yaml # Hash Tokenizer
run003_xsa.yaml # XSA Feature
run004_leakyrelu.yaml # LeakyReLU
run005_mixed_quant.yaml # Mixed Quant

models/factories/
backbone_factory.py # Model Builder
feature_gate.py # Feature Management

tokenizers/
tokenizers.py # Byte, Bigram, Trigram

quant/
quantizers.py # INT5, INT6, Mixed, GPTQ-lite

research/
ablation_engine.py # Kill Rules, Reports

core/
registry.py # +Lineage Tracking
```

## Feature-Gates im Detail

| Gate | Beschreibung | Kill-Kriterium |
|------|--------------|----------------|
| xsa | Cross-Sequence Attention | >2ms/step ohne BPB-Gewinn |
| film | Feature-wise Linear Modulation | Artifact > 16MB |
| ttt | Test-Time Training | >50ms/step |
| hasher | Hash Tokenizer | BPB > 2.0 |
| leaky_relu | LeakyReLU-Aktivierung | - |
| gated_mlp | Gated MLP | Artifact > 16MB |
| mixed_quant | Mixed INT5/INT6 | Quant-Gap > 0.1 BPB |

## Nächste Schritte (Phase 3)

- [ ] Sweep Runner für automatische Experiment-Sets
- [ ] Promotion System (1 Seed → 3 Seeds → Final)
- [ ] Submission Builder für GitHub-PRs
- [ ] Dashboard/CLI für Run-Übersicht
- [ ] Multi-Seed-Orchestrierung

## MVP Status Phase 2

- [x] Backbone Factory mit Feature-Gates
- [x] Tokenizer-Lab (byte, bigram, trigram)
- [x] Quant-Lab (int6, int5, mixed)
- [x] Ablation Reporter mit Kill-Regeln
- [x] Lineage Tracking erweitert
- [x] Alle Komponenten getestet
