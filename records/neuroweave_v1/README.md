# neuroweave_v1 — Systematische Ablation über Architektur, Tokenisierung und Quantisierung

**Author:** Thomas Speckert (@csheep131)  
**Date:** 2026-03-25  
**Run ID:** neuroweave_v1

## Ergebnis

| Metrik | Wert |
|--------|------|
| val_bpb | 1.15 |
| vs. Baseline | -.0744 |
| Training Time | ~10 min on 8xH100 |
| Artifact Size | <16MB |

## Ansatz

Systematische Ablation über Architektur, Tokenisierung und Quantisierung

### Wichtigste Änderungen

- [Feature 1: z.B. GQA mit gqa_groups=4]
- [Feature 2: z.B. SwiGLU Gated MLP]
- [Feature 3: z.B. TrigramHash-8192 Tokenizer]
- [Feature 4: z.B. INT6-Quantisierung]

## Reproduzierbarkeit

```bash
# Training ausführen
RUN_ID=neuroweave_v1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Logs

- Training Log: neuroweave_v1_train.log
- Final Metrics: Siehe submission.json

## Anforderungen

Siehe Haupt-README.md im Repository-Root für:
- Setup-Anweisungen
- Abhängigkeiten (requirements.txt)
- Hardware-Anforderungen
