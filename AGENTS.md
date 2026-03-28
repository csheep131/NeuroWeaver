# AGENTS.md — Verbindliche Regeln fuer alle AI-Agenten

Dieses Dokument ist PFLICHTLEKTUERE fuer jeden Agenten der in diesem
Repository arbeitet (Hermes, Qwen, Cline, Codex, etc.).

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

1. train_gpt.py verbessern (niedrigerer val_bpb)
2. Dabei die Constraints einhalten (16MB, 10min)
3. Aenderungen testen und dokumentieren
4. Das Tooling (orchestrator etc.) nur als Hilfsmittel nutzen

Optimierungs-Ideen die nachweislich gut funktionieren
(siehe Leaderboard in regeln.md):

- Mehr Layer (10-11 statt 9)
- INT6/INT5 Quantisierung statt INT8
- GQA, XSA (Cross-Scale Attention)
- EMA statt SWA
- GPTQ-lite Quantisierung
- Sliding Window Evaluation
- LeakyReLU² Activation
- Partial RoPE
- BigramHash Tokenizer
- Test-Time Training (TTT)
- Muon Optimizer mit Weight Decay

## Regel 6: Englisch im Submission-Repo

Alles was in Repo 2 (Fork/PR) landet muss auf Englisch sein.
README.md, submission.json, Kommentare im Code — alles Englisch.

Hier in Repo 1 ist Deutsch OK.
