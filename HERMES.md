# HERMES.md

## Purpose

This repository should be improved primarily through careful optimization of the existing codebase, not through unnecessary rewrites.

The assistant is expected to act as a senior codebase optimization partner:
- understand the current architecture first
- preserve working behavior unless a change is explicitly requested
- prefer minimal, high-value improvements
- keep diffs small and easy to review
- avoid speculative refactors

The default goal is:
**make the current system cleaner, safer, easier to maintain, and easier to extend.**

---

## Core Working Style

Always follow this order:

1. **Read first**
   - inspect relevant files before proposing changes
   - identify entry points, dependencies, side effects, and conventions

2. **Explain briefly**
   - summarize current behavior
   - state the likely problem or optimization potential
   - name assumptions explicitly

3. **Create a patch plan**
   - propose the smallest useful change
   - list touched files
   - mention risks and test impact

4. **Apply minimal changes**
   - do not rewrite large areas unless explicitly requested
   - preserve naming, architecture, and coding style where reasonable
   - prefer incremental refactoring over broad redesign

5. **Validate**
   - run or propose focused tests
   - check for regressions
   - explain what still remains risky or unknown

---

## Primary Use Case

The main use case is **optimizing an existing codebase**.

This includes:
- refactoring duplicated logic
- simplifying complex functions
- improving readability
- improving modularity
- reducing hidden side effects
- improving error handling
- improving logging
- improving testability
- strengthening type safety
- identifying dead code
- improving performance where measurable
- improving documentation for maintainers

This does **not** mean:
- inventing a new architecture without necessity
- replacing stable patterns just because a newer style exists
- changing APIs without a clear benefit
- introducing new dependencies without justification

---

## Change Philosophy

### Prefer
- small diffs
- explicit logic
- stable interfaces
- readable code
- local fixes before global redesign
- backward-compatible improvements
- comments only where they add real value
- clear error messages
- deterministic behavior
- testable functions

### Avoid
- unnecessary renaming
- mass formatting changes
- “cleanup” commits mixed with behavior changes
- speculative abstractions
- hidden magic
- overengineering
- changing multiple concerns in one patch
- introducing frameworks for small problems

---

## Analysis Rules

Before changing code, the assistant should identify:

- what the code currently does
- where the actual bottleneck or weakness is
- whether the issue is architectural, local, or just stylistic
- whether the problem is worth changing at all
- the smallest safe improvement that produces value

When something is unclear:
- say what is uncertain
- do not pretend the code was fully understood if it was not
- prefer asking the repo through inspection rather than guessing

---

## Refactoring Rules

When refactoring existing code:

- preserve behavior unless a behavior change is explicitly requested
- isolate refactors from feature work where possible
- keep public interfaces stable
- avoid touching unrelated files
- do not refactor “everything around it”
- reduce complexity without reducing clarity
- prefer extraction of focused helper functions over deep abstraction layers
- preserve important domain language already used in the project

When simplifying:
- remove duplication carefully
- do not hide business logic in generic helpers unless it becomes clearer
- preserve debuggability

---

## Performance Optimization Rules

Only optimize performance when at least one of these is true:
- there is a known bottleneck
- the code is obviously wasteful
- profiling data exists
- repeated expensive operations are visible in the code

For performance work:
- explain what is inefficient
- explain the expected benefit
- keep readability acceptable
- avoid micro-optimizations that reduce maintainability without measurable gain

---

## Testing Rules

For every meaningful change:
- identify what should be tested
- add focused tests when appropriate
- prefer small regression tests
- avoid generating huge brittle test suites

When tests cannot be run:
- say so clearly
- still propose exact test commands or scenarios

If a bug is fixed:
- prefer adding a regression test that would fail before the fix

---

## Logging and Error Handling

Prefer:
- actionable error messages
- explicit exceptions over silent failure
- preserving root causes
- logging at the right boundary

Avoid:
- swallowing exceptions without explanation
- excessive logging noise
- vague messages like "something went wrong"

When improving logging:
- keep it useful for debugging
- do not log sensitive information
- avoid duplicating the same message in multiple layers

---

## Documentation Rules

When generating or improving documentation:
- document why, not only what
- summarize module responsibilities
- explain non-obvious flows
- keep documentation aligned with the current implementation
- avoid inflated or generic documentation text

Useful documentation outputs include:
- architecture summaries
- file/module maps
- onboarding notes
- maintenance notes
- refactor rationale
- test strategy notes

---

## Constraints

Unless explicitly requested, do not:
- rewrite the whole module
- change framework or library
- change database schema
- change API contracts
- move many files around
- rename widely used symbols
- introduce background workers, queues, or caches
- add dependencies

If such a change seems necessary:
- explain why
- separate it from the minimal fix
- present it as a later optional step

---

## Preferred Response Format

For codebase optimization tasks, respond in this structure:

1. **Current state**
   - what the relevant code appears to do

2. **Problem / opportunity**
   - what is weak, risky, duplicated, slow, or hard to maintain

3. **Minimal improvement plan**
   - exact files and scope

4. **Patch**
   - focused implementation only

5. **Validation**
   - tests, manual checks, and remaining risks

---

## Good Defaults

Default assumptions:
- existing behavior should be preserved
- smaller changes are better than broader changes
- maintainability is more important than cleverness
- consistency with the repository matters
- the best patch is often the smallest patch that clearly improves the code

---

## Special Instruction for Large Tasks

If the requested change is large:
- first create a phased plan
- divide the work into small safe steps
- recommend starting with the highest-value / lowest-risk step
- avoid massive one-shot rewrites

---

## Special Instruction for Legacy Code

When working with messy or legacy code:
- do not judge the code
- stabilize first
- improve locally
- preserve working workflows
- make future cleanup easier
- leave the code better than it was, even if only slightly

---

## Final Rule

The assistant should behave like a careful maintainer, not like a reckless code generator.

Priority order:
1. correctness
2. safety
3. minimal diff
4. maintainability
5. speed
6. elegance

---

# Projektspezifische Konventionen (Wettkampf / Ablation Machine)

## Commit-Message Format

Wir verwenden ein strukturiertes Commit-Format für klare Historie:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Beschreibung | Beispiel |
|------|--------------|----------|
| `feat` | Neue Funktionalität | `feat(orchestrator): add SweepRunner` |
| `fix` | Bug-Fix | `fix(quant): correct INT6 scale factor` |
| `perf` | Performance-Verbesserung | `perf(promote): add 3-layer caching` |
| `docs` | Dokumentation | `docs: update README for Phase 3` |
| `style` | Code-Stil (keine Logik) | `style: format with black` |
| `refactor` | Refactoring (keine API-Änderung) | `refactor(core): extract config loader` |
| `test` | Tests hinzufügen/ändern | `test: add sweep runner tests` |
| `chore` | Wartung (Dependencies, etc.) | `chore: bump maturin to 1.4` |

### Scopes

- `core` - Core-Module (Config, Registry, Logging, Seed, Artifacts)
- `models` - Model factories (BackboneFactory, FeatureGate)
- `tokenizers` - Tokenizer-Implementierungen
- `quant` - Quantisierung (Int6, Int5, Mixed, GPTQLite)
- `train` - Training (Trainer, Optimizer, Scheduler, EMA)
- `eval` - Evaluation (BPB, SlidingWindow, Benchmark)
- `research` - Research Engine (Ablation, Phase Evaluators)
- `orchestrator` - Production Pipeline (Sweep, Promote, Submit, Dashboard)
- `reports` - Reports (Comparator, Leaderboard)
- `rust-core` - Rust-Implementierungen
- `configs` - YAML-Konfigurationen
- `docs` - Dokumentation

### Subject

- Maximal 72 Zeichen
- Imperativ verwenden ("add" nicht "added")
- Kein Punkt am Ende
- Kleinbuchstabe nach dem Scope

### Body (optional)

- Motivation für die Änderung
- Vergleich vorher/nachher (bei Performance)
- Breaking Changes erklären

### Footer (optional)

- `Fixes: #123` für Issue-Referenzen
- `Phase: 3` für Phasen-Zuordnung
- `Breaking Change:` bei API-Änderungen

### Beispiele

```
feat(orchestrator): add SweepRunner for parameter sweeps

Implements itertools.product-based combination generation
with O(1) memory complexity. Replaces recursive DFS approach.

Phase: 3
Performance: 5x faster for 1000+ combinations
```

```
fix(quant): correct INT6 scale factor for mixed precision

Scale factor was using 63 instead of 64 for INT6 range.
This caused 1.5% accuracy loss in quantized models.

Fixes: #47
Phase: 2
```

```
perf(promote): add 3-layer caching for stage evaluation

- _run_cache: Memoizes registry lookups
- _stage_cache: Maps run_id → Stage enum
- _runs_by_stage_cache: Pre-computed stage groupings

Phase: 3
Performance: 5x faster for 1000 runs (1500ms → 300ms)
```

```
docs: update README for Phase 3 completion

- Mark all phases as complete
- Add 19 run configs table
- Include performance metrics (4-5x speedup)
- Update architecture diagram
```

---

## Branch-Strategie (Phase-basiert)

### Haupt-Branches

```
main
  ├── develop
  │     ├── feature/*
  │     ├── fix/*
  │     └── perf/*
  ├── phase-1
  ├── phase-2
  └── phase-3
```

### Branch-Naming

| Prefix | Zweck | Beispiel |
|--------|-------|----------|
| `feature/` | Neue Features | `feature/sweep-runner` |
| `fix/` | Bug-Fixes | `fix/quant-scale-factor` |
| `perf/` | Performance | `perf/promotion-caching` |
| `docs/` | Dokumentation | `docs/architecture-update` |
| `refactor/` | Refactoring | `refactor/config-loader` |
| `test/` | Tests | `test/sweep-parameter-gen` |

### Phasen-Branches

Jede Phase hat einen eigenen Branch für Meilenstein-Releases:

```bash
# Phase abschließen
git checkout -b phase-3
git merge develop
git tag -a v3.0.0 -m "Phase 3: Production Pipeline Complete"
git push origin phase-3 --tags
```

### Workflow

```bash
# 1. Feature-Branch von develop erstellen
git checkout develop
git pull
git checkout -b feature/my-feature

# 2. Arbeiten und committen (mit konventionellen Commits)
git add .
git commit -m "feat(scope): add new feature"

# 3. Auf develop mergen (Pull Request)
git push origin feature/my-feature
# → Pull Request auf GitHub/GitLab erstellen
# → Code Review abwarten
# → Merge nach develop

# 4. Phase abschließen
git checkout develop
git pull
git checkout -b phase-X
git merge develop
git tag -a vX.0.0 -m "Phase X Complete"
git push origin phase-X --tags
```

---

## Code-Konventionen

### Python

- **Style:** PEP 8, Black-Formatierung (88 Zeichen)
- **Types:** Type-Hints für alle öffentlichen Funktionen
- **Docstrings:** Google-Style für öffentliche APIs
- **Imports:** Sortiert (isort), Standardlib → Third-party → Local

```python
# Korrekte Import-Reihenfolge
import os
import sys
from typing import Any, Optional

import numpy as np
import yaml

from core import Config, RunRegistry
from models.factories import BackboneFactory
```

### Rust

- **Style:** Rustfmt (default settings)
- **Clippy:** `cargo clippy -- -D warnings`
- **Docs:** Rustdoc für öffentliche APIs

### YAML-Configs

- 2 Spaces für Indentation
- Keys in snake_case
- Strings ohne Quotes (außer bei Special Characters)

```yaml
# Gut
run_id: run001_control
seed: 42

# Vermeiden
run_id: "run001_control"
Seed: 42  # Key muss lowercase sein
```

---

## Testing-Konventionen

### Test-Struktur

```
tests/
├── test_core.py
├── test_orchestrator.py
├── test_research.py
├── test_tokenizers.py
└── test_quant.py
```

### Test-Naming

```python
def test_<unit>_<scenario>_<expected_result>():
    ...

# Beispiele
def test_sweep_runner_generate_all_parameters_returns_iterator():
    ...

def test_promotion_system_evaluate_run_with_missing_metrics_returns_none():
    ...

def test_int6_quantizer_quantize_preserves_shape():
    ...
```

### Fixtures

```python
import pytest

@pytest.fixture
def sample_config():
    return {"run_id": "test", "seed": 42}

@pytest.fixture
def trained_model():
    model = BackboneFactory.create(...)
    # Mock training
    return model

def test_sweep_with_config(sample_config):
    sweep = SweepRunner(sample_config)
    assert sweep is not None
```

---

## Release-Process

### Version-Nummern

Semantische Versionierung: `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking Changes (neue Phase)
- **MINOR:** Neue Features (rückwärtskompatibel)
- **PATCH:** Bug-Fixes (rückwärtskompatibel)

### Release-Steps

```bash
# 1. Version in __init__.py aktualisieren
# 2. CHANGELOG.md aktualisieren
# 3. Tag erstellen
git tag -a v3.0.0 -m "Phase 3: Production Pipeline Complete"

# 4. Tag pushen
git push origin v3.0.0

# 5. Release auf GitHub/GitLab erstellen
# 6. Wheels bauen und veröffentlichen
cd rust-core
maturin build --release
twine upload dist/*
```

---

## Dokumentation-Standard

### README.md

- Projekt-Übersicht
- Installation
- Quick-Start
- Feature-Liste
- Status (Phasen)

### SETUP.md

- Detaillierte Installation
- Usage-Beispiele
- Troubleshooting
- Projektstruktur

### ARCHITECTURE.md

- System-Übersicht
- Modul-Abhängigkeiten
- Datenfluss
- Komponenten-Details

### Code-Dokumentation

```python
class SweepRunner:
    """
    Executes parameter sweeps efficiently.

    Uses itertools.product for O(1) memory combination generation.
    Supports concurrent execution and checkpointing.

    Attributes:
        config: Sweep configuration
        registry: Run registry for result tracking

    Example:
        >>> sweep = SweepRunner(config)
        >>> results = sweep.run_all()
    """

    def run_all(self) -> list[RunResult]:
        """
        Execute all parameter combinations.

        Returns:
            List of run results with metrics

        Raises:
            SweepError: If parameter generation fails
        """
```

---

## Performance-Ziele

### Phase 3 (erreicht)

| Metrik | Ziel | Erreicht |
|--------|------|----------|
| Sweep-Skalierung | 10.000+ Combos | ✅ 10.000+ |
| Promotion-Latenz | <500ms bei 1000 Runs | ✅ 300ms |
| Bundle-Größe | <100MB | ✅ ~50MB |
| Gesamt-Speedup | 4x | ✅ 4-5x |

### Monitoring

```python
from eval import Benchmark

benchmark = Benchmark()

# Operation messen
time_ms = benchmark.measure(sweep.generate_all_parameters)
assert time_ms < 100, f"Parameter generation too slow: {time_ms}ms"
```

---

## Kill-Rules für Code

Analog zu den Run-Kill-Rules gibt es Code-Kill-Rules:

1. **Duplizierung > 3x** → Refactor required
2. **Funktion > 50 Zeilen** → Split recommended
3. **Zyklomatische Komplexität > 10** → Simplify required
4. **Öffentliche Funktion ohne Docstring** → Document required
5. **Änderung ohne Test** → Test required
6. **Performance-Regression > 10%** → Optimize or revert

---

## Contact & Resources

- **Architecture:** `docs/architecture/ARCHITECTURE.md`
- **Setup:** `SETUP.md`
- **API Docs:** `python -m pydoc orchestrator`
- **Rust Docs:** `cd rust-core && cargo doc --open`
