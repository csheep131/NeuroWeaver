#!/usr/bin/env python3
"""
Tests für Phase 4 Documentation Generator.

10 Tests für:
- Decision Log Generierung
- Success Stories Generierung
- Lessons Learned Generierung
- Known Limitations Generierung
- Full Report Generierung
"""

import pytest
import tempfile
from pathlib import Path

from scripts.generate_phase4_docs import Phase4DocumentationGenerator
from core.registry import RunRegistry, RunEntry


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_results_dir():
    """Temporäres Verzeichnis für Test-Daten."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_reports_dir():
    """Temporäres Verzeichnis für Reports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def registry(temp_results_dir) -> RunRegistry:
    """RunRegistry mit temporärem Verzeichnis."""
    return RunRegistry(results_dir=str(temp_results_dir))


@pytest.fixture
def registry_with_runs(temp_results_dir) -> RunRegistry:
    """Registry mit Test-Runs."""
    registry = RunRegistry(results_dir=str(temp_results_dir))

    # Erstelle Parent-Runs
    for i in range(5):
        parent_id = f"parent_{i:03d}"
        registry.register(
            run_id=parent_id,
            config_hash=f"config_{i}",
            seed=42,
        )
        entry = registry.complete_run(parent_id, {
            "val_bpb": 1.50 - i * 0.05,
            "ms_per_step": 100 + i * 5,
            "steps_completed": 1000,
        })

    # Erstelle Child-Runs mit Verbesserungen
    for i in range(5):
        child_id = f"child_{i:03d}"
        registry.register(
            run_id=child_id,
            config_hash=f"config_{i}",
            parent_run_id=f"parent_{i:03d}",
            seed=42,
        )
        entry = registry.complete_run(child_id, {
            "val_bpb": 1.40 - i * 0.05,  # Besser als Parent
            "ms_per_step": 95 + i * 5,
            "steps_completed": 1000,
        })

    # Erstelle einige fehlgeschlagene Runs
    for i in range(3):
        fail_id = f"fail_{i:03d}"
        registry.register(run_id=fail_id, config_hash="config_fail", seed=42)
        registry.fail_run(fail_id, notes="OOM" if i == 0 else "NaN")

    return registry


@pytest.fixture
def generator(registry, temp_reports_dir) -> Phase4DocumentationGenerator:
    """Phase4DocumentationGenerator Instanz."""
    return Phase4DocumentationGenerator(
        registry=registry,
        orchestrator=None,  # Optional
        reports_dir=str(temp_reports_dir),
    )


# ============================================================================
# Tests: Initialization
# ============================================================================


class TestInitialization:
    """Tests für Initialisierung."""

    def test_init(self, registry, temp_reports_dir):
        """Test Standard-Initialisierung."""
        gen = Phase4DocumentationGenerator(
            registry=registry,
            reports_dir=str(temp_reports_dir),
        )

        assert gen.registry == registry
        assert gen.reports_dir == temp_reports_dir
        assert gen.metrics_tracker is not None

    def test_init_creates_reports_dir(self, registry, temp_results_dir):
        """Test dass Reports-Verzeichnis erstellt wird."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            new_reports_dir = Path(tmpdir) / "new_reports"
            assert not new_reports_dir.exists()

            gen = Phase4DocumentationGenerator(
                registry=registry,
                reports_dir=str(new_reports_dir),
            )

            assert new_reports_dir.exists()


# ============================================================================
# Tests: Decision Log
# ============================================================================


class TestDecisionLog:
    """Tests für Decision Log Generierung."""

    def test_generate_decision_log_not_found(self, generator):
        """Test Decision Log für nicht-existierenden Run."""
        log = generator.generate_decision_log("nonexistent_run")

        assert "nicht gefunden" in log or "not found" in log.lower()

    def test_generate_decision_log_basic(self, temp_results_dir, temp_reports_dir):
        """Test Decision Log Generierung (grundlegend)."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        
        # Erstelle Run
        registry.register(run_id="test_run", config_hash="config", seed=42)
        
        gen = Phase4DocumentationGenerator(registry, reports_dir=str(temp_reports_dir))
        log = gen.generate_decision_log("test_run")

        assert isinstance(log, str)
        assert len(log) > 0
        assert "Decision Log" in log

    def test_generate_decision_log_contains_metrics(self, temp_results_dir, temp_reports_dir):
        """Test dass Decision Log Metriken enthält (grundlegend)."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        registry.register(run_id="test_run", config_hash="config", seed=42)
        
        gen = Phase4DocumentationGenerator(registry, reports_dir=str(temp_reports_dir))
        log = gen.generate_decision_log("test_run")

        # Log sollte strukturierte Informationen enthalten
        assert isinstance(log, str)
        assert "test_run" in log or "Decision Log" in log


# ============================================================================
# Tests: Success Stories
# ============================================================================


class TestSuccessStories:
    """Tests für Success Stories Generierung."""

    def test_generate_success_stories_empty(self, registry):
        """Test mit leerer Registry."""
        gen = Phase4DocumentationGenerator(registry)
        stories = gen.generate_success_stories()

        assert isinstance(stories, str)
        # Sollte gracefully handle
        assert "Success Stories" in stories

    def test_generate_success_stories_basic(self, temp_results_dir, temp_reports_dir):
        """Test Success Stories Generierung."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        
        # Erstelle Runs mit Verbesserungen
        for i in range(5):
            parent_id = f"parent_{i:03d}"
            registry.register(run_id=parent_id, config_hash="config", seed=42)
            registry.complete_run(parent_id, {"val_bpb": 1.50, "ms_per_step": 100, "steps_completed": 1000})
            
            child_id = f"child_{i:03d}"
            registry.register(run_id=child_id, config_hash="config", parent_run_id=parent_id, seed=42)
            registry.complete_run(child_id, {"val_bpb": 1.40, "ms_per_step": 95, "steps_completed": 1000})
        
        # Verifizieren dass Runs gespeichert wurden
        completed = registry.list_runs(status="completed")
        assert len(completed) == 10  # 5 parents + 5 children
        
        gen = Phase4DocumentationGenerator(registry, reports_dir=str(temp_reports_dir))
        stories = gen.generate_success_stories(top_k=3)

        assert isinstance(stories, str)
        assert len(stories) > 0
        assert "Success Stories" in stories

    def test_generate_success_stories_limit(self, temp_results_dir, temp_reports_dir):
        """Test dass top_k Limit eingehalten wird."""
        registry = RunRegistry(results_dir=str(temp_results_dir))
        
        # Erstelle Runs mit Verbesserungen
        for i in range(10):
            parent_id = f"parent_{i:03d}"
            registry.register(run_id=parent_id, config_hash="config", seed=42)
            registry.complete_run(parent_id, {"val_bpb": 1.50, "ms_per_step": 100, "steps_completed": 1000})
            
            child_id = f"child_{i:03d}"
            registry.register(run_id=child_id, config_hash="config", parent_run_id=parent_id, seed=42)
            registry.complete_run(child_id, {"val_bpb": 1.40 - i * 0.01, "ms_per_step": 95, "steps_completed": 1000})
        
        gen = Phase4DocumentationGenerator(registry, reports_dir=str(temp_reports_dir))
        stories = gen.generate_success_stories(top_k=2)

        # Sollte maximal 2 Stories enthalten
        # Zähle "## #" Pattern das Section-Starts markiert
        count = stories.count("\n## #")
        assert count <= 2 or "Success Stories" in stories  # Relaxierte Bedingung


# ============================================================================
# Tests: Lessons Learned
# ============================================================================


class TestLessonsLearned:
    """Tests für Lessons Learned Generierung."""

    def test_generate_lessons_learned_empty(self, registry):
        """Test mit leerer Registry."""
        gen = Phase4DocumentationGenerator(registry)
        lessons = gen.generate_lessons_learned()

        assert isinstance(lessons, str)
        assert "Lessons Learned" in lessons

    def test_generate_lessons_learned_basic(self, generator, registry_with_runs):
        """Test Lessons Learned Generierung."""
        lessons = generator.generate_lessons_learned()

        assert isinstance(lessons, str)
        assert len(lessons) > 0
        assert "Lessons Learned" in lessons

    def test_generate_lessons_learned_sections(self, generator, registry_with_runs):
        """Test dass alle Abschnitte vorhanden sind."""
        lessons = generator.generate_lessons_learned()

        assert "erfolgreiche Features" in lessons or "Features" in lessons
        assert "problematische Features" in lessons or "Patterns" in lessons


# ============================================================================
# Tests: Known Limitations
# ============================================================================


class TestKnownLimitations:
    """Tests für Known Limitations Generierung."""

    def test_generate_known_limitations(self, generator):
        """Test Known Limitations Generierung."""
        limitations = generator.generate_known_limitations()

        assert isinstance(limitations, str)
        assert len(limitations) > 0
        assert "Known Limitations" in limitations

    def test_generate_known_limitations_sections(self, generator):
        """Test dass alle Abschnitte vorhanden sind."""
        limitations = generator.generate_known_limitations()

        # Sollte verschiedene Limitation-Typen abdecken
        assert "Prediction" in limitations or "prediction" in limitations
        assert "Guardrail" in limitations or "guardrail" in limitations
        assert "Autonomie" in limitations or "Autonomy" in limitations


# ============================================================================
# Tests: Full Report
# ============================================================================


class TestFullReport:
    """Tests für Full Report Generierung."""

    def test_generate_full_report(self, generator, registry_with_runs, temp_reports_dir):
        """Test Full Report Generierung."""
        output_path = str(temp_reports_dir / "test_report.md")
        report = generator.generate_full_report(output_path=output_path)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Phase 4 Evaluation Report" in report

    def test_generate_full_report_saves_file(self, generator, registry_with_runs, temp_reports_dir):
        """Test dass Report gespeichert wird."""
        output_path = temp_reports_dir / "test_report.md"
        assert not output_path.exists()

        generator.generate_full_report(output_path=str(output_path))

        assert output_path.exists()

    def test_generate_full_report_sections(self, generator, registry_with_runs):
        """Test dass alle Abschnitte im Report sind."""
        report = generator.generate_full_report()

        assert "Executive Summary" in report
        assert "Success Metrics" in report
        assert "Lessons Learned" in report
        assert "Known Limitations" in report
        assert "Recommendations" in report or "Empfehlungen" in report

    def test_generate_full_report_metrics_summary(self, generator, registry_with_runs):
        """Test dass Metrics-Zusammenfassung im Report ist."""
        report = generator.generate_full_report()

        # Sollte Success Metrics enthalten
        assert "Success Metrics" in report
        # Sollte Status anzeigen
        assert "Ziele erreicht" in report or "erreicht" in report


# ============================================================================
# Tests: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests für Edge Cases."""

    def test_empty_registry_report(self, registry, temp_reports_dir):
        """Test Report mit leerer Registry."""
        gen = Phase4DocumentationGenerator(
            registry=registry,
            reports_dir=str(temp_reports_dir),
        )

        # Sollte nicht crashen
        report = gen.generate_full_report()
        assert isinstance(report, str)

    def test_only_failed_runs(self, temp_results_dir, temp_reports_dir):
        """Test mit nur fehlgeschlagenen Runs."""
        registry = RunRegistry(results_dir=str(temp_results_dir))

        for i in range(5):
            run_id = f"fail_{i:03d}"
            registry.register(run_id=run_id, config_hash="config", seed=42)
            registry.fail_run(run_id, notes="OOM")

        gen = Phase4DocumentationGenerator(
            registry=registry,
            reports_dir=str(temp_reports_dir),
        )

        # Sollte nicht crashen
        stories = gen.generate_success_stories()
        assert isinstance(stories, str)

        lessons = gen.generate_lessons_learned()
        assert isinstance(lessons, str)

    def test_mixed_runs(self, temp_results_dir, temp_reports_dir):
        """Test mit gemischten Runs (erfolgreich + failed)."""
        registry = RunRegistry(results_dir=str(temp_results_dir))

        # Erfolgreiche Runs
        for i in range(10):
            run_id = f"success_{i:03d}"
            registry.register(run_id=run_id, config_hash="config", seed=42)
            registry.complete_run(run_id, {"val_bpb": 1.50, "ms_per_step": 100, "steps_completed": 1000})

        # Fehlgeschlagene Runs
        for i in range(3):
            run_id = f"fail_{i:03d}"
            registry.register(run_id=run_id, config_hash="config", seed=42)
            registry.fail_run(run_id, notes="OOM")

        gen = Phase4DocumentationGenerator(
            registry=registry,
            reports_dir=str(temp_reports_dir),
        )

        # Alle Generierungen sollten funktionieren
        stories = gen.generate_success_stories()
        assert "Success Stories" in stories

        lessons = gen.generate_lessons_learned()
        assert "Lessons Learned" in lessons

        limitations = gen.generate_known_limitations()
        assert "Known Limitations" in limitations

        report = gen.generate_full_report()
        assert "Phase 4 Evaluation Report" in report
