"""
Unit tests for core/meta_features.py.

Tests für Meta-Feature Extraktion und Analyse.

Usage:
    pytest tests/test_meta_features.py -v
    python -m pytest tests/test_meta_features.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

from core.meta_features import MetaFeatureExtractor, RunMetaFeatures
from core.registry import RunEntry, RunRegistry


class TestRunMetaFeatures(unittest.TestCase):
    """Test die RunMetaFeatures Datenklasse."""

    def test_default_values(self):
        """Teste Standardwerte für Meta-Features."""
        features = RunMetaFeatures(run_id="test_run")
        
        self.assertEqual(features.run_id, "test_run")
        self.assertEqual(features.features_active, [])
        self.assertIsNone(features.parent_run_id)
        self.assertEqual(features.lineage_depth, 0)
        self.assertEqual(features.siblings_count, 0)
        self.assertEqual(features.budget_class, "medium")
        self.assertEqual(features.sequence_length, "local")
        self.assertEqual(features.quantization_type, "none")
        self.assertEqual(features.step_time_ms, 0.0)
        self.assertEqual(features.memory_usage_mb, 0.0)
        self.assertEqual(features.training_stability, 1.0)
        self.assertEqual(features.delta_bpb_vs_parent, 0.0)
        self.assertEqual(features.efficiency_gain_percent, 0.0)
        self.assertEqual(features.model_size_change_percent, 0.0)
        self.assertEqual(features.quant_gap, 0.0)
        self.assertEqual(features.seed_variance, 0.0)
        self.assertEqual(features.confidence_interval_width, 0.0)
        self.assertEqual(features.days_since_first_feature_introduction, 0)
        self.assertEqual(features.runs_since_feature_last_successful, 0)
        self.assertEqual(features.co_occurrence_with, {})
        self.assertEqual(features.previous_success_rate_in_similar_context, 0.0)

    def test_to_dict(self):
        """Teste Konvertierung zu Dictionary."""
        features = RunMetaFeatures(
            run_id="test_run",
            features_active=["gqa", "film"],
            parent_run_id="parent_run",
            lineage_depth=2,
            budget_class="high",
            quantization_type="int6",
            delta_bpb_vs_parent=-0.05,
        )
        
        data = features.to_dict()
        
        self.assertEqual(data["run_id"], "test_run")
        self.assertEqual(data["features_active"], ["gqa", "film"])
        self.assertEqual(data["parent_run_id"], "parent_run")
        self.assertEqual(data["lineage_depth"], 2)
        self.assertEqual(data["budget_class"], "high")
        self.assertEqual(data["quantization_type"], "int6")
        self.assertEqual(data["delta_bpb_vs_parent"], -0.05)

    def test_from_dict(self):
        """Teste Erstellen aus Dictionary."""
        data = {
            "run_id": "test_run",
            "features_active": ["gqa", "film", "leaky_relu"],
            "parent_run_id": "parent_run",
            "lineage_depth": 3,
            "siblings_count": 5,
            "budget_class": "low",
            "sequence_length": "remote",
            "quantization_type": "mixed",
            "step_time_ms": 15.5,
            "delta_bpb_vs_parent": -0.02,
        }
        
        features = RunMetaFeatures.from_dict(data)
        
        self.assertEqual(features.run_id, "test_run")
        self.assertEqual(features.features_active, ["gqa", "film", "leaky_relu"])
        self.assertEqual(features.parent_run_id, "parent_run")
        self.assertEqual(features.lineage_depth, 3)
        self.assertEqual(features.siblings_count, 5)
        self.assertEqual(features.budget_class, "low")
        self.assertEqual(features.sequence_length, "remote")
        self.assertEqual(features.quantization_type, "mixed")
        self.assertEqual(features.step_time_ms, 15.5)
        self.assertEqual(features.delta_bpb_vs_parent, -0.02)

    def test_roundtrip_serialization(self):
        """Teste Rundtrip Serialisierung (to_dict -> from_dict)."""
        original = RunMetaFeatures(
            run_id="test_run",
            features_active=["gqa", "rope"],
            parent_run_id="parent",
            lineage_depth=1,
            budget_class="medium",
            quantization_type="none",
            delta_bpb_vs_parent=-0.01,
            efficiency_gain_percent=5.0,
        )
        
        data = original.to_dict()
        restored = RunMetaFeatures.from_dict(data)
        
        self.assertEqual(original.run_id, restored.run_id)
        self.assertEqual(original.features_active, restored.features_active)
        self.assertEqual(original.parent_run_id, restored.parent_run_id)
        self.assertEqual(original.lineage_depth, restored.lineage_depth)
        self.assertEqual(original.budget_class, restored.budget_class)
        self.assertEqual(original.quantization_type, restored.quantization_type)
        self.assertEqual(original.delta_bpb_vs_parent, restored.delta_bpb_vs_parent)
        self.assertEqual(original.efficiency_gain_percent, restored.efficiency_gain_percent)


class TestMetaFeatureExtractor(unittest.TestCase):
    """Test die MetaFeatureExtractor Klasse."""

    def setUp(self):
        """Set up test fixtures."""
        # Erstelle temporäres Verzeichnis für Test-Configs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.configs_dir = Path(self.temp_dir.name) / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path(self.temp_dir.name) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Erstelle Test-Registry
        self.registry = RunRegistry(results_dir=str(self.results_dir))
        
        # Erstelle Extractor
        self.extractor = MetaFeatureExtractor(configs_dir=self.configs_dir)

    def tearDown(self):
        """Räume temporäres Verzeichnis auf."""
        self.temp_dir.cleanup()

    def _create_test_config(self, run_id: str, config: Dict[str, Any]) -> Path:
        """Erstelle Test-Konfigurationsdatei."""
        import yaml
        
        config_path = self.configs_dir / "runs"
        config_path.mkdir(parents=True, exist_ok=True)
        
        file_path = config_path / f"{run_id}.yaml"
        with open(file_path, "w") as f:
            yaml.dump(config, f)
        
        return file_path

    def _create_test_run_entry(self, run_id: str, **kwargs: Any) -> RunEntry:
        """Erstelle Test-Run-Eintrag im Registry."""
        return self.registry.register(
            run_id=run_id,
            config_hash=kwargs.get("config_hash", f"hash_{run_id}"),
            parent_run_id=kwargs.get("parent_run_id"),
            seed=kwargs.get("seed", 42),
        )

    def test_extract_basic_features(self):
        """Teste Extraktion grundlegender Features."""
        # Erstelle Test-Run mit explizit medium Budget (d_model=512, num_layers=6)
        run_id = "test_run"
        config = {
            "run_id": run_id,
            "model": {
                "d_model": 512,
                "num_layers": 8,  # Medium: 4-12 layers
                "attention": {
                    "type": "gqa",
                    "rope": True,
                },
            },
            "features": {
                "hasher": False,
            },
            "quant": {
                "enabled": False,
            },
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        # Extrahiere Features
        features = self.extractor.extract(run_id, self.registry)
        
        # Verifiziere grundlegende Eigenschaften
        self.assertEqual(features.run_id, run_id)
        self.assertIsInstance(features.features_active, list)
        self.assertIn("gqa", features.features_active)
        self.assertIn("rope", features.features_active)
        # Budget-Klasse hängt von approx_params ab (12 * d_model^2 * num_layers / 1e6)
        # 512^2 * 8 * 12 / 1e6 = ~25M Parameter -> medium
        self.assertIn(features.budget_class, ["low", "medium"])
        self.assertEqual(features.sequence_length, "local")
        self.assertEqual(features.quantization_type, "none")

    def test_extract_features_with_parent(self):
        """Teste Extraktion mit Parent-Run."""
        # Erstelle Parent-Run
        parent_id = "parent_run"
        parent_config = {
            "run_id": parent_id,
            "model": {"d_model": 512, "num_layers": 6},
            "quant": {"enabled": False},
        }
        self._create_test_config(parent_id, parent_config)
        parent_entry = self._create_test_run_entry(parent_id)
        
        # Aktualisiere Parent zuerst mit Metriken (wichtig für delta_bpb Berechnung)
        self.registry.complete_run(parent_id, {
            "val_bpb": 1.5,
            "ms_per_step": 12.0,
            "steps_completed": 100,
            "artifact_bytes": 12_000_000,
        })
        
        # Erstelle Child-Run
        child_id = "child_run"
        child_config = {
            "run_id": child_id,
            "parent_run_id": parent_id,
            "model": {"d_model": 512, "num_layers": 6},
            "quant": {"enabled": False},
        }
        self._create_test_config(child_id, child_config)
        child_entry = self._create_test_run_entry(child_id, parent_run_id=parent_id)
        
        # Aktualisiere Child mit Metriken
        self.registry.complete_run(child_id, {
            "val_bpb": 1.4,
            "ms_per_step": 10.0,
            "steps_completed": 100,
            "artifact_bytes": 10_000_000,
        })
        
        # Extrahiere Features
        features = self.extractor.extract(child_id, self.registry)
        
        # Verifiziere Lineage-Informationen
        self.assertEqual(features.parent_run_id, parent_id)
        self.assertGreaterEqual(features.lineage_depth, 1)
        
        # Verifiziere ΔBPB (sollte negativ sein = Verbesserung)
        # Hinweis: delta_bpb wird beim complete_run berechnet und im Entry gespeichert
        # Der Extractor liest diesen Wert direkt aus dem Entry
        self.assertLess(features.delta_bpb_vs_parent, 0)  # Verbesserung
        
        # Verifiziere Effizienz-Gewinn (sollte positiv sein = schneller)
        self.assertGreater(features.efficiency_gain_percent, 0)

    def test_extract_features_quantization(self):
        """Teste Extraktion von Quantisierungs-Features."""
        run_id = "quant_run"
        config = {
            "run_id": run_id,
            "model": {"d_model": 512, "num_layers": 6},
            "quant": {
                "enabled": True,
                "type": "int6",
            },
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertEqual(features.quantization_type, "int6")
        self.assertIn("quant_int6", features.features_active)

    def test_extract_features_mixed_quant(self):
        """Teste Extraktion von Mixed-Quantisierung."""
        run_id = "mixed_quant_run"
        config = {
            "run_id": run_id,
            "model": {"d_model": 512, "num_layers": 6},
            "quant": {
                "enabled": True,
                "type": "int5_int6_mixed",
            },
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertEqual(features.quantization_type, "mixed")
        self.assertIn("mixed_quant", features.features_active)

    def test_extract_features_gptq_lite(self):
        """Teste Extraktion von GPTQ-Lite Quantisierung."""
        run_id = "gptq_run"
        config = {
            "run_id": run_id,
            "model": {"d_model": 512, "num_layers": 6},
            "quant": {
                "enabled": True,
                "type": "gptq_lite",
                "gptq_lite": True,
            },
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertEqual(features.quantization_type, "gptq_lite")
        self.assertIn("gptq_lite", features.features_active)

    def test_budget_class_low(self):
        """Teste Budget-Klasse 'low'."""
        run_id = "low_budget_run"
        config = {
            "run_id": run_id,
            "model": {
                "d_model": 128,  # Klein
                "num_layers": 2,  # Flach
            },
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertEqual(features.budget_class, "low")

    def test_budget_class_high(self):
        """Teste Budget-Klasse 'high'."""
        run_id = "high_budget_run"
        config = {
            "run_id": run_id,
            "model": {
                "d_model": 1024,  # Groß
                "num_layers": 24,  # Tief
            },
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertEqual(features.budget_class, "high")

    def test_sequence_length_remote(self):
        """Teste Sequenzlänge 'remote'."""
        run_id = "remote_run"
        config = {
            "run_id": run_id,
            "model": {
                "d_model": 512,
                "num_layers": 6,
                "max_seq_len": 2048,  # Lang
            },
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertEqual(features.sequence_length, "remote")

    def test_lineage_depth(self):
        """Teste Lineage-Depth Berechnung."""
        # Erstelle Generationskette: grandparent -> parent -> child
        gp_id = "grandparent"
        p_id = "parent"
        c_id = "child"
        
        base_config = {
            "model": {"d_model": 512, "num_layers": 6},
            "quant": {"enabled": False},
        }
        
        # Grandparent
        base_config["run_id"] = gp_id
        self._create_test_config(gp_id, base_config)
        self._create_test_run_entry(gp_id)
        
        # Parent
        base_config["run_id"] = p_id
        base_config["parent_run_id"] = gp_id
        self._create_test_config(p_id, base_config)
        self._create_test_run_entry(p_id, parent_run_id=gp_id)
        
        # Child
        base_config["run_id"] = c_id
        base_config["parent_run_id"] = p_id
        self._create_test_config(c_id, base_config)
        self._create_test_run_entry(c_id, parent_run_id=p_id)
        
        # Extrahiere Features für Child
        features = self.extractor.extract(c_id, self.registry)
        
        # Sollte Depth 2 haben (parent + grandparent)
        self.assertEqual(features.lineage_depth, 2)

    def test_siblings_count(self):
        """Teste Siblings-Count Berechnung."""
        parent_id = "parent"
        
        base_config = {
            "model": {"d_model": 512, "num_layers": 6},
            "quant": {"enabled": False},
        }
        
        # Parent
        base_config["run_id"] = parent_id
        self._create_test_config(parent_id, base_config)
        self._create_test_run_entry(parent_id)
        
        # Drei Siblings
        for i in range(3):
            child_id = f"child_{i}"
            base_config["run_id"] = child_id
            base_config["parent_run_id"] = parent_id
            self._create_test_config(child_id, base_config)
            self._create_test_run_entry(child_id, parent_run_id=parent_id)
        
        # Extrahiere Features für letzten Child
        features = self.extractor.extract("child_2", self.registry)
        
        # Sollte 2 Siblings haben (child_0 und child_1)
        self.assertEqual(features.siblings_count, 2)

    def test_training_stability_completed(self):
        """Teste Training-Stabilität für abgeschlossene Runs."""
        run_id = "completed_run"
        config = {
            "run_id": run_id,
            "model": {"d_model": 512, "num_layers": 6},
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        entry = self._create_test_run_entry(run_id)
        
        # Markiere als completed mit Metriken
        self.registry.complete_run(run_id, {
            "val_bpb": 1.4,
            "ms_per_step": 10.0,
            "steps_completed": 100,
            "artifact_bytes": 10_000_000,
        })
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertEqual(features.training_stability, 1.0)

    def test_training_stability_failed(self):
        """Teste Training-Stabilität für fehlgeschlagene Runs."""
        run_id = "failed_run"
        config = {
            "run_id": run_id,
            "model": {"d_model": 512, "num_layers": 6},
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        # Markiere als failed
        self.registry.fail_run(run_id, "NaN gradients")
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertEqual(features.training_stability, 0.0)

    def test_extract_batch(self):
        """Teste Batch-Extraktion."""
        # Erstelle mehrere Runs
        for i in range(3):
            run_id = f"batch_run_{i}"
            config = {
                "run_id": run_id,
                "model": {"d_model": 512, "num_layers": 6},
                "quant": {"enabled": False},
            }
            self._create_test_config(run_id, config)
            self._create_test_run_entry(run_id)
        
        # Extrahiere Batch
        run_ids = ["batch_run_0", "batch_run_1", "batch_run_2"]
        features_list = self.extractor.extract_batch(run_ids, self.registry)
        
        self.assertEqual(len(features_list), 3)
        self.assertTrue(all(isinstance(f, RunMetaFeatures) for f in features_list))

    def test_extract_batch_with_missing_runs(self):
        """Teste Batch-Extraktion mit fehlenden Runs."""
        # Erstelle nur einen Run
        run_id = "existing_run"
        config = {
            "run_id": run_id,
            "model": {"d_model": 512, "num_layers": 6},
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        # Versuche Batch mit nicht-existierendem Run
        run_ids = ["existing_run", "nonexistent_run"]
        features_list = self.extractor.extract_batch(run_ids, self.registry)
        
        # Sollte nur den existierenden Run zurückgeben
        self.assertEqual(len(features_list), 1)
        self.assertEqual(features_list[0].run_id, "existing_run")

    def test_extract_nonexistent_run(self):
        """Teste Extraktion für nicht-existierenden Run."""
        with self.assertRaises(ValueError):
            self.extractor.extract("nonexistent_run", self.registry)

    def test_extract_missing_config(self):
        """Teste Extraktion wenn Config fehlt."""
        # Erstelle Run ohne Config
        run_id = "no_config_run"
        self._create_test_run_entry(run_id)
        
        # Sollte trotzdem funktionieren mit Default-Werten
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertEqual(features.run_id, run_id)
        self.assertEqual(features.features_active, [])

    def test_compute_co_occurrence(self):
        """Teste Co-occurrence Berechnung."""
        # Erstelle Features mit bekannten Co-occurrences
        features = [
            RunMetaFeatures(run_id="run1", features_active=["a", "b", "c"]),
            RunMetaFeatures(run_id="run2", features_active=["a", "b"]),
            RunMetaFeatures(run_id="run3", features_active=["a", "c"]),
            RunMetaFeatures(run_id="run4", features_active=["b", "c"]),
        ]
        
        co_occ = self.extractor.compute_co_occurrence(features)
        
        # (a, b) sollte 2 mal vorkommen (run1, run2)
        self.assertEqual(co_occ.get(("a", "b"), 0), 2)
        
        # (a, c) sollte 2 mal vorkommen (run1, run3)
        self.assertEqual(co_occ.get(("a", "c"), 0), 2)
        
        # (b, c) sollte 2 mal vorkommen (run1, run4)
        self.assertEqual(co_occ.get(("b", "c"), 0), 2)

    def test_compute_co_occurrence_empty(self):
        """Teste Co-occurrence für leere Liste."""
        co_occ = self.extractor.compute_co_occurrence([])
        
        self.assertEqual(co_occ, {})

    def test_compute_co_occurrence_single_feature(self):
        """Teste Co-occurrence für Runs mit nur einem Feature."""
        features = [
            RunMetaFeatures(run_id="run1", features_active=["a"]),
            RunMetaFeatures(run_id="run2", features_active=["b"]),
        ]
        
        co_occ = self.extractor.compute_co_occurrence(features)
        
        # Keine Paare möglich
        self.assertEqual(co_occ, {})

    def test_enrich_features_with_co_occurrence(self):
        """Teste Anreicherung von Features mit Co-occurrence."""
        features = [
            RunMetaFeatures(run_id="run1", features_active=["a", "b"]),
            RunMetaFeatures(run_id="run2", features_active=["a", "c"]),
        ]
        
        enriched = self.extractor.enrich_features_with_co_occurrence(features)
        
        self.assertEqual(len(enriched), 2)
        
        # run1 hat "a" und "b", beide kommen mit "a" zusammen vor
        self.assertIn("b", enriched[0].co_occurrence_with)
        self.assertIn("a", enriched[0].co_occurrence_with)
        
        # run2 hat "a" und "c", beide kommen mit "a" zusammen vor
        self.assertIn("c", enriched[1].co_occurrence_with)
        self.assertIn("a", enriched[1].co_occurrence_with)


class TestFeatureExtraction(unittest.TestCase):
    """Teste spezifische Feature-Extraktion."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.configs_dir = Path(self.temp_dir.name) / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(self.temp_dir.name) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.registry = RunRegistry(results_dir=str(self.results_dir))
        self.extractor = MetaFeatureExtractor(configs_dir=self.configs_dir)

    def tearDown(self):
        """Räume temporäres Verzeichnis auf."""
        self.temp_dir.cleanup()

    def _create_test_config(self, run_id: str, config: dict) -> None:
        """Erstelle Test-Konfigurationsdatei."""
        import yaml
        config_path = self.configs_dir / "runs"
        config_path.mkdir(parents=True, exist_ok=True)
        file_path = config_path / f"{run_id}.yaml"
        with open(file_path, "w") as f:
            yaml.dump(config, f)

    def _create_test_run_entry(self, run_id: str, **kwargs) -> RunEntry:
        """Erstelle Test-Run-Eintrag."""
        return self.registry.register(
            run_id=run_id,
            config_hash=kwargs.get("config_hash", f"hash_{run_id}"),
            parent_run_id=kwargs.get("parent_run_id"),
        )

    def test_extract_attention_features(self):
        """Teste Extraktion von Attention-Features."""
        run_id = "attention_run"
        config = {
            "run_id": run_id,
            "model": {
                "d_model": 512,
                "num_layers": 6,
                "attention": {
                    "type": "gqa",
                    "kv_heads": 2,
                    "gqa_groups": 4,
                    "kv_sharing": True,
                    "rope": True,
                    "partial_rope": False,
                },
            },
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertIn("gqa", features.features_active)
        self.assertIn("rope", features.features_active)
        self.assertIn("kv_sharing", features.features_active)
        self.assertNotIn("partial_rope", features.features_active)

    def test_extract_recurrence_features(self):
        """Teste Extraktion von Recurrence-Features."""
        run_id = "recurrence_run"
        config = {
            "run_id": run_id,
            "model": {
                "d_model": 512,
                "num_layers": 6,
                "recurrence": {
                    "enabled": True,
                    "type": "tied",
                    "tied": True,
                    "depth": 4,
                    "loop_embeddings": True,
                },
            },
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertIn("recurrence", features.features_active)
        self.assertIn("tied_recurrence", features.features_active)
        self.assertIn("loop_embeddings", features.features_active)

    def test_extract_activation_features(self):
        """Teste Extraktion von Aktivierungs-Features."""
        # Teste LeakyReLU
        run_id = "leaky_relu_run"
        config = {
            "run_id": run_id,
            "model": {
                "d_model": 512,
                "num_layers": 6,
                "activation": "leaky_relu",
            },
            "features": {"leaky_relu": True},
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertIn("leaky_relu", features.features_active)

    def test_extract_feature_gates(self):
        """Teste Extraktion von Feature-Gates."""
        run_id = "gates_run"
        config = {
            "run_id": run_id,
            "model": {
                "d_model": 512,
                "num_layers": 6,
                "xsa": {"enabled": True},
                "film": {"enabled": True},
                "ttt": {"enabled": True},
                "gated_mlp": {"enabled": True},
            },
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertIn("xsa", features.features_active)
        self.assertIn("film", features.features_active)
        self.assertIn("ttt", features.features_active)
        self.assertIn("gated_mlp", features.features_active)

    def test_extract_tokenizer_features(self):
        """Teste Extraktion von Tokenizer-Features."""
        run_id = "tokenizer_run"
        config = {
            "run_id": run_id,
            "model": {"d_model": 512, "num_layers": 6},
            "tokenizer": {
                "type": "bigram_hash",
                "vocab_size": 4096,
                "byte_fallback": True,
            },
            "quant": {"enabled": False},
        }
        self._create_test_config(run_id, config)
        self._create_test_run_entry(run_id)
        
        features = self.extractor.extract(run_id, self.registry)
        
        self.assertIn("bigram_hash", features.features_active)
        self.assertIn("byte_fallback", features.features_active)


class TestIntegrationWithRealRuns(unittest.TestCase):
    """Integrationstests mit echten Runs aus dem Registry."""

    def test_extract_from_real_registry(self):
        """Teste Extraktion aus echtem Registry (wenn vorhanden)."""
        # Verwende echtes Results-Verzeichnis
        results_dir = Path(__file__).parent.parent / "results"
        
        if not results_dir.exists():
            self.skipTest("Results-Verzeichnis existiert nicht")
        
        registry = RunRegistry(results_dir=str(results_dir))
        configs_dir = Path(__file__).parent.parent / "configs"
        
        if not configs_dir.exists():
            self.skipTest("Configs-Verzeichnis existiert nicht")
        
        extractor = MetaFeatureExtractor(configs_dir=configs_dir)
        
        # Hole alle Runs
        runs = registry.list_runs()
        
        if not runs:
            self.skipTest("Keine Runs im Registry")
        
        # Extrahiere Features für alle Runs
        features_list = extractor.extract_batch([r.run_id for r in runs], registry)
        
        # Sollte mindestens ein Feature extrahiert haben
        self.assertGreater(len(features_list), 0)
        
        # Alle Features sollten gültige run_ids haben
        for features in features_list:
            self.assertIsInstance(features.run_id, str)
            self.assertTrue(len(features.run_id) > 0)


if __name__ == "__main__":
    unittest.main()
