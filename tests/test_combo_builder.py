"""Unit tests for orchestrator/combo_builder.py."""

import unittest
from unittest.mock import Mock, patch
import tempfile
import yaml
from pathlib import Path

from orchestrator.combo_builder import (
    DynamicComboBuilder,
    GateStatus,
    FeatureCandidate,
    ComboConfig,
    create_combo_builder,
    generate_phase3_combos,
)
from core.registry import RunEntry


class TestDynamicComboBuilder(unittest.TestCase):
    """Test the DynamicComboBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_registry = Mock()
        self.builder = DynamicComboBuilder(self.mock_registry)

    def test_infer_gate_status_pending(self):
        """Test gate inference for pending runs."""
        entry = RunEntry(
            run_id="test_run",
            config_hash="hash",
            seed=42,
            status="running",
            val_bpb=None,
            delta_bpb=None,
            ms_per_step=None,
            delta_ms=None,
            artifact_bytes=0,
            notes=None,
            parent_run_id=None,
            start_time=None,
            end_time=None,
            git_commit=None,
        )
        status = self.builder._infer_gate_status(entry)
        self.assertEqual(status, GateStatus.PENDING)

    def test_infer_gate_status_killed(self):
        """Test gate inference for killed runs."""
        entry = RunEntry(
            run_id="test_run",
            config_hash="hash",
            seed=42,
            status="killed",
            val_bpb=1.5,
            delta_bpb=0.0,
            ms_per_step=10.0,
            delta_ms=0.0,
            artifact_bytes=10_000_000,
            notes=None,
            parent_run_id=None,
            start_time=None,
            end_time=None,
            git_commit=None,
        )
        status = self.builder._infer_gate_status(entry)
        self.assertEqual(status, GateStatus.FAIL)

    def test_infer_gate_status_pass_from_notes(self):
        """Test gate inference from notes field."""
        entry = RunEntry(
            run_id="test_run",
            config_hash="hash",
            seed=42,
            status="completed",
            val_bpb=1.4,
            delta_bpb=-0.1,
            ms_per_step=10.0,
            delta_ms=0.0,
            artifact_bytes=10_000_000,
            notes="Feature passed all tests - PASS",
            parent_run_id=None,
            start_time=None,
            end_time=None,
            git_commit=None,
        )
        status = self.builder._infer_gate_status(entry)
        self.assertEqual(status, GateStatus.PASS)

    def test_infer_gate_status_watch_from_metrics(self):
        """Test gate inference from metrics."""
        entry = RunEntry(
            run_id="test_run",
            config_hash="hash",
            seed=42,
            status="completed",
            val_bpb=1.5,
            delta_bpb=-0.02,  # Within WATCH range
            ms_per_step=10.0,
            delta_ms=0.0,
            artifact_bytes=10_000_000,
            notes=None,
            parent_run_id=None,
            start_time=None,
            end_time=None,
            git_commit=None,
        )
        status = self.builder._infer_gate_status(entry)
        self.assertEqual(status, GateStatus.WATCH)

    def test_infer_gate_status_fail_from_metrics(self):
        """Test gate inference for failing metrics."""
        entry = RunEntry(
            run_id="test_run",
            config_hash="hash",
            seed=42,
            status="completed",
            val_bpb=1.6,
            delta_bpb=0.06,  # Above FAIL threshold
            ms_per_step=10.0,
            delta_ms=0.0,
            artifact_bytes=10_000_000,
            notes=None,
            parent_run_id=None,
            start_time=None,
            end_time=None,
            git_commit=None,
        )
        status = self.builder._infer_gate_status(entry)
        self.assertEqual(status, GateStatus.FAIL)

    def test_infer_gate_status_null_safety(self):
        """Test gate inference with null entry."""
        status = self.builder._infer_gate_status(None)
        self.assertEqual(status, GateStatus.PENDING)

    def test_compute_priority_score_improvement(self):
        """Test priority score calculation for improvements."""
        entry = RunEntry(
            run_id="test_run",
            config_hash="hash",
            seed=42,
            status="completed",
            val_bpb=1.4,
            delta_bpb=-0.05,  # Improvement
            ms_per_step=8.0,
            delta_ms=-2.0,  # Speed improvement
            artifact_bytes=8_000_000,  # Smaller artifact
            notes=None,
            parent_run_id=None,
            start_time=None,
            end_time=None,
            git_commit=None,
        )
        score = self.builder._compute_priority_score(entry)
        self.assertGreater(score, 0)

    def test_compute_priority_score_null_safety(self):
        """Test priority score calculation with null entry."""
        score = self.builder._compute_priority_score(None)
        self.assertEqual(score, 0.0)

    def test_feature_candidate_is_combinable(self):
        """Test feature candidate combinable check."""
        pass_candidate = FeatureCandidate(
            feature_name="test",
            run_id="run001",
            gate_status=GateStatus.PASS,
        )
        self.assertTrue(pass_candidate.is_combinable())

        watch_candidate = FeatureCandidate(
            feature_name="test",
            run_id="run001",
            gate_status=GateStatus.WATCH,
        )
        self.assertFalse(watch_candidate.is_combinable())

    @patch.object(DynamicComboBuilder, 'analyze_phase1_results')
    @patch.object(DynamicComboBuilder, 'analyze_phase2_results')
    def test_check_gate_freeze_success(self, mock_phase2, mock_phase1):
        """Test gate freeze check when all runs completed."""
        # Mock registry to return completed entries
        def mock_get(run_id):
            entry = Mock()
            entry.status = "completed"
            return entry
        
        self.mock_registry.get = mock_get
        
        ready, blocking = self.builder.check_gate_freeze()
        self.assertTrue(ready)
        self.assertEqual(len(blocking), 0)

    @patch.object(DynamicComboBuilder, 'analyze_phase1_results')
    @patch.object(DynamicComboBuilder, 'analyze_phase2_results')
    def test_check_gate_freeze_blocked(self, mock_phase2, mock_phase1):
        """Test gate freeze check when runs are pending."""
        # Mock registry to return None for some runs
        def mock_get(run_id):
            if run_id == "run001_control":
                return None
            entry = Mock()
            entry.status = "completed"
            return entry
        
        self.mock_registry.get = mock_get
        
        ready, blocking = self.builder.check_gate_freeze()
        self.assertFalse(ready)
        self.assertGreater(len(blocking), 0)

    def test_combo_config_save(self):
        """Test saving combo config to YAML."""
        config = ComboConfig(
            combo_id="run016_best_combo_a",
            parent_run_id="run001_control",
            seed=42,
            tokenizer_type="byte",
            attention_type="gqa",
            activation="gelu",
            selected_features=[],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_config.yaml"
            config.save(output_path)
            
            # Verify file exists
            self.assertTrue(output_path.exists())
            
            # Verify YAML content
            with open(output_path, 'r') as f:
                content = yaml.safe_load(f)
            
            self.assertEqual(content['run_id'], "run016_best_combo_a")
            self.assertEqual(content['parent_run_id'], "run001_control")
            self.assertEqual(content['seed'], 42)
            self.assertEqual(content['model']['activation'], "gelu")

    def test_generate_combo_config(self):
        """Test generating combo configuration."""
        # Mock all selection methods to return candidates
        with patch.object(self.builder, 'select_best_tokenizer') as mock_tokenizer, \
             patch.object(self.builder, 'select_best_activation') as mock_activation, \
             patch.object(self.builder, 'select_best_attention') as mock_attention:
            
            mock_tokenizer.return_value = FeatureCandidate(
                feature_name="tokenizer_bigram_4k",
                run_id="run002a_bigram_4k",
                gate_status=GateStatus.PASS,
                priority_score=10.0,
            )
            mock_activation.return_value = FeatureCandidate(
                feature_name="leaky_relu",
                run_id="run004_leakyrelu",
                gate_status=GateStatus.PASS,
                priority_score=8.0,
            )
            mock_attention.return_value = FeatureCandidate(
                feature_name="gqa",
                run_id="run009_gqa",
                gate_status=GateStatus.PASS,
                priority_score=12.0,
            )
            
            # Mock gate freeze to pass
            with patch.object(self.builder, 'check_gate_freeze', return_value=(True, [])):
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = self.builder.generate_combo_run(
                        is_quantized=False,
                        output_dir=tmpdir,
                    )
                    
                    # Should generate config
                    self.assertIsNotNone(result)
                    self.assertEqual(result.combo_id, "run016_best_combo_a")


class TestGateStatusEnum(unittest.TestCase):
    """Test GateStatus enum."""
    
    def test_enum_values(self):
        """Test enum value mapping."""
        self.assertEqual(GateStatus.PASS.value, "pass")
        self.assertEqual(GateStatus.WATCH.value, "watch")
        self.assertEqual(GateStatus.FAIL.value, "fail")
        self.assertEqual(GateStatus.PENDING.value, "pending")


class TestIntegration(unittest.TestCase):
    """Integration tests for combo builder."""
    
    def test_create_combo_builder(self):
        """Test factory function."""
        builder = create_combo_builder()
        self.assertIsInstance(builder, DynamicComboBuilder)
    
    def test_generate_phase3_combos_gate_frozen(self):
        """Test generate_phase3_combos with gate freeze."""
        with patch('orchestrator.combo_builder.RunRegistry') as mock_registry_class, \
             patch('orchestrator.combo_builder.create_combo_builder') as mock_factory:
            
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            
            mock_builder = Mock()
            mock_factory.return_value = mock_builder
            
            # Mock gate freeze to fail
            mock_builder.check_gate_freeze.return_value = (False, ["run001_control pending"])
            
            best_combo, quant_combo = generate_phase3_combos(force=False)
            
            # Should return None when gate not frozen
            self.assertIsNone(best_combo)
            self.assertIsNone(quant_combo)
            
            # Verify builder methods called
            mock_builder.check_gate_freeze.assert_called()


if __name__ == '__main__':
    unittest.main()