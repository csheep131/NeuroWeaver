"""Orchestrator module for run automation."""

from .sweep import SweepRunner, SweepConfig, SweepParameter, create_sweep
from .promote import PromotionSystem, Stage, StageConfig, create_promotion_system
from .submit_bundle import SubmissionBuilder, SubmissionBundle, create_submission_bundle
from .dashboard import main as dashboard_main
from .multi_seed import MultiSeedOrchestrator, MultiSeedConfig, create_multi_seed_orchestrator
from .combo_builder import DynamicComboBuilder, ComboConfig, create_combo_builder, generate_phase3_combos

__all__ = [
    # Sweep
    "SweepRunner",
    "SweepConfig",
    "SweepParameter",
    "create_sweep",
    # Promotion
    "PromotionSystem",
    "Stage",
    "StageConfig",
    "create_promotion_system",
    # Submission
    "SubmissionBuilder",
    "SubmissionBundle",
    "create_submission_bundle",
    # Dashboard
    "dashboard_main",
    # Multi-Seed
    "MultiSeedOrchestrator",
    "MultiSeedConfig",
    "create_multi_seed_orchestrator",
    # Combo Builder (Phase 3)
    "DynamicComboBuilder",
    "ComboConfig",
    "create_combo_builder",
    "generate_phase3_combos",
]
