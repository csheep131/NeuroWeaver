"""Research module for ablation studies."""

from .ablation_engine import (
    AblationReporter,
    AblationReport,
    KillRule,
    KillReason,
    create_ablation_reporter,
)

from .phase1_evaluator import (
    Phase1Evaluator,
    Phase1SuccessCriteria,
    Phase1Metrics,
    Phase1Report,
    RunType as Phase1RunType,
    create_phase1_evaluator,
)

from .phase2_evaluator import (
    Phase2Evaluator,
    Phase2SuccessCriteria,
    Phase2Metrics,
    Phase2Report,
    RunType as Phase2RunType,
    create_phase2_evaluator,
)

from .phase3_evaluator import (
    Phase3Evaluator,
    Phase3SuccessCriteria,
    Phase3Metrics,
    Phase3Report,
    RunType as Phase3RunType,
    create_phase3_evaluator,
)

__all__ = [
    # Ablation Engine
    "AblationReporter",
    "AblationReport",
    "KillRule",
    "KillReason",
    "create_ablation_reporter",
    # Phase 1
    "Phase1Evaluator",
    "Phase1SuccessCriteria",
    "Phase1Metrics",
    "Phase1Report",
    "Phase1RunType",
    "create_phase1_evaluator",
    # Phase 2
    "Phase2Evaluator",
    "Phase2SuccessCriteria",
    "Phase2Metrics",
    "Phase2Report",
    "Phase2RunType",
    "create_phase2_evaluator",
    # Phase 3
    "Phase3Evaluator",
    "Phase3SuccessCriteria",
    "Phase3Metrics",
    "Phase3Report",
    "Phase3RunType",
    "create_phase3_evaluator",
]
