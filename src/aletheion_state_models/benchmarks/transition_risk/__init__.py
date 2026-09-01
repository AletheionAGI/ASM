"""CPU-only primitives for ATTR transition-risk prediction and intervention."""

from .baselines import (
    ControlResults,
    KalmanBaseline,
    MarkovRiskBaseline,
    evaluate_controls,
)
from .dataset import HazardEpisode, collate_episodes, make_episodes, make_worlds
from .model_adapters import ASMModelAdapter, ModelAdapter, TransformerModelAdapter
from .p2_seal import (
    MODEL_ARMS,
    TRAINING_SEEDS,
    P2DatasetSeal,
    P2SplitSpec,
    create_p2_seal,
    default_p2_specs,
    open_p2_seal,
    read_p2_seal,
    write_p2_seal,
)
from .model_heads import HazardHead, NextStateHead, SeverityHead, TransitionRiskHeads
from .intervention import (
    BranchRollout,
    ClonedIntervention,
    clone_simulator,
    cloned_intervention,
    rollout_branch,
    run_cloned_intervention,
)
from .labels import (
    build_multi_horizon_labels,
    labels_by_horizon,
    multi_horizon_labels,
    unsafe_entry_steps,
    validate_horizons,
)
from .leakage import (
    FeatureAvailability,
    LeakageAudit,
    LeakageViolation,
    audit_episode_splits,
    audit_feature_availability,
    audit_leakage,
)
from .metrics import (
    BasicRiskMetrics,
    auprc,
    average_precision,
    basic_risk_metrics,
    brier_score,
    first_sustained_alarm,
    recall_at_false_positive_rate,
    useful_lead_time,
)
from .p2_statistics import (
    aggregate_by_horizon,
    evaluate_g2_g5,
    paired_hierarchical_bootstrap,
)
from .shield import HardShield, apply_hard_shield, evaluate_candidates
from .types import DEFAULT_HORIZONS, HazardPrediction, HorizonLabels, ShieldDecision

__all__ = [
    "ASMModelAdapter",
    "ControlResults",
    "HazardEpisode",
    "KalmanBaseline",
    "MarkovRiskBaseline",
    "collate_episodes",
    "evaluate_controls",
    "make_episodes",
    "make_worlds",
    "MODEL_ARMS",
    "TRAINING_SEEDS",
    "P2DatasetSeal",
    "P2SplitSpec",
    "create_p2_seal",
    "default_p2_specs",
    "open_p2_seal",
    "read_p2_seal",
    "write_p2_seal",
    "HazardHead",
    "ModelAdapter",
    "NextStateHead",
    "SeverityHead",
    "TransformerModelAdapter",
    "TransitionRiskHeads",
    "DEFAULT_HORIZONS",
    "BasicRiskMetrics",
    "BranchRollout",
    "ClonedIntervention",
    "FeatureAvailability",
    "HardShield",
    "HazardPrediction",
    "HorizonLabels",
    "LeakageAudit",
    "LeakageViolation",
    "ShieldDecision",
    "aggregate_by_horizon",
    "apply_hard_shield",
    "audit_episode_splits",
    "audit_feature_availability",
    "audit_leakage",
    "auprc",
    "average_precision",
    "basic_risk_metrics",
    "brier_score",
    "build_multi_horizon_labels",
    "clone_simulator",
    "cloned_intervention",
    "evaluate_candidates",
    "evaluate_g2_g5",
    "first_sustained_alarm",
    "labels_by_horizon",
    "multi_horizon_labels",
    "paired_hierarchical_bootstrap",
    "recall_at_false_positive_rate",
    "rollout_branch",
    "run_cloned_intervention",
    "unsafe_entry_steps",
    "useful_lead_time",
    "validate_horizons",
]
