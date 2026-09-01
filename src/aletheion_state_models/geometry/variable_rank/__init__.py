"""Variable-rank state, projection, transport, and diagnostics for ASM-VR."""

from .batch_state import VariableRankBatchState
from .block_core import VariableRankBlockCore
from .diagnostics import CycleDiagnostics, diagnose_cycle, jacobian_operator, numerical_rank
from .experiments import run_phase0_experiments
from .frame import FrameState, LearnedOrthonormalFrame
from .information_probe import InformationProbeResult, linear_information_probe
from .intrinsic_dynamics import IntrinsicCollapseDynamics, IntrinsicStepResult
from .phase1_experiments import run_phase1_experiments
from .rank_controller import HardRankObservation, InputHardRankController
from .rank_curriculum import RankCurriculumState, phase2_rank_curriculum
from .rank_objectives import RankLosses, rank_regularization
from .projector import (
    hard_project,
    hard_projector_matrix,
    project_effective_coordinates,
    reconstruct_ambient_state,
    soft_access_filter,
)
from .state import VariableRankState
from .transport import TransportResult, padded_transport_operator, transport_state

__all__ = [
    "CycleDiagnostics",
    "HardRankObservation",
    "FrameState",
    "InformationProbeResult",
    "InputHardRankController",
    "IntrinsicCollapseDynamics",
    "IntrinsicStepResult",
    "LearnedOrthonormalFrame",
    "RankCurriculumState",
    "RankLosses",
    "TransportResult",
    "VariableRankBatchState",
    "VariableRankBlockCore",
    "VariableRankState",
    "diagnose_cycle",
    "hard_project",
    "hard_projector_matrix",
    "jacobian_operator",
    "linear_information_probe",
    "numerical_rank",
    "phase2_rank_curriculum",
    "padded_transport_operator",
    "project_effective_coordinates",
    "rank_regularization",
    "reconstruct_ambient_state",
    "run_phase0_experiments",
    "run_phase1_experiments",
    "soft_access_filter",
    "transport_state",
]
