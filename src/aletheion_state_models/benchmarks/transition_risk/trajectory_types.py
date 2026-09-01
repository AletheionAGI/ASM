"""Value types for planned physical-trajectory supervision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch

PLAN_HORIZON = 8
TRAJECTORY_HORIZON = PLAN_HORIZON
TRAINING_SEEDS = (29, 43, 71, 89, 107)
TARGET_CARDINALITIES: Mapping[str, int] = {
    "agent_cell": 81,
    "moving_hazard_cell": 81,
    "velocity_row": 3,
    "velocity_col": 3,
    "energy_bin": 64,
    "low_energy_steps": 4,
    "recovery_left": 4,
    "hidden_mode": 3,
    "safe_terminal": 2,
}


@dataclass(frozen=True)
class TrajectoryTargets:
    """Categorical physical state targets, each shaped ``[T, 8]``."""

    agent_cell: torch.Tensor
    moving_hazard_cell: torch.Tensor
    velocity_row: torch.Tensor
    velocity_col: torch.Tensor
    energy_bin: torch.Tensor
    low_energy_steps: torch.Tensor
    recovery_left: torch.Tensor
    hidden_mode: torch.Tensor
    safe_terminal: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def validate(self, shape: tuple[int, int]) -> None:
        for name, cardinality in TARGET_CARDINALITIES.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != torch.long:
                raise ValueError(f"target {name} must be long with shape {shape}")
            if value.numel() and (value.min() < 0 or value.max() >= cardinality):
                raise ValueError(f"target {name} is outside its categorical range")


@dataclass(frozen=True)
class TrajectoryEpisode:
    """One causal behavior trace with cloned eight-action branches per origin."""

    episode_id: str
    world_id: str
    input_ids: torch.Tensor
    step_positions: torch.Tensor
    plan_actions: torch.Tensor
    trap_cells: torch.Tensor
    targets: TrajectoryTargets
    valid_mask: torch.Tensor
    unsafe_truth: torch.Tensor
    behavior_actions: tuple[str, ...]
    failure_delay: int

    def __post_init__(self) -> None:
        steps = self.step_positions.numel()
        branch_shape = (steps, PLAN_HORIZON)
        if self.input_ids.ndim != 1 or self.step_positions.ndim != 1:
            raise ValueError("input_ids and step_positions must be one-dimensional")
        if (
            self.plan_actions.shape != branch_shape
            or self.plan_actions.dtype != torch.long
        ):
            raise ValueError(f"plan_actions must be long with shape {branch_shape}")
        if self.trap_cells.shape != (steps, 3) or self.trap_cells.dtype != torch.long:
            raise ValueError(f"trap_cells must be long with shape {(steps, 3)}")
        if self.valid_mask.shape != branch_shape or self.valid_mask.dtype != torch.bool:
            raise ValueError(f"valid_mask must be bool with shape {branch_shape}")
        if (
            self.unsafe_truth.shape != branch_shape
            or self.unsafe_truth.dtype != torch.bool
        ):
            raise ValueError(f"unsafe_truth must be bool with shape {branch_shape}")
        if len(self.behavior_actions) != steps:
            raise ValueError("behavior actions must align with origins")
        if type(self.failure_delay) is not int or not 1 <= self.failure_delay <= 3:
            raise ValueError("failure_delay must be an integer in [1, 3]")
        self.targets.validate(branch_shape)
