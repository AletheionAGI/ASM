"""Immutable data contracts for the frozen ATTR-RTG dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from world_model.hazard_world_types import (
    ACTIONS,
    HazardTransition,
    HazardWorldConfig,
    HazardWorldState,
)

Frame: TypeAlias = tuple[int, int, int, int]
YCommon: TypeAlias = tuple[int, ...]
NONSTOP_ACTIONS = tuple(action for action in ACTIONS if action != "STOP")
Y_COMMON_CARDINALITIES = (81, 81, 81, 81, 81, 3, 3, 64, 4, 4, 2)
Y_COMMON_SLICES = {
    "trap_1": slice(0, 81),
    "trap_2": slice(81, 162),
    "trap_3": slice(162, 243),
    "agent": slice(243, 324),
    "moving_hazard": slice(324, 405),
    "velocity_row": slice(405, 408),
    "velocity_col": slice(408, 411),
    "energy": slice(411, 475),
    "low_energy_steps": slice(475, 479),
    "recovery_left": slice(479, 483),
    "safe_terminal": slice(483, 485),
}


@dataclass(frozen=True)
class CandidateInput:
    """Causal candidate data; this is the only predictive candidate view."""

    frame: Frame
    fixed_frame: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.frame) != 4 or any(
            type(value) is not int or not 0 <= value < 256 for value in self.frame
        ):
            raise ValueError("candidate frame must contain exactly four bytes")
        if self.fixed_frame.shape != (32,) or self.fixed_frame.dtype != torch.float32:
            raise ValueError("fixed candidate frame must be float32 with shape (32,)")


@dataclass(frozen=True)
class CandidateTruth:
    """Privileged physical consequence kept separate from candidate inputs."""

    target: YCommon
    unsafe: bool
    transition: HazardTransition


@dataclass(frozen=True)
class OriginMetadata:
    """Audit-only identity fields that must never enter a model input."""

    split_id: str
    world_id: str
    episode_id: str
    t: int


@dataclass(frozen=True)
class OriginInput:
    """Causal history and six candidate frames for one pre-action origin."""

    history: tuple[int, ...]
    candidates: tuple[CandidateInput, ...]

    def __post_init__(self) -> None:
        if not self.history or len(self.history) % 4:
            raise ValueError("origin history must contain complete prior frames")
        if len(self.candidates) != len(NONSTOP_ACTIONS):
            raise ValueError("each origin must contain exactly six candidates")
        reference = self.candidates[0].frame
        for index, candidate in enumerate(self.candidates):
            frame = candidate.frame
            same_observation = (
                frame[0] == reference[0]
                and frame[1] // len(ACTIONS) == reference[1] // len(ACTIONS)
                and frame[2:] == reference[2:]
            )
            expected_action = ACTIONS.index(NONSTOP_ACTIONS[index])
            if not same_observation or frame[1] % len(ACTIONS) != expected_action:
                raise ValueError(
                    "candidate frames violate common observation or action order"
                )


@dataclass(frozen=True)
class PhysicalSnapshot:
    """Audit-only simulator snapshot retained without executing a branch."""

    config: HazardWorldConfig
    state: HazardWorldState


@dataclass(frozen=True)
class PreparedRtgOrigin:
    """Pre-export origin: causal input plus an unmaterialized physical snapshot."""

    metadata: OriginMetadata
    inputs: OriginInput
    snapshot: PhysicalSnapshot

    def __post_init__(self) -> None:
        if (
            self.metadata.t < 1
            or self.snapshot.state.terminal
            or self.snapshot.state.unsafe
        ):
            raise ValueError("prepared origin must be safe, pre-terminal, and t >= 1")
        if self.snapshot.state.step != self.metadata.t:
            raise ValueError("prepared snapshot step must equal origin t")
        if self.metadata.world_id != self.snapshot.config.world_id:
            raise ValueError(
                "prepared origin world identity does not match its snapshot"
            )


@dataclass(frozen=True)
class OriginTruth:
    """Privileged pre-state persistence and six aligned cloned consequences."""

    persistence_target: YCommon
    candidates: tuple[CandidateTruth, ...]
    failure_delay: int

    def __post_init__(self) -> None:
        if len(self.persistence_target) != len(Y_COMMON_CARDINALITIES):
            raise ValueError("physical persistence target must contain eleven groups")
        if len(self.candidates) != len(NONSTOP_ACTIONS):
            raise ValueError("each origin must contain six candidate truths")
        if self.failure_delay not in {1, 3}:
            raise ValueError("origin truth failure_delay must be 1 or 3")


@dataclass(frozen=True)
class RtgOrigin:
    """Audit record with explicit metadata/input/ground-truth boundaries."""

    metadata: OriginMetadata
    inputs: OriginInput
    truth: OriginTruth
