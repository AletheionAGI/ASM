"""HazardWorld construction and causal-input adapters."""

from __future__ import annotations

from collections.abc import Sequence

from world_model.hazard_world import HazardWorld
from world_model.hazard_world_types import HazardWorldConfig, HazardWorldState

from .constants import CANDIDATES
from .data_contracts import ModelProcessInput


def hazard_world(
    config: HazardWorldConfig, state: HazardWorldState | None = None
) -> HazardWorld:
    """Build the protocol clone source without executing a transition."""
    return HazardWorld(config, state)


def candidate_frames(
    frame_for_action: dict[str, Sequence[int]],
) -> tuple[tuple[int, int, int, int], ...]:
    """Put externally encoded four-byte frames on the frozen candidate axis."""
    if set(frame_for_action) != set(CANDIDATES):
        raise ValueError("frames must cover exactly the six registered candidates")
    return tuple(tuple(frame_for_action[action]) for action in CANDIDATES)  # type: ignore[return-value]


def model_process_input(
    histories: Sequence[Sequence[int]],
    candidate_rows: Sequence[Sequence[Sequence[int]]],
    masks: Sequence[Sequence[bool]],
    logical_lengths: Sequence[int],
) -> ModelProcessInput:
    """Copy causal values into the exact four-field process contract."""
    return ModelProcessInput(
        tuple(tuple(row) for row in histories),
        tuple(tuple(tuple(frame) for frame in row) for row in candidate_rows),  # type: ignore[arg-type]
        tuple(tuple(row) for row in masks),
        tuple(logical_lengths),
    )
