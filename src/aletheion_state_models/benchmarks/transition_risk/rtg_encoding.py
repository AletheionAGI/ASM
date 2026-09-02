"""Frozen candidate-frame encodings for ATTR-RTG."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from world_model.hazard_world_types import HazardObservation

from .dataset import encode_frame
from .rtg_types import NONSTOP_ACTIONS, CandidateInput, Frame


def fixed_encode(frame: Sequence[int]) -> torch.Tensor:
    """Expand four bytes to LSB-first signed bits without learned parameters."""
    if len(frame) != 4 or any(
        type(value) is not int or not 0 <= value < 256 for value in frame
    ):
        raise ValueError("fixed_encode requires exactly four byte integers")
    values = [1.0 if byte & (1 << bit) else -1.0 for byte in frame for bit in range(8)]
    return torch.tensor(values, dtype=torch.float32)


def encode_candidate(
    observation: HazardObservation,
    action: str,
    *,
    grid_size: int = 9,
) -> CandidateInput:
    """Materialize one causal frame and its fixed, parameter-free encoding."""
    if action not in NONSTOP_ACTIONS:
        raise ValueError("ATTR-RTG candidates must be non-STOP actions")
    frame: Frame = encode_frame(observation, action, grid_size)
    return CandidateInput(frame=frame, fixed_frame=fixed_encode(frame))


def encode_all_candidates(
    observation: HazardObservation, *, grid_size: int = 9
) -> tuple[CandidateInput, ...]:
    """Encode all six candidates in the preregistered stable order."""
    return tuple(
        encode_candidate(observation, action, grid_size=grid_size)
        for action in NONSTOP_ACTIONS
    )
