"""Intrinsic no-bypass dynamics for ASM-VR Phase 1.

Each step transports and collapses the state before computing token forcing.
Consequently, neither the forcing network nor the emitter can inspect coordinates
removed by the current hard mask. The module stores no recurrent cache or memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .frame import FrameState
from .state import VariableRankState
from .transport import transport_state


@dataclass(frozen=True)
class IntrinsicStepResult:
    """Observable values from one memory-free intrinsic transition."""

    state: VariableRankState
    forcing: Tensor
    output: Tensor


class IntrinsicCollapseDynamics(nn.Module):
    """Fixed-frame hard-gated recurrence with post-collapse forcing.

    ``input_state`` first passes through the hard transport into ``next_mask``.
    Only that collapsed state and ``token_input`` are then available to the
    forcing network. This ordering prevents discarded coordinates from being
    copied into surviving coordinates by token-to-state forcing.
    """

    def __init__(
        self,
        frame_state: FrameState,
        input_dimension: int,
        output_dimension: int,
    ) -> None:
        super().__init__()
        if input_dimension < 1:
            raise ValueError("input_dimension must be positive")
        if output_dimension < 1:
            raise ValueError("output_dimension must be positive")
        if frame_state.basis.is_complex():
            raise TypeError("Phase 1 intrinsic dynamics requires a real frame")
        self.frame_state = frame_state
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        width = frame_state.frame_width
        factory = {"device": frame_state.basis.device, "dtype": frame_state.basis.dtype}
        self.forcing = nn.Linear(width + input_dimension, width, **factory)
        self.emitter = nn.Linear(width, output_dimension, **factory)

    def step(
        self,
        input_state: VariableRankState,
        next_mask: Tensor,
        token_input: Tensor,
    ) -> IntrinsicStepResult:
        """Apply one collapse, post-collapse write, and effective-state emit."""
        self._validate_state(input_state)
        self._validate_token(input_state, token_input)

        collapsed = transport_state(input_state, self.frame_state, next_mask)
        padded = self._pad(collapsed)
        features = torch.cat((padded, token_input), dim=-1)
        padded_forcing = torch.tanh(self.forcing(features))
        compact_forcing = padded_forcing[..., next_mask]
        next_state = VariableRankState(
            effective_coordinates=collapsed.effective_coordinates + compact_forcing,
            active_mask=next_mask,
            rank=collapsed.rank,
            frame_state=self.frame_state,
        )
        effective = self._pad(next_state)
        return IntrinsicStepResult(
            state=next_state,
            forcing=compact_forcing,
            output=self.emitter(effective),
        )

    def _pad(self, state: VariableRankState) -> Tensor:
        padded = state.effective_coordinates.new_zeros(
            *state.effective_coordinates.shape[:-1], self.frame_state.frame_width
        )
        padded[..., state.active_mask] = state.effective_coordinates
        return padded

    def _validate_state(self, state: VariableRankState) -> None:
        if state.frame_state is not self.frame_state:
            raise ValueError("Phase 1 dynamics uses one fixed frame instance")
        if state.transition_memory is not None:
            raise ValueError("Phase 1 dynamics forbids transition memory")

    def _validate_token(self, state: VariableRankState, token_input: Tensor) -> None:
        expected = (*state.effective_coordinates.shape[:-1], self.input_dimension)
        if token_input.shape != expected:
            raise ValueError(f"token_input must have shape {expected}")
        basis = self.frame_state.basis
        if token_input.device != basis.device or token_input.dtype != basis.dtype:
            raise ValueError("token_input and fixed frame must share device and dtype")
        if not bool(torch.isfinite(token_input).all()):
            raise ValueError("token_input must contain only finite values")


__all__ = ["IntrinsicCollapseDynamics", "IntrinsicStepResult"]
