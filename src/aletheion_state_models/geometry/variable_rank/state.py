"""Canonical state container for variable-rank geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .frame import FrameState


@dataclass(frozen=True, eq=False)
class VariableRankState:
    """Store only usable coordinates and structural rank metadata.

    The last coordinate dimension is compact and equals ``rank``. In
    particular, this type has no ambient or ``full_state`` field: an ambient
    representation may be reconstructed only as an ephemeral tensor.

    ``transition_memory`` is reserved for a future typed structural-memory
    contract. Phase 0 accepts only ``None`` so no untyped payload can become an
    accidental information bypass.
    """

    effective_coordinates: torch.Tensor
    active_mask: torch.Tensor
    rank: int
    frame_state: FrameState
    transition_memory: None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame_state, FrameState):
            raise TypeError("frame_state must be a FrameState")
        if not isinstance(self.effective_coordinates, torch.Tensor):
            raise TypeError("effective_coordinates must be a torch.Tensor")
        if self.effective_coordinates.ndim < 1:
            raise ValueError("effective_coordinates must have at least one dimension")
        if not (
            self.effective_coordinates.is_floating_point()
            or self.effective_coordinates.is_complex()
        ):
            raise TypeError("effective_coordinates must use a floating-point or complex dtype")
        if not torch.isfinite(self.effective_coordinates).all().item():
            raise ValueError("effective_coordinates must contain only finite values")

        if not isinstance(self.active_mask, torch.Tensor):
            raise TypeError("active_mask must be a torch.Tensor")
        if self.active_mask.ndim != 1:
            raise ValueError("active_mask must have shape (frame_width,)")
        if self.active_mask.dtype is not torch.bool:
            raise TypeError("active_mask must have dtype torch.bool")
        if self.active_mask.shape[0] != self.frame_state.frame_width:
            raise ValueError("active_mask length must equal frame_state.frame_width")

        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an int")
        active_count = int(torch.count_nonzero(self.active_mask).item())
        if self.rank != active_count:
            raise ValueError("rank must equal the number of active mask entries")
        if self.effective_coordinates.shape[-1] != self.rank:
            raise ValueError("the last effective-coordinate dimension must equal rank")

        basis = self.frame_state.basis
        if self.effective_coordinates.device != basis.device:
            raise ValueError("effective_coordinates and frame basis must share a device")
        if self.active_mask.device != basis.device:
            raise ValueError("active_mask and frame basis must share a device")
        if self.effective_coordinates.dtype != basis.dtype:
            raise ValueError("effective_coordinates and frame basis must share a dtype")
        if self.transition_memory is not None:
            raise ValueError("Phase 0 transition_memory must be None")


__all__ = ["VariableRankState"]
