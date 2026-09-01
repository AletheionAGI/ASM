"""Batched effective-state contract for integrated ASM-VR block recurrence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class VariableRankBatchState:
    """Persist padded effective coordinates and a per-example hard mask.

    Inactive coordinates must be exactly zero. The padded layout permits mixed
    ranks in one batch without retaining discarded values.
    """

    effective_coordinates: Tensor
    active_mask: Tensor

    def __post_init__(self) -> None:
        coordinates = self.effective_coordinates
        mask = self.active_mask
        if not isinstance(coordinates, Tensor) or coordinates.ndim != 2:
            raise TypeError("effective_coordinates must be a rank-2 tensor")
        if not (coordinates.is_floating_point() or coordinates.is_complex()):
            raise TypeError("effective_coordinates must use a floating or complex dtype")
        if not isinstance(mask, Tensor) or mask.ndim != 2:
            raise TypeError("active_mask must be a rank-2 tensor")
        if mask.dtype is not torch.bool:
            raise TypeError("active_mask must have dtype torch.bool")
        if coordinates.shape != mask.shape:
            raise ValueError("effective_coordinates and active_mask must share shape")
        if coordinates.device != mask.device:
            raise ValueError("effective_coordinates and active_mask must share device")
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("effective_coordinates must contain only finite values")
        if bool(torch.any(coordinates.masked_select(~mask) != 0)):
            raise ValueError("inactive effective coordinates must be exactly zero")

    @property
    def ranks(self) -> Tensor:
        """Return the hard rank of each batch example."""
        return self.active_mask.sum(dim=-1)

    def detach(self) -> "VariableRankBatchState":
        """Detach tensors while preserving the validated state contract."""
        return VariableRankBatchState(
            self.effective_coordinates.detach(),
            self.active_mask.detach(),
        )


__all__ = ["VariableRankBatchState"]
