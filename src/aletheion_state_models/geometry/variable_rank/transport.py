"""Explicit padded transport for ASM-VR effective coordinates.

The routines in this module never reconstruct or retain an ambient ``full_state``.
Transport and token forcing are separate inputs so newly enabled coordinates cannot
be mistaken for recovered information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .frame import FrameState
    from .state import VariableRankState


@dataclass(frozen=True)
class TransportResult:
    """Values produced by one transport step in padded coordinates."""

    transported: Tensor
    forcing: Tensor
    coordinates: Tensor
    operator: Tensor


def padded_transport_operator(
    old_basis: Tensor,
    new_basis: Tensor,
    old_mask: Tensor,
    new_mask: Tensor,
) -> Tensor:
    """Build ``J = diag(m_new) Q_new^T Q_old diag(m_old)``.

    Bases have shape ``(..., ambient_dim, max_rank)`` and masks have shape
    ``(..., max_rank)``. Leading dimensions follow PyTorch broadcasting rules.
    The returned operator maps padded old coordinates to padded new coordinates.
    """
    _validate_basis_pair(old_basis, new_basis)
    _validate_mask(old_mask, old_basis, "old_mask")
    _validate_mask(new_mask, new_basis, "new_mask")

    overlap = new_basis.mH @ old_basis
    old_diagonal = torch.diag_embed(old_mask.to(dtype=overlap.dtype))
    new_diagonal = torch.diag_embed(new_mask.to(dtype=overlap.dtype))
    return new_diagonal @ overlap @ old_diagonal


def apply_transport(
    coordinates: Tensor,
    operator: Tensor,
    forcing: Tensor | None = None,
) -> TransportResult:
    """Apply a padded operator, then add explicitly supplied forcing.

    ``forcing=None`` means zero forcing. The result exposes transported and forced
    terms independently for diagnostics. No ambient state is created or cached.
    """
    if coordinates.shape[-1] != operator.shape[-1]:
        raise ValueError("coordinates and operator input dimensions do not match")
    transported = torch.matmul(operator, coordinates.unsqueeze(-1)).squeeze(-1)
    if forcing is None:
        applied_forcing = torch.zeros_like(transported)
    else:
        if forcing.shape != transported.shape:
            raise ValueError("forcing must have the transported coordinate shape")
        applied_forcing = forcing
    return TransportResult(
        transported=transported,
        forcing=applied_forcing,
        coordinates=transported + applied_forcing,
        operator=operator,
    )


def transport_coordinates(
    coordinates: Tensor,
    old_basis: Tensor,
    new_basis: Tensor,
    old_mask: Tensor,
    new_mask: Tensor,
    forcing: Tensor | None = None,
) -> TransportResult:
    """Build and apply the explicit ASM-VR padded transport in one call."""
    operator = padded_transport_operator(old_basis, new_basis, old_mask, new_mask)
    return apply_transport(coordinates, operator, forcing)


def transport_state(
    state: "VariableRankState",
    new_frame: "FrameState",
    new_mask: Tensor,
    forcing: Tensor | None = None,
) -> "VariableRankState":
    """Transport a compact state without constructing an ambient full state.

    ``forcing`` uses the compact new active-coordinate layout. Temporary padded
    coordinate tensors make the mask factors explicit, but are not persisted.
    Structural transition memory is passed through unchanged.
    """
    from .state import VariableRankState

    if new_mask.ndim != 1 or new_mask.dtype is not torch.bool:
        raise TypeError("new_mask must be a one-dimensional boolean tensor")
    new_rank = int(torch.count_nonzero(new_mask).item())
    expected_forcing_shape = (*state.effective_coordinates.shape[:-1], new_rank)
    if forcing is not None and forcing.shape != expected_forcing_shape:
        raise ValueError("forcing must use the compact new active-coordinate shape")

    old_padded = state.effective_coordinates.new_zeros(
        *state.effective_coordinates.shape[:-1], state.frame_state.frame_width
    )
    old_padded[..., state.active_mask] = state.effective_coordinates
    new_forcing = old_padded.new_zeros(*old_padded.shape[:-1], new_frame.frame_width)
    if forcing is not None:
        new_forcing[..., new_mask] = forcing
    operator = padded_transport_operator(
        state.frame_state.basis,
        new_frame.basis,
        state.active_mask,
        new_mask,
    )
    result = apply_transport(old_padded, operator, new_forcing)
    return VariableRankState(
        effective_coordinates=result.coordinates[..., new_mask],
        active_mask=new_mask,
        rank=new_rank,
        frame_state=new_frame,
        transition_memory=state.transition_memory,
    )


def _validate_basis_pair(old_basis: Tensor, new_basis: Tensor) -> None:
    if old_basis.ndim < 2 or new_basis.ndim < 2:
        raise ValueError("bases must be matrices or batches of matrices")
    if old_basis.shape[-2] != new_basis.shape[-2]:
        raise ValueError("old and new bases must share the ambient dimension")
    if old_basis.device != new_basis.device or old_basis.dtype != new_basis.dtype:
        raise ValueError("old and new bases must share device and dtype")
    if not bool(torch.isfinite(old_basis).all() and torch.isfinite(new_basis).all()):
        raise ValueError("bases must contain only finite values")


def _validate_mask(mask: Tensor, basis: Tensor, name: str) -> None:
    if mask.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if mask.shape[-1] != basis.shape[-1]:
        raise ValueError(f"{name} width must equal its basis width")
    if mask.dtype is not torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if mask.device != basis.device:
        raise ValueError(f"{name} and its basis must share a device")


# Short mathematical alias used in experiment code.
build_transport = padded_transport_operator

__all__ = [
    "TransportResult",
    "apply_transport",
    "build_transport",
    "padded_transport_operator",
    "transport_coordinates",
    "transport_state",
]
