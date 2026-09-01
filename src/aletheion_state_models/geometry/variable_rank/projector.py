"""Ephemeral hard projection and soft access filtering utilities."""

from __future__ import annotations

import torch

from .frame import FrameState


def _validate_active_mask(active_mask: torch.Tensor, frame_state: FrameState) -> None:
    if not isinstance(active_mask, torch.Tensor):
        raise TypeError("active_mask must be a torch.Tensor")
    if active_mask.ndim != 1 or active_mask.shape[0] != frame_state.frame_width:
        raise ValueError("active_mask must have shape (frame_state.frame_width,)")
    if active_mask.dtype != torch.bool:
        raise TypeError("active_mask must have dtype torch.bool")
    if active_mask.device != frame_state.basis.device:
        raise ValueError("active_mask and frame basis must share a device")


def _validate_ambient_state(ambient_state: torch.Tensor, frame_state: FrameState) -> None:
    if not isinstance(ambient_state, torch.Tensor):
        raise TypeError("ambient_state must be a torch.Tensor")
    if ambient_state.ndim < 1 or ambient_state.shape[-1] != frame_state.ambient_dimension:
        raise ValueError("ambient_state must end with frame_state.ambient_dimension")
    if ambient_state.device != frame_state.basis.device:
        raise ValueError("ambient_state and frame basis must share a device")
    if ambient_state.dtype != frame_state.basis.dtype:
        raise ValueError("ambient_state and frame basis must share a dtype")
    if not torch.isfinite(ambient_state).all().item():
        raise ValueError("ambient_state must contain only finite values")


def active_basis(frame_state: FrameState, active_mask: torch.Tensor) -> torch.Tensor:
    """Select active frame columns without storing an ambient state."""

    _validate_active_mask(active_mask, frame_state)
    return frame_state.basis[:, active_mask]


def hard_projector_matrix(
    frame_state: FrameState,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Construct the orthogonal hard projector ``P = B Bᴴ`` ephemerally.

    Because ``B`` consists of selected orthonormal columns, ``P @ P == P`` up
    to floating-point precision. Callers should not persist this ambient
    matrix in :class:`VariableRankState`.
    """

    basis = active_basis(frame_state, active_mask)
    return basis @ basis.mH


def project_effective_coordinates(
    ambient_state: torch.Tensor,
    frame_state: FrameState,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Project ambient vectors to compact active coordinates ``Bᴴ z``."""

    _validate_ambient_state(ambient_state, frame_state)
    basis = active_basis(frame_state, active_mask)
    return ambient_state @ basis.conj()


def reconstruct_ambient_state(
    effective_coordinates: torch.Tensor,
    frame_state: FrameState,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct ``B a`` ephemerally from compact effective coordinates."""

    if not isinstance(effective_coordinates, torch.Tensor):
        raise TypeError("effective_coordinates must be a torch.Tensor")
    basis = active_basis(frame_state, active_mask)
    rank = basis.shape[1]
    if effective_coordinates.ndim < 1 or effective_coordinates.shape[-1] != rank:
        raise ValueError("effective_coordinates must end with the active rank")
    if effective_coordinates.device != basis.device:
        raise ValueError("effective_coordinates and frame basis must share a device")
    if effective_coordinates.dtype != basis.dtype:
        raise ValueError("effective_coordinates and frame basis must share a dtype")
    if not torch.isfinite(effective_coordinates).all().item():
        raise ValueError("effective_coordinates must contain only finite values")
    return effective_coordinates @ basis.transpose(0, 1)


def hard_project(
    ambient_state: torch.Tensor,
    frame_state: FrameState,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the idempotent hard projection through compact coordinates."""

    coordinates = project_effective_coordinates(ambient_state, frame_state, active_mask)
    return reconstruct_ambient_state(coordinates, frame_state, active_mask)


def soft_access_filter(
    ambient_state: torch.Tensor,
    frame_state: FrameState,
    activation_intensities: torch.Tensor,
) -> torch.Tensor:
    """Apply ``Q diag(s) Qᴴ`` as a soft access filter, not a projector.

    Values in ``activation_intensities`` must lie in ``[0, 1]``. Unless they
    are binary, the resulting contraction is intentionally not idempotent.
    """

    _validate_ambient_state(ambient_state, frame_state)
    if not isinstance(activation_intensities, torch.Tensor):
        raise TypeError("activation_intensities must be a torch.Tensor")
    if (
        activation_intensities.ndim != 1
        or activation_intensities.shape[0] != frame_state.frame_width
    ):
        raise ValueError("activation_intensities must have shape (frame_width,)")
    if not activation_intensities.is_floating_point():
        raise TypeError("activation_intensities must use a floating-point dtype")
    if activation_intensities.device != frame_state.basis.device:
        raise ValueError("activation_intensities and frame basis must share a device")
    if not torch.isfinite(activation_intensities).all().item():
        raise ValueError("activation_intensities must contain only finite values")
    if torch.any((activation_intensities < 0) | (activation_intensities > 1)).item():
        raise ValueError("activation_intensities must lie in [0, 1]")

    frame_coordinates = ambient_state @ frame_state.basis.conj()
    compatible_intensities = activation_intensities.to(dtype=frame_coordinates.real.dtype)
    filtered_coordinates = frame_coordinates * compatible_intensities
    return filtered_coordinates @ frame_state.basis.transpose(0, 1)


__all__ = [
    "active_basis",
    "hard_project",
    "hard_projector_matrix",
    "project_effective_coordinates",
    "reconstruct_ambient_state",
    "soft_access_filter",
]
