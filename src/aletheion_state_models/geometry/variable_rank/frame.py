"""Orthonormal frame state for variable-rank geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, eq=False)
class FrameState:
    """Immutable reference to an orthonormal ambient-space frame.

    ``basis`` has shape ``(ambient_dimension, frame_width)``. Its columns are
    the maximum catalog of directions from which an active subspace is chosen.
    """

    basis: torch.Tensor
    orthogonality_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if not isinstance(self.basis, torch.Tensor):
            raise TypeError("basis must be a torch.Tensor")
        if self.basis.ndim != 2:
            raise ValueError("basis must have shape (ambient_dimension, frame_width)")
        ambient_dimension, frame_width = self.basis.shape
        if ambient_dimension < 1:
            raise ValueError("ambient_dimension must be positive")
        if frame_width < 1 or frame_width > ambient_dimension:
            raise ValueError("frame_width must be in [1, ambient_dimension]")
        if not (self.basis.is_floating_point() or self.basis.is_complex()):
            raise TypeError("basis must use a floating-point or complex dtype")
        if not torch.isfinite(self.basis).all().item():
            raise ValueError("basis must contain only finite values")
        if self.orthogonality_tolerance <= 0:
            raise ValueError("orthogonality_tolerance must be positive")

        gram = self.basis.mH @ self.basis
        identity = torch.eye(frame_width, dtype=gram.dtype, device=gram.device)
        if not torch.allclose(
            gram,
            identity,
            atol=self.orthogonality_tolerance,
            rtol=self.orthogonality_tolerance,
        ):
            error = (gram - identity).abs().max().item()
            raise ValueError(f"basis columns are not orthonormal (max error {error:.3e})")

    @property
    def ambient_dimension(self) -> int:
        """Return the ambient vector-space dimension."""

        return self.basis.shape[0]

    @property
    def frame_width(self) -> int:
        """Return the number of available frame directions."""

        return self.basis.shape[1]


class LearnedOrthonormalFrame(nn.Module):
    """Learn a global frame while exposing only its orthonormalized form.

    The QR factorization is differentiable. A deterministic diagonal sign
    convention removes otherwise arbitrary QR sign flips.
    """

    def __init__(
        self,
        ambient_dimension: int,
        frame_width: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if ambient_dimension < 1:
            raise ValueError("ambient_dimension must be positive")
        if frame_width < 1 or frame_width > ambient_dimension:
            raise ValueError("frame_width must be in [1, ambient_dimension]")
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("LearnedOrthonormalFrame requires a real floating-point dtype")
        factory_kwargs = {"dtype": dtype, "device": device}
        initial = torch.empty(ambient_dimension, frame_width, **factory_kwargs)
        nn.init.orthogonal_(initial)
        self.unconstrained_basis = nn.Parameter(initial)

    def orthonormal_basis(self) -> torch.Tensor:
        """Return the current differentiable orthonormal basis."""

        q, r = torch.linalg.qr(self.unconstrained_basis, mode="reduced")
        diagonal = torch.diagonal(r)
        signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
        return q * signs.unsqueeze(0)

    def forward(self) -> FrameState:
        """Build the validated frame state for the current parameters."""

        return FrameState(self.orthonormal_basis())


__all__ = ["FrameState", "LearnedOrthonormalFrame"]
