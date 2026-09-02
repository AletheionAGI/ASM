"""Registered fixed orthonormal state projections for ATTR-RTG."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

ProjectionKind = Literal["asm", "transformer"]
PROJECTION_SEEDS = {"asm": 2_026_090_201, "transformer": 2_026_090_202}
PROJECTION_SHAPES = {"asm": (28, 28), "transformer": (32, 28)}


def _signed_qr(matrix: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def make_registered_projection(kind: ProjectionKind) -> torch.Tensor:
    """Generate the frozen CPU/PCG64/float64 QR map and store it as float32."""
    rng = np.random.Generator(np.random.PCG64(PROJECTION_SEEDS[kind]))
    if kind == "asm":
        source = rng.standard_normal((28, 28), dtype=np.float64)
    elif kind == "transformer":
        source = rng.standard_normal((28, 32), dtype=np.float64).T
    else:
        raise ValueError(f"unknown ATTR-RTG projection: {kind}")
    projection = torch.from_numpy(_signed_qr(source).astype(np.float32))
    validate_projection(projection, kind)
    return projection


def validate_projection(projection: torch.Tensor, kind: ProjectionKind) -> None:
    if projection.device.type != "cpu" or projection.dtype != torch.float32:
        raise ValueError("ATTR-RTG projection must be CPU float32")
    if tuple(projection.shape) != PROJECTION_SHAPES[kind]:
        raise ValueError("ATTR-RTG projection shape differs")
    gram = projection.T.to(torch.float64) @ projection.to(torch.float64)
    if not torch.allclose(gram, torch.eye(28, dtype=torch.float64), atol=2e-6, rtol=0):
        raise ValueError("ATTR-RTG projection is not column-orthonormal")


def project_state(states: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    if states.shape[-1] != projection.shape[0]:
        raise ValueError("state and projection dimensions do not align")
    return states @ projection.to(device=states.device, dtype=states.dtype)


def save_projection(path: str | Path, kind: ProjectionKind) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "projection": make_registered_projection(kind)}, destination)
    return destination


def load_projection(path: str | Path, kind: ProjectionKind) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("kind") != kind:
        raise ValueError("ATTR-RTG projection artifact kind differs")
    projection = payload.get("projection")
    if not isinstance(projection, torch.Tensor):
        raise TypeError("ATTR-RTG projection artifact is malformed")
    validate_projection(projection, kind)
    if not torch.equal(projection, make_registered_projection(kind)):
        raise ValueError("ATTR-RTG projection values differ from registration")
    return projection
