"""Tensor validation and CUDA-first FP64 placement."""

from __future__ import annotations

import torch


def statistics_device() -> torch.device:
    """Return CUDA when it is available, otherwise deterministic CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fp64(value, *, device=None) -> torch.Tensor:
    target = (
        device
        if device is not None
        else (value.device if isinstance(value, torch.Tensor) else statistics_device())
    )
    result = torch.as_tensor(value, dtype=torch.float64, device=target)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("statistics input must be finite")
    return result


def binary(value, *, device=None) -> torch.Tensor:
    result = fp64(value, device=device)
    if not bool(((result == 0) | (result == 1)).all()):
        raise ValueError("labels must be binary")
    return result
