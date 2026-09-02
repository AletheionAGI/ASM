"""Frozen train-only state normalization for ATTR-RTG."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StateNormalization:
    pre_mean: torch.Tensor
    pre_std: torch.Tensor
    next_mean: torch.Tensor
    next_std: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (self.pre_mean, self.pre_std, self.next_mean, self.next_std)
        if any(value.dtype != torch.float32 or value.ndim != 1 for value in tensors):
            raise ValueError("normalization vectors must be one-dimensional float32")
        if len({tuple(value.shape) for value in tensors}) != 1:
            raise ValueError("normalization vectors must have equal shape")
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("normalization statistics must be finite")
        if torch.any(self.pre_std < 1e-6) or torch.any(self.next_std < 1e-6):
            raise ValueError("normalization std must be clamped to 1e-6")

    def normalize_pre(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.pre_mean.to(values.device)) / self.pre_std.to(values.device)

    def normalize_next(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.next_mean.to(values.device)) / self.next_std.to(values.device)


def fit_train_normalization(pre_states: torch.Tensor, next_states: torch.Tensor) -> StateNormalization:
    """Fit per-dimension population statistics, uniformly over candidates."""
    if pre_states.ndim != 2 or next_states.shape != pre_states.shape or pre_states.shape[0] < 1:
        raise ValueError("pre/next train states must be aligned nonempty matrices")
    pre = pre_states.detach().to(device="cpu", dtype=torch.float32)
    nxt = next_states.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(pre).all() or not torch.isfinite(nxt).all():
        raise ValueError("train states must be finite")
    return StateNormalization(
        pre.mean(0), pre.std(0, correction=0).clamp_min(1e-6),
        nxt.mean(0), nxt.std(0, correction=0).clamp_min(1e-6),
    )
