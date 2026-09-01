"""Differentiable rank objectives for ASM-VR Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RankLosses:
    budget: Tensor
    binary: Tensor
    switch: Tensor


def rank_regularization(soft_gates: Tensor, *, target_rank: float) -> RankLosses:
    """Measure budget, ambiguity, and block-to-block gate switching."""
    if soft_gates.ndim != 3:
        raise ValueError("soft_gates must have shape [batch, blocks, width]")
    if not 0.0 <= target_rank <= soft_gates.shape[-1]:
        raise ValueError("target_rank must lie in [0, width]")
    if not bool(torch.isfinite(soft_gates).all()):
        raise ValueError("soft_gates must be finite")
    if bool(torch.any((soft_gates < 0) | (soft_gates > 1))):
        raise ValueError("soft_gates must lie in [0, 1]")
    mean_rank = soft_gates.sum(dim=-1).mean()
    budget = torch.square(mean_rank - target_rank)
    binary = (soft_gates * (1.0 - soft_gates)).mean()
    if soft_gates.shape[1] < 2:
        switch = soft_gates.sum() * 0.0
    else:
        switch = torch.abs(soft_gates[:, 1:] - soft_gates[:, :-1]).mean()
    return RankLosses(budget=budget, binary=binary, switch=switch)


__all__ = ["RankLosses", "rank_regularization"]
