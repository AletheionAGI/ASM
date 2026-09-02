"""Equal-weight candidate/origin/episode/world folds."""

from __future__ import annotations

import torch

from .validation import fp64


def masked_mean(values, mask, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.as_tensor(mask, dtype=torch.bool, device=values.device)
    count = mask.sum(dim=dim)
    valid = count > 0
    total = torch.where(mask, values, 0.0).sum(dim=dim)
    return total / count.clamp_min(1), valid


def equal_fold(values, eligible=None, *, dimensions: int) -> torch.Tensor:
    """Fold the final ``dimensions`` axes, rejecting every empty group."""
    x = fp64(values)
    mask = (
        torch.ones_like(x, dtype=torch.bool)
        if eligible is None
        else torch.as_tensor(eligible, dtype=torch.bool, device=x.device)
    )
    if mask.shape != x.shape:
        raise ValueError("eligible mask shape mismatch")
    for _ in range(dimensions):
        x, group_valid = masked_mean(x, mask, -1)
        if not bool(group_valid.all()):
            raise ValueError("missing fold denominator")
        mask = group_valid
    return x
