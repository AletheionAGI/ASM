"""Origin-level binary ECE15 and outer folds."""

from __future__ import annotations

import torch

from .folds import equal_fold
from .validation import binary, fp64


def origin_ece15(
    probabilities, labels, eligible=None
) -> tuple[torch.Tensor, torch.Tensor]:
    p, y = fp64(probabilities), binary(labels)
    if p.shape != y.shape or bool(((p < 0) | (p > 1)).any()):
        raise ValueError("invalid ECE input")
    mask = (
        torch.ones_like(p, dtype=torch.bool)
        if eligible is None
        else torch.as_tensor(eligible, dtype=torch.bool, device=p.device)
    )
    if mask.shape != p.shape:
        raise ValueError("eligible mask shape mismatch")
    n = mask.sum(-1)
    valid = n > 0
    result = torch.zeros_like(n, dtype=torch.float64)
    bins = torch.clamp(torch.floor(p * 15).to(torch.int64), max=14)
    for index in range(15):
        selected = mask & (bins == index)
        count = selected.sum(-1)
        denom = count.clamp_min(1)
        mp = torch.where(selected, p, 0).sum(-1) / denom
        my = torch.where(selected, y, 0).sum(-1) / denom
        result += count / n.clamp_min(1) * (mp - my).abs()
    return result, valid


def ece15(probabilities, labels, eligible=None) -> torch.Tensor:
    origin, valid = origin_ece15(probabilities, labels, eligible)
    return equal_fold(origin, valid, dimensions=3)
