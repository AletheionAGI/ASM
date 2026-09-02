"""Decision endpoint construction and world-level folding."""

from __future__ import annotations

import torch

from .folds import equal_fold
from .validation import binary, fp64


def decision_origin_metrics(executed_labels, safe_candidate_exists, coverage):
    unsafe = binary(executed_labels)
    safe = torch.as_tensor(
        safe_candidate_exists, dtype=torch.bool, device=unsafe.device
    )
    cov = binary(coverage)
    if unsafe.shape != safe.shape or cov.shape != unsafe.shape:
        raise ValueError("origin shapes differ")
    service = torch.where(safe, 1 - unsafe, torch.nan)
    return {
        "unsafe_rate": unsafe,
        "safe_service": service,
        "coverage": cov,
        "abstention_rate": 1 - cov,
    }


def fold_decision_metric(values, eligible=None) -> torch.Tensor:
    """Fold a tensor ending in world, episode, origin."""
    x = fp64(values)
    mask = torch.isfinite(x)
    if eligible is not None:
        mask &= torch.as_tensor(eligible, dtype=torch.bool, device=x.device)
    return equal_fold(torch.nan_to_num(x), mask, dimensions=3)
