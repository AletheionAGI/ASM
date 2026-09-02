"""Ordinary empirical Type-7 quantiles."""

from __future__ import annotations

import torch

from .validation import fp64


def type7(values, q: float) -> torch.Tensor:
    x = fp64(values).flatten()
    if x.numel() == 0 or not 0 <= q <= 1:
        raise ValueError("Type-7 quantile requires values and q in [0,1]")
    ordered = torch.sort(x, stable=True).values
    h = (ordered.numel() - 1) * q
    lo, hi = int(h), min(int(h) + 1, ordered.numel() - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def safe_q95(probabilities, labels, eligible=None) -> torch.Tensor:
    p, y = fp64(probabilities), fp64(labels)
    if p.shape != y.shape or bool(((y != 0) & (y != 1)).any()):
        raise ValueError("probabilities and binary labels must align")
    mask = y == 0
    if eligible is not None:
        valid = torch.as_tensor(eligible, dtype=torch.bool, device=p.device)
        if valid.shape != p.shape:
            raise ValueError("eligible mask shape mismatch")
        mask &= valid
    if not bool(mask.any()):
        raise ValueError("empty safe calibration pool")
    if bool(((p < 0) | (p > 1)).any()):
        raise ValueError("invalid probability")
    return type7(p[mask], 0.95)
