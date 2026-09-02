"""The six autonomous ordered arm contrasts."""

from __future__ import annotations

import torch

from .constants import CONTRASTS
from .validation import fp64


def six_contrasts(endpoints: dict[str, object]) -> dict[str, torch.Tensor]:
    missing = {arm for pair in CONTRASTS for arm in pair} - endpoints.keys()
    if missing:
        raise ValueError(f"missing arms: {sorted(missing)}")
    tensors = {arm: fp64(value) for arm, value in endpoints.items()}
    shape = tensors["R"].shape
    if any(tensors[arm].shape != shape for arm in tensors):
        raise ValueError("arm shapes differ")
    return {f"{a}-{b}": tensors[a] - tensors[b] for a, b in CONTRASTS}
