"""H8 NLL and its frozen hierarchical fold."""

from __future__ import annotations

import torch

from .folds import equal_fold
from .validation import binary, fp64

EPSILON = 2.0**-24


def candidate_nll(probabilities, labels) -> torch.Tensor:
    p, y = fp64(probabilities), binary(labels)
    if p.shape != y.shape or bool(((p < 0) | (p > 1)).any()):
        raise ValueError("aligned probabilities in [0,1] required")
    p = p.clamp(EPSILON, 1 - EPSILON)
    return -y * torch.log(p) - (1 - y) * torch.log1p(-p)


def h8_nll(probabilities, labels, eligible=None) -> torch.Tensor:
    """Fold tensors ending in world, episode, origin, candidate."""
    return equal_fold(candidate_nll(probabilities, labels), eligible, dimensions=4)
