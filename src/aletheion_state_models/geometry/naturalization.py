"""Metric naturalization without introducing another parameterized module."""

from __future__ import annotations

import torch

from drm_language_emitter.metric import RelationalMetric


def naturalize(
    metric: RelationalMetric,
    velocity: torch.Tensor,
    metric_diag: torch.Tensor,
    metric_u: torch.Tensor,
    *,
    strength: float = 1.0,
    damping: float = 0.0,
) -> torch.Tensor:
    """Apply the tested relational-metric preconditioner to a velocity."""

    return metric.naturalize(
        velocity,
        metric_diag,
        metric_u,
        strength=strength,
        damping=damping,
    )


__all__ = ["naturalize"]
