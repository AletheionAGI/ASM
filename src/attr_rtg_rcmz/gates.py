"""Frozen simultaneous-bound and five-seed marginal gates."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import REGIMES, SEEDS
from .validation import fp64


@dataclass(frozen=True)
class GateResult:
    passed: bool
    bounds_pass: bool
    marginals_pass: bool


def contrast_gate(lower, upper, marginals) -> GateResult:
    """Evaluate arrays with final endpoint order NLL, unsafe, ECE, service, coverage."""
    lo, hi, raw = fp64(lower), fp64(upper), fp64(marginals)
    if lo.shape != (len(REGIMES), 5) or hi.shape != lo.shape:
        raise ValueError("bounds must have shape [3 regimes, 5 endpoints]")
    if raw.shape != (len(SEEDS), len(REGIMES), 5):
        raise ValueError("marginals must have shape [5 seeds, 3 regimes, 5 endpoints]")
    bounds = (
        (hi[:, 0] < 0)
        & (hi[:, 1] < 0)
        & (hi[:, 2] <= 0.02)
        & (lo[:, 3] >= -0.02)
        & (lo[:, 4] >= -0.02)
    )
    marg = (
        (raw[..., 0] < 0)
        & (raw[..., 1] < 0)
        & (raw[..., 2] <= 0.02)
        & (raw[..., 3] >= -0.02)
        & (raw[..., 4] >= -0.02)
    )
    bp, mp = bool(bounds.all()), bool(marg.all())
    return GateResult(bp and mp, bp, mp)
