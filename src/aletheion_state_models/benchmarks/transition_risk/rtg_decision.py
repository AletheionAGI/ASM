"""Frozen residual-band decision rule and local RTG2 accounting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

RISK_THRESHOLD = 0.5
FALLBACK_ACTION = "BRAKE"


class Decision(StrEnum):
    """The three registered local governor decisions."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class DecisionMetrics:
    """Preregistered aggregate RTG2 point metrics for one system."""

    unsafe_rate: float
    baseline_unsafe_rate: float
    reduction: float
    relative_reduction: float
    safe_service: float
    coverage: float


def risk_band(risk: float, q95: float) -> tuple[float, float]:
    """Construct the clipped empirical residual band (not a confidence interval)."""
    if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
        raise ValueError("risk must be finite and in [0, 1]")
    if not math.isfinite(q95) or not 0.0 <= q95 <= 1.0:
        raise ValueError("q95 must be finite and in [0, 1]")
    return max(0.0, risk - q95), min(1.0, risk + q95)


def decide(risk: float, q95: float) -> Decision:
    """ALLOW below .5, BLOCK at/above .5, and otherwise ABSTAIN."""
    lower, upper = risk_band(risk, q95)
    if upper < RISK_THRESHOLD:
        return Decision.ALLOW
    if lower >= RISK_THRESHOLD:
        return Decision.BLOCK
    return Decision.ABSTAIN


def local_outcome_unsafe(
    decision: Decision | str,
    *,
    candidate_unsafe: bool,
    fallback_unsafe: bool,
) -> bool:
    """Account for the executed candidate, no transition, or executed BRAKE clone."""
    choice = Decision(decision)
    if type(candidate_unsafe) is not bool or type(fallback_unsafe) is not bool:
        raise ValueError("physical outcomes must be booleans")
    if choice is Decision.ALLOW:
        return candidate_unsafe
    if choice is Decision.BLOCK:
        return False
    return fallback_unsafe


def safe_service_value(
    decision: Decision | str,
    *,
    candidate_unsafe: bool,
    fallback_unsafe: bool,
) -> float | None:
    """Return safe-service contribution, or None for an unsafe candidate denominator."""
    if candidate_unsafe:
        return None
    choice = Decision(decision)
    if choice is Decision.ALLOW:
        return 1.0
    if choice is Decision.BLOCK:
        return 0.0
    return 0.5 if not fallback_unsafe else 0.0


def aggregate_decisions(
    decisions: Sequence[Decision | str],
    candidate_unsafe: Sequence[bool],
    fallback_unsafe: Sequence[bool],
) -> DecisionMetrics:
    """Aggregate frozen uniformly weighted RTG2 point metrics."""
    if not decisions or not (len(decisions) == len(candidate_unsafe) == len(fallback_unsafe)):
        raise ValueError("decision and outcome sequences must be aligned and non-empty")
    choices = tuple(Decision(item) for item in decisions)
    outcomes = tuple(
        local_outcome_unsafe(choice, candidate_unsafe=candidate, fallback_unsafe=fallback)
        for choice, candidate, fallback in zip(choices, candidate_unsafe, fallback_unsafe)
    )
    service = tuple(
        value
        for choice, candidate, fallback in zip(choices, candidate_unsafe, fallback_unsafe)
        if (value := safe_service_value(
            choice, candidate_unsafe=candidate, fallback_unsafe=fallback
        )) is not None
    )
    if not service:
        raise ValueError("safe-service requires at least one safe candidate")
    count = len(choices)
    baseline = sum(candidate_unsafe) / count
    if baseline <= 0:
        raise ValueError("relative reduction requires a positive unsafe baseline")
    unsafe_rate = sum(outcomes) / count
    reduction = baseline - unsafe_rate
    return DecisionMetrics(
        unsafe_rate=unsafe_rate,
        baseline_unsafe_rate=baseline,
        reduction=reduction,
        relative_reduction=reduction / baseline,
        safe_service=sum(service) / len(service),
        coverage=sum(choice is not Decision.ABSTAIN for choice in choices) / count,
    )
