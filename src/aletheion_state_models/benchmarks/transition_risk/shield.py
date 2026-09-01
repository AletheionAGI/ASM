"""A model-agnostic external hard shield shared by all benchmark arms."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass

from .types import HazardPrediction, ShieldDecision


@dataclass(frozen=True)
class HardShield:
    """Reject candidates over a fixed calibrated hazard upper bound."""

    hazard_threshold: float
    fallback_action: Hashable = "STOP"

    def __post_init__(self) -> None:
        if not 0.0 <= self.hazard_threshold <= 1.0:
            raise ValueError("hazard_threshold must be in [0, 1]")

    def select(self, predictions: Iterable[HazardPrediction]) -> ShieldDecision:
        candidates = tuple(predictions)
        if not candidates:
            return ShieldDecision(self.fallback_action, (), (), True, "no_candidates")
        if any(candidate.is_ood for candidate in candidates):
            return ShieldDecision(
                self.fallback_action,
                (),
                tuple(candidate.action for candidate in candidates),
                True,
                "ood_input",
            )
        accepted = tuple(
            candidate
            for candidate in candidates
            if candidate.upper_bound <= self.hazard_threshold
        )
        rejected = tuple(
            candidate.action
            for candidate in candidates
            if candidate.upper_bound > self.hazard_threshold
        )
        if not accepted:
            return ShieldDecision(
                self.fallback_action, (), rejected, True, "no_safe_action"
            )
        # Input order provides a deterministic final tie-breaker.
        selected = max(
            enumerate(accepted), key=lambda item: (item[1].utility, -item[0])
        )[1]
        return ShieldDecision(
            selected.action,
            tuple(candidate.action for candidate in accepted),
            rejected,
            False,
            "highest_utility_safe_action",
        )


def evaluate_candidates(
    actions: Iterable[Hashable],
    predictor: Callable[[Hashable], tuple[float, float]],
    utility: Callable[[Hashable], float],
    *,
    is_ood: bool = False,
) -> tuple[HazardPrediction, ...]:
    """Apply identical prediction and utility contracts to candidate actions.

    ``predictor(action)`` returns ``(probability, calibrated_upper_bound)``.
    """
    evaluated: list[HazardPrediction] = []
    for action in actions:
        probability, upper_bound = predictor(action)
        if not (0.0 <= probability <= 1.0 and 0.0 <= upper_bound <= 1.0):
            raise ValueError("hazard probabilities and bounds must be in [0, 1]")
        if upper_bound < probability:
            raise ValueError("calibrated upper bound cannot be below probability")
        evaluated.append(
            HazardPrediction(
                action, probability, upper_bound, float(utility(action)), is_ood
            )
        )
    return tuple(evaluated)


def apply_hard_shield(
    predictions: Iterable[HazardPrediction],
    threshold: float,
    fallback_action: Hashable = "STOP",
) -> ShieldDecision:
    """Functional entry point for the common hard-shield rule."""
    return HardShield(threshold, fallback_action).select(predictions)
