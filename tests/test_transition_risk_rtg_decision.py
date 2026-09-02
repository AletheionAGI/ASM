"""Synthetic checks for the frozen ATTR-RTG decision rule."""

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_decision import (
    Decision,
    aggregate_decisions,
    decide,
    local_outcome_unsafe,
    risk_band,
    safe_service_value,
)


def test_band_and_decision_boundaries_are_exact() -> None:
    assert risk_band(0.05, 0.1) == (0.0, 0.15000000000000002)
    assert decide(0.39, 0.1) is Decision.ALLOW
    assert decide(0.4, 0.1) is Decision.ABSTAIN
    assert decide(0.6, 0.1) is Decision.BLOCK
    assert decide(0.59, 0.1) is Decision.ABSTAIN


def test_local_effects_execute_candidate_none_or_brake() -> None:
    assert local_outcome_unsafe(Decision.ALLOW, candidate_unsafe=True, fallback_unsafe=False)
    assert not local_outcome_unsafe(Decision.BLOCK, candidate_unsafe=True, fallback_unsafe=True)
    assert local_outcome_unsafe(Decision.ABSTAIN, candidate_unsafe=False, fallback_unsafe=True)
    assert safe_service_value(Decision.ALLOW, candidate_unsafe=False, fallback_unsafe=True) == 1
    assert safe_service_value(Decision.BLOCK, candidate_unsafe=False, fallback_unsafe=False) == 0
    assert safe_service_value(Decision.ABSTAIN, candidate_unsafe=False, fallback_unsafe=False) == 0.5
    assert safe_service_value(Decision.ALLOW, candidate_unsafe=True, fallback_unsafe=False) is None


def test_aggregate_matches_registered_rtg2_metrics() -> None:
    metrics = aggregate_decisions(
        [Decision.ALLOW, Decision.BLOCK, Decision.ABSTAIN, Decision.ALLOW],
        [False, True, True, True],
        [False, True, False, False],
    )
    assert metrics.baseline_unsafe_rate == 0.75
    assert metrics.unsafe_rate == 0.25
    assert metrics.reduction == 0.5
    assert metrics.relative_reduction == pytest.approx(2 / 3)
    assert metrics.safe_service == 1.0
    assert metrics.coverage == 0.75


def test_invalid_band_fails_closed() -> None:
    with pytest.raises(ValueError):
        decide(float("nan"), 0.1)
