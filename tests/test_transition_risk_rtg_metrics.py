from __future__ import annotations

import math

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_metrics_governance import (
    comparative_metrics,
    governance_metrics,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_metrics_state import (
    consequence_macro_accuracy,
    consequence_nll,
    expected_calibration_error,
    transition_state_metrics,
)


def _rows(seeds=(0,)):
    rows = []
    for seed in seeds:
        for action in range(6):
            unsafe = action < 2
            rows.append({
                "seed": seed, "world_id": "w", "episode_id": "e",
                "t": 0, "action_index": action,
                "predicted_state": [1.0] * 28, "true_state": [0.0] * 28,
                "persistence_state": [2.0] * 28,
                "group_nll": [float(action)] * 11,
                "group_predictions": [0] * 10 + [action % 2],
                "group_targets": [0] * 11,
                "risk": float(unsafe), "unsafe": unsafe,
                "candidate_unsafe": unsafe, "brake_unsafe": False,
                "decision": "BLOCK" if unsafe else "ALLOW",
            })
    return rows


def test_exact_state_and_physical_metrics():
    rows = _rows()
    result = transition_state_metrics(rows)
    assert result == {"mse_g": 1.0, "mse_state_persistence": 4.0,
                      "transition_nmse": 0.25}
    assert consequence_nll(rows) == 2.5
    assert consequence_macro_accuracy(rows) == 10.5 / 11
    assert expected_calibration_error(rows) == 0.0


def test_governance_executed_outcomes_and_comparison():
    result = governance_metrics(_rows())
    assert result["base"] == pytest.approx(1 / 3)
    assert result["unsafe_rate"] == 0
    assert result["relative_reduction"] == 1
    assert result["safe_service"] == 1
    assert result["coverage"] == 1
    worse = dict(result, unsafe_rate=0.1, safe_service=0.98, coverage=0.99)
    assert comparative_metrics(result, worse) == pytest.approx({
        "delta_safety": 0.1, "delta_safe_service": 0.02,
        "coverage_difference": 0.01,
    })


def test_fail_closed_on_non_six_cluster_or_zero_baseline():
    with pytest.raises(ValueError, match="six candidates"):
        transition_state_metrics(_rows()[:-1])
    rows = _rows()
    for row in rows:
        row["persistence_state"] = [0.0] * 28
    with pytest.raises(ValueError, match="positive"):
        transition_state_metrics(rows)


def test_consequence_requires_exactly_eleven_groups():
    rows = _rows()
    rows[0]["group_nll"] = [math.log(2)] * 10
    with pytest.raises(ValueError, match="11 groups"):
        consequence_nll(rows)
