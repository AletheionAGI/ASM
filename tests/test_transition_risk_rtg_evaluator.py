"""Toy-only checks for the pure ATTR-RTG evaluator."""

from __future__ import annotations

import torch

from aletheion_state_models.benchmarks.transition_risk.rtg_calibration import (
    RtgCalibration,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_evaluator import (
    evaluate_c_records,
    evaluate_g_records,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_types import (
    Y_COMMON_CARDINALITIES,
)


def _rows():
    truth = (0, 1, 2, 10, 20, 1, 1, 32, 0, 1, 0)
    return [
        {
            "seed": 29,
            "split_id": "toy-split",
            "world_id": "toy-world",
            "episode_id": "toy-episode",
            "origin_id": "toy-origin",
            "t": 1,
            "action_index": action,
            "normalized_state": [0.0] * 28,
            "fixed_frame": [-1.0] * 32,
            "true_next_state": [1.0] * 28,
            "persistence_state": [0.0] * 28,
            "y_common": truth,
            "persistence_target": truth,
            "failure_delay": 3,
            "candidate_unsafe": action % 2 == 0,
            "brake_unsafe": False,
        }
        for action in range(6)
    ]


def _physical_logits() -> torch.Tensor:
    truth = _rows()[0]["y_common"]
    groups = []
    for cardinality, target in zip(Y_COMMON_CARDINALITIES, truth):
        group = torch.full((cardinality,), -20.0)
        group[target] = 20.0
        groups.append(group)
    return torch.cat(groups)


def test_g_evaluator_produces_state_physics_decision_and_metrics() -> None:
    seen = []

    def g(value):
        seen.append(tuple(value.shape))
        return torch.zeros((1, 28))

    def d(value):
        assert value.shape == (1, 28)
        return _physical_logits().unsqueeze(0)

    result = evaluate_g_records(_rows(), g, d, RtgCalibration(1.0, 0.01))
    assert seen == [(1, 60)] * 6
    assert len(result.records) == 6
    assert result.metrics["transition_nmse"] == 1.0
    assert result.metrics["d_macro_accuracy"] == 1.0
    assert result.metrics["consequence_nll"] < 1e-12
    assert all(row["decision"] == "ALLOW" for row in result.records)


def test_c_evaluator_uses_raw_logit_and_common_governance_metrics() -> None:
    result = evaluate_c_records(
        _rows(), lambda value: torch.zeros((value.shape[0], 1)), RtgCalibration(1.0, 0.0)
    )
    assert all(row["risk"] == 0.5 and row["decision"] == "BLOCK" for row in result.records)
    assert result.metrics["unsafe_rate"] == 0.0
    assert result.metrics["coverage"] == 1.0


def test_evaluator_rejects_training_modules() -> None:
    module = torch.nn.Linear(60, 1)
    try:
        evaluate_c_records(_rows(), module, RtgCalibration(1.0, 0.0))
    except ValueError as error:
        assert "eval mode" in str(error)
    else:
        raise AssertionError("training module was accepted")
