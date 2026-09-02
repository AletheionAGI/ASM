import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk.rtg_normalization import (
    StateNormalization,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_state_records import (
    CandidateStateRecord,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_validation import (
    extract_calibration_scores,
    preliminary_rtg1,
)


def records():
    target = (0, 1, 2, 3, 4, 1, 1, 10, 0, 0, 0)
    return tuple(CandidateStateRecord(
        "calibration", "w", "e", 1, action,
        torch.zeros(28), torch.ones(28), torch.zeros(32), target, target, False, 3,
    ) for action in range(6))


def test_validation_metrics_and_calibration_scores_are_finite():
    stats = StateNormalization(torch.zeros(28), torch.ones(28), torch.zeros(28), torch.ones(28))
    g = nn.Linear(60, 28)
    d = nn.Linear(28, 485)
    c = nn.Linear(60, 1)
    items = records()
    metrics = preliminary_rtg1(items, stats, g, d, 13)
    assert metrics["mse_state_persistence"] > 0
    assert metrics["nll_d_true_next"] > 0
    scores = extract_calibration_scores(items, stats, g, d, c, 13, failure_delay=3)
    assert len(scores) == 6
    assert {row["unsafe"] for row in scores} == {False}
