import pytest

from aletheion_state_models.benchmarks.transition_risk.p2_conditioned_metrics import (
    next_state_nll_by_hazard,
)
from aletheion_state_models.benchmarks.transition_risk.p2_evaluation import (
    EpisodePrediction,
)


def _record():
    return EpisodePrediction(
        episode_id="episode",
        world_id="world",
        horizons=(1, 4, 8, 16),
        actions=("L", "R", "BRAKE"),
        hazard_labels=((0, 0, 0, 1), (0, 0, 1, 1), (0, 0, 1, 1)),
        hazard_probabilities=((0.1, 0.2, 0.3, 0.4),) * 3,
        next_state_nll=(1.0, 3.0, 5.0),
        severity=(0.0,) * 3,
        severity_predictions=(0.0,) * 3,
        time_to_hazard=(0.0,) * 3,
        time_to_hazard_predictions=(0.0,) * 3,
    )


def test_next_state_nll_is_conditioned_on_future_hazard_label():
    result = next_state_nll_by_hazard([_record()], horizon=8)
    assert result == {
        "negative": {"mean": 1.0, "n": 1},
        "positive": {"mean": 4.0, "n": 2},
    }


def test_conditioned_nll_requires_both_classes():
    record = _record()
    only_positive = EpisodePrediction(
        **{**record.__dict__, "hazard_labels": ((0, 0, 1, 1),) * 3}
    )
    with pytest.raises(ValueError, match="both H-negative and H-positive"):
        next_state_nll_by_hazard([only_positive])
