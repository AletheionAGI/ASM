from aletheion_state_models.benchmarks.transition_risk import (
    auprc,
    brier_score,
    labels_by_horizon,
    useful_lead_time,
)


def test_multi_horizon_labels_do_not_cross_episodes_or_include_present():
    unsafe = [False, False, True, False, False, True]
    episodes = ["a", "a", "a", "b", "b", "b"]
    labels = labels_by_horizon(unsafe, (1, 2), episodes)
    assert labels[1] == (0, 1, 0, 0, 1, 0)
    assert labels[2] == (1, 1, 0, 1, 1, 0)


def test_basic_metrics_and_actionable_lead_time():
    assert auprc([1, 0, 1], [0.9, 0.8, 0.7]) == (1.0 + 2 / 3) / 2
    assert brier_score([0, 1], [0.25, 0.75]) == 0.0625
    assert useful_lead_time([0.1, 0.8, 0.9, 0.2], 4, 2, 0.7, 2) == 3
    assert useful_lead_time([0.1, 0.8, 0.1, 0.9], 4, 2, 0.7, 2) is None
