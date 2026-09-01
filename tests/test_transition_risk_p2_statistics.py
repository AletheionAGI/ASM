from copy import deepcopy

import pytest

from aletheion_state_models.benchmarks.transition_risk.p2_statistics import (
    aggregate_by_horizon,
    evaluate_g2_g5,
    paired_hierarchical_bootstrap,
)


def _records(arm, good=True):
    records = []
    for seed in range(5):
        for world in range(2):
            for episode in range(2):
                labels = [0, 1, 0, 1]
                scores = [0.05, 0.95, 0.1, 0.9] if good else [0.9, 0.1, 0.8, 0.2]
                records.append({
                    "seed": seed,
                    "arm": arm,
                    "split": "test",
                    "world_id": f"w{world}",
                    "episode_id": f"e{episode}",
                    "horizons": [1, 4, 8, 16],
                    "hazard_labels": [labels, labels, labels, labels],
                    "hazard_probabilities": [scores, scores, scores, scores],
                })
    return records


def test_aggregate_by_horizon_is_dependency_free_and_pools_episodes():
    metrics = aggregate_by_horizon(_records("left"))
    assert set(metrics) == {1, 4, 8, 16}
    assert metrics[8]["auprc"] == pytest.approx(1.0)
    assert metrics[8]["brier"] == pytest.approx(0.00625)
    assert metrics[8]["n"] == 80
    assert metrics[8]["positives"] == 40


def test_paired_bootstrap_is_deterministic_and_reports_seed_direction():
    left, right = _records("left"), _records("right", good=False)
    first = paired_hierarchical_bootstrap(left, right, horizon=8, replicates=60, seed=91)
    second = paired_hierarchical_bootstrap(left, right, horizon=8, replicates=60, seed=91)
    assert first == second
    assert first["mean_delta_auprc"] > 0
    assert first["mean_delta_brier"] < 0
    assert first["delta_auprc_ci95"][0] > 0
    assert len(first["per_seed"]) == 5
    assert all(item["direction"] == "left" for item in first["per_seed"].values())


def test_pairing_rejects_missing_episode_or_changed_labels():
    left, right = _records("left"), _records("right", good=False)
    with pytest.raises(ValueError, match="identical paired keys"):
        paired_hierarchical_bootstrap(left, right[:-1], replicates=2)
    changed = deepcopy(right)
    changed[0]["hazard_labels"][2][0] = 1
    with pytest.raises(ValueError, match="identical hazard labels"):
        paired_hierarchical_bootstrap(left, changed, replicates=2)


def test_registered_gates_pass_and_fail_closed():
    statistics = {
        "mean_delta_auprc": 0.04,
        "delta_auprc_ci95": [0.005, 0.08],
        "mean_delta_brier": 0.005,
        "per_seed": {seed: {"positive": seed != 4} for seed in range(5)},
    }
    assert evaluate_g2_g5(statistics, {"critical_shift": True}) == {"g2": True, "g5": True}
    assert evaluate_g2_g5({}, {}) == {"g2": False, "g5": False}
    assert evaluate_g2_g5(statistics, None)["g5"] is False
    unsafe = deepcopy(statistics)
    unsafe["delta_auprc_ci95"][0] = 0.0
    assert evaluate_g2_g5(unsafe, {"critical_shift": False}) == {"g2": False, "g5": False}


def test_invalid_serializable_shape_is_rejected():
    records = _records("left")
    records[0]["hazard_probabilities"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        aggregate_by_horizon(records)
