from aletheion_state_models.benchmarks.transition_risk.p2_evaluation import (
    EpisodePrediction,
)
from aletheion_state_models.benchmarks.transition_risk.p2_summary import (
    paired_nll_bootstrap,
)


def _record(seed, value):
    return seed, EpisodePrediction(
        episode_id=f"episode-{seed}",
        world_id=f"world-{seed}",
        horizons=(1, 4, 8, 16),
        actions=("BRAKE", "R"),
        hazard_labels=((0, 0, 1, 1), (0, 0, 0, 1)),
        hazard_probabilities=((0.1, 0.2, 0.7, 0.8), (0.1, 0.2, 0.3, 0.7)),
        next_state_nll=(value, value),
        severity=(0.0, 0.0),
        severity_predictions=(0.1, 0.1),
        time_to_hazard=(2.0, 1.0),
        time_to_hazard_predictions=(2.0, 1.0),
    )


def test_paired_nll_bootstrap_preserves_hierarchical_pairing():
    left = [_record(seed, 2.0) for seed in (29, 43, 71, 89, 107)]
    right = [_record(seed, 1.5) for seed in (29, 43, 71, 89, 107)]
    result = paired_nll_bootstrap(left, right, replicates=50, seed=7)
    assert result["mean_delta_nll"] == 0.5
    assert result["delta_nll_ci95"] == [0.5, 0.5]
