from scripts.check_drm_fix_promotion import build_decision


def _payload(candidate_values, baseline_values):
    runs = []
    for seed, value in enumerate(candidate_values, 1):
        runs.append({"variant": "J", "seed": seed, "validation_ce": value})
    for seed, value in enumerate(baseline_values, 1):
        runs.append({"variant": "F", "seed": seed, "validation_ce": value})
    return {
        "runs": runs,
        "aggregate": [
            {
                "variant": "J",
                "seeds": len(candidate_values),
                "validation_ce_mean": sum(candidate_values) / len(candidate_values),
                "validation_ce_std": 0.002,
            },
            {
                "variant": "F",
                "seeds": len(baseline_values),
                "validation_ce_mean": sum(baseline_values) / len(baseline_values),
                "validation_ce_std": 0.002,
            },
        ],
    }


def test_promotion_requires_mean_gain_and_majority_of_paired_seeds():
    decision = build_decision(
        _payload([1.79, 1.80, 1.81], [1.80, 1.81, 1.82]),
        "J",
        "F",
        required_ce_improvement=0.005,
        required_seeds=3,
        max_candidate_std=0.03,
    )
    assert decision["promote"]
    assert decision["paired_wins"] == 3


def test_promotion_rejects_mean_gain_caused_by_only_one_seed():
    decision = build_decision(
        _payload([1.70, 1.82, 1.82], [1.80, 1.81, 1.81]),
        "J",
        "F",
        required_ce_improvement=0.005,
        required_seeds=3,
        max_candidate_std=0.03,
    )
    assert not decision["promote"]
    assert decision["paired_wins"] == 1
