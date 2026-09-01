import torch

from aletheion_state_models.geometry.variable_rank import (
    linear_information_probe,
    run_phase0_experiments,
)


def test_probe_finds_no_discarded_information_in_effective_state():
    generator = torch.Generator().manual_seed(123)
    ambient = torch.randn(2048, 8, generator=generator, dtype=torch.float64)
    effective = ambient[:, :3]
    discarded = ambient[:, 3:]

    result = linear_information_probe(effective, discarded, seed=123)

    assert result.recovery_score < 0.05
    assert result.mse >= 0.95 * result.baseline_mse


def test_probe_recovers_discarded_information_only_from_external_archive():
    generator = torch.Generator().manual_seed(456)
    discarded = torch.randn(1024, 5, generator=generator, dtype=torch.float64)

    result = linear_information_probe(discarded, discarded, seed=456)

    assert result.recovery_score > 0.999
    assert result.mse < 1e-10


def test_phase0_experiment_passes_all_acceptance_checks():
    result = run_phase0_experiments(samples=1024, seed=789)

    assert result["passed"]
    assert result["effective_probe_recovery"] < 0.05
    assert result["external_memory_probe_recovery"] > 0.99
    assert result["cycle_numerical_rank"] <= result["cycle_rank_bound"]
