import torch

from aletheion_state_models.benchmarks.transition_risk.rtg_normalization import (
    fit_train_normalization,
)


def test_population_statistics_are_separate_and_clamped():
    pre = torch.tensor([[1.0, 2.0], [3.0, 2.0]])
    nxt = torch.tensor([[10.0, 5.0], [14.0, 5.0]])
    stats = fit_train_normalization(pre, nxt)
    assert torch.equal(stats.pre_mean, torch.tensor([2.0, 2.0]))
    assert torch.equal(stats.pre_std, torch.tensor([1.0, 1e-6]))
    assert torch.equal(stats.next_mean, torch.tensor([12.0, 5.0]))
    assert torch.equal(stats.next_std, torch.tensor([2.0, 1e-6]))
