import torch
from torch.nn import functional as F

from aletheion_state_models.benchmarks.transition_risk.rtg_losses import (
    PHYSICAL_CARDINALITIES,
    direct_unsafe_bce,
    masked_next_byte_ce,
    physical_group_ce,
    transition_mse,
)


def test_next_byte_ce_excludes_padding_and_last_byte():
    ids = torch.tensor([[1, 2, 3, 99], [4, 5, 99, 99]])
    logits = torch.zeros(2, 4, 256)
    logits[0, 0, 2] = logits[0, 1, 3] = 5
    logits[1, 0, 5] = 5
    expected = torch.stack((
        F.cross_entropy(logits[0, 0:1], ids[0, 1:2]),
        F.cross_entropy(logits[0, 1:2], ids[0, 2:3]),
        F.cross_entropy(logits[1, 0:1], ids[1, 1:2]),
    )).mean()
    assert torch.allclose(masked_next_byte_ce(logits, ids, torch.tensor([3, 2])), expected)


def test_registered_head_losses_are_explicit_and_finite():
    targets = torch.tensor([[0] * len(PHYSICAL_CARDINALITIES)])
    assert torch.isfinite(physical_group_ce(torch.zeros(1, 485), targets))
    assert transition_mse(torch.zeros(1, 28), torch.ones(1, 28)) == 1
    assert torch.allclose(direct_unsafe_bce(torch.zeros(2, 1), torch.tensor([0, 1])), torch.log(torch.tensor(2.0)))
