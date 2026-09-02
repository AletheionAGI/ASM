import torch

from aletheion_state_models.benchmarks.transition_risk.rtg_normalization import (
    fit_train_normalization,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_state_records import (
    CandidateStateRecord,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_train_heads import (
    auxiliary_batch_indices,
    train_direct_c,
    train_physical_d,
    train_transition_g,
)


def records():
    return tuple(CandidateStateRecord(
        "toy", "w", "e", 1, index,
        torch.full((28,), float(index)), torch.full((28,), float(index + 1)),
        torch.tensor([(index >> bit) & 1 for bit in range(32)], dtype=torch.float32),
        tuple([0] * 11), tuple([0] * 11), bool(index % 2), 3,
    ) for index in range(6))


def test_heads_use_common_batches_independent_seeds_and_true_next_for_d():
    items = records()
    stats = fit_train_normalization(torch.stack([x.pre_state for x in items]), torch.stack([x.next_state for x in items]))
    assert auxiliary_batch_indices(6, 29, updates=2, batch_size=64) == auxiliary_batch_indices(6, 29, updates=2, batch_size=64)
    results = [
        train_transition_g(items, stats, 29, updates=1, batch_size=64),
        train_physical_d(items, stats, 29, updates=1, batch_size=64),
        train_direct_c(items, stats, 29, updates=1, batch_size=64),
    ]
    assert all(result.terminal_update == 1 and len(result.losses) == 1 for result in results)
