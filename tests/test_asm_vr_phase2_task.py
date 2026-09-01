import torch

from aletheion_state_models.synthetic import (
    generate_variable_capacity_copy_batch,
    masked_copy_metrics,
    rank_difficulty_correlation,
)


def test_variable_capacity_copy_is_deterministic_balanced_and_masked():
    first = generate_variable_capacity_copy_batch(
        batch_size=8, vocab_size=23, seed=17, step=4
    )
    second = generate_variable_capacity_copy_batch(
        batch_size=8, vocab_size=23, seed=17, step=4
    )

    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.targets, second.targets)
    assert sorted(first.difficulty.tolist()) == [1, 1, 1, 1, 3, 3, 3, 3]
    assert torch.all(first.input_ids[:, 0] == first.input_ids[:, 4])
    assert torch.all(first.loss_mask[:, 4])
    assert torch.all(first.loss_mask.sum(dim=-1) == first.difficulty)


def test_masked_copy_metrics_can_reach_perfect_accuracy():
    batch = generate_variable_capacity_copy_batch(
        batch_size=4, vocab_size=16, seed=2, step=0
    )
    logits = torch.zeros(4, 7, 16)
    logits.scatter_(2, batch.targets.unsqueeze(-1), 10.0)
    loss, accuracy = masked_copy_metrics(logits, batch.targets, batch.loss_mask)

    assert loss.item() < 0.001
    assert accuracy.item() == 1.0


def test_rank_difficulty_correlation_detects_adaptation_and_constant_rank():
    difficulty = torch.tensor([1, 1, 3, 3])
    adaptive = rank_difficulty_correlation(torch.tensor([2, 2, 6, 6]), difficulty)
    fixed = rank_difficulty_correlation(torch.tensor([4, 4, 4, 4]), difficulty)

    assert torch.allclose(adaptive, torch.tensor(1.0))
    assert fixed.item() == 0.0
