"""Deterministic low/high-capacity copy task for ASM-VR Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class VariableCapacityCopyBatch:
    input_ids: Tensor
    targets: Tensor
    loss_mask: Tensor
    difficulty: Tensor


def generate_variable_capacity_copy_batch(
    *,
    batch_size: int,
    vocab_size: int,
    seed: int,
    step: int,
    device: torch.device | str = "cpu",
) -> VariableCapacityCopyBatch:
    """Generate balanced one-item and three-item copy examples.

    Every four-token block starts with a causal capacity marker. Token ``1``
    means one live item and token ``2`` means three. Payload tokens start at 3.
    The second block repeats the payload, and loss is evaluated only there.
    """
    if batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if vocab_size < 11:
        raise ValueError("vocab_size must be at least eleven")
    if step < 0:
        raise ValueError("step must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 1_000_003 + int(step))
    high = torch.arange(batch_size) % 2 == 1
    high = high[torch.randperm(batch_size, generator=generator)]
    payload = torch.randint(3, vocab_size, (batch_size, 3), generator=generator)
    marker = torch.where(high, torch.tensor(2), torch.tensor(1))
    sequence = torch.zeros(batch_size, 8, dtype=torch.long)
    sequence[:, 0] = marker
    sequence[:, 1] = payload[:, 0]
    sequence[:, 4] = marker
    sequence[:, 5] = payload[:, 0]
    sequence[high, 2:4] = payload[high, 1:3]
    sequence[high, 6:8] = payload[high, 1:3]
    input_ids = sequence[:, :-1]
    targets = sequence[:, 1:]
    loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    loss_mask[:, 4] = True
    loss_mask[high, 5:7] = True
    difficulty = torch.where(high, torch.tensor(3), torch.tensor(1))
    return VariableCapacityCopyBatch(
        input_ids.to(device),
        targets.to(device),
        loss_mask.to(device),
        difficulty.to(device),
    )


def masked_copy_metrics(
    logits: Tensor, targets: Tensor, loss_mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Return masked cross-entropy and token accuracy."""
    if logits.shape[:-1] != targets.shape or targets.shape != loss_mask.shape:
        raise ValueError("logits, targets, and loss_mask shapes are incompatible")
    selected_logits = logits[loss_mask]
    selected_targets = targets[loss_mask]
    if selected_targets.numel() == 0:
        raise ValueError("loss_mask must select at least one target")
    loss = torch.nn.functional.cross_entropy(selected_logits, selected_targets)
    accuracy = (selected_logits.argmax(dim=-1) == selected_targets).float().mean()
    return loss, accuracy


def rank_difficulty_correlation(ranks: Tensor, difficulty: Tensor) -> Tensor:
    """Return a stable Pearson correlation for the two-level workload oracle."""
    ranks = ranks.float().reshape(-1)
    difficulty = difficulty.float().reshape(-1)
    if ranks.shape != difficulty.shape:
        raise ValueError("ranks and difficulty must share shape")
    centered_rank = ranks - ranks.mean()
    centered_difficulty = difficulty - difficulty.mean()
    denominator = torch.linalg.vector_norm(centered_rank) * torch.linalg.vector_norm(
        centered_difficulty
    )
    if denominator == 0:
        return ranks.new_tensor(0.0)
    return torch.dot(centered_rank, centered_difficulty) / denominator


__all__ = [
    "VariableCapacityCopyBatch",
    "generate_variable_capacity_copy_batch",
    "masked_copy_metrics",
    "rank_difficulty_correlation",
]
