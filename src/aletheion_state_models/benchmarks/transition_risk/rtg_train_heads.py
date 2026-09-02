"""Independent registered G, D, and C training for ATTR-RTG."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .rtg_batching import make_batch_plan
from .rtg_heads import DirectC, PhysicalD, TransitionG
from .rtg_losses import direct_unsafe_bce, physical_group_ce, transition_mse
from .rtg_normalization import StateNormalization
from .rtg_state_records import CandidateStateRecord
from .rtg_train_backbone import _finite_step, make_adamw


@dataclass(frozen=True)
class HeadTrainingResult:
    module: nn.Module
    terminal_update: int
    losses: tuple[float, ...]


def auxiliary_batch_indices(
    item_count: int, training_seed: int, *, updates: int = 1_000, batch_size: int = 64
) -> tuple[tuple[int, ...], ...]:
    """Common PCG64(50000+seed) cyclic batches for all three heads."""
    if item_count < 1 or updates < 1 or batch_size < 1:
        raise ValueError("batch dimensions must be positive")
    return make_batch_plan(
        item_count,
        training_seed,
        "auxiliary",
        batch_size=batch_size,
        updates=updates,
    ).batches


def _batch(
    records: tuple[CandidateStateRecord, ...], indices: tuple[int, ...], normalization: StateNormalization, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen = [records[index] for index in indices]
    pre = normalization.normalize_pre(torch.stack([item.pre_state for item in chosen])).to(device)
    true_next = normalization.normalize_next(torch.stack([item.next_state for item in chosen])).to(device)
    frames = torch.stack([item.fixed_frame for item in chosen]).to(device)
    physical = torch.tensor([item.physical_target for item in chosen], dtype=torch.long, device=device)
    unsafe = torch.tensor([item.unsafe for item in chosen], dtype=torch.float32, device=device)
    return pre, true_next, frames, physical, unsafe


def _run(
    module: nn.Module,
    records: tuple[CandidateStateRecord, ...],
    normalization: StateNormalization,
    training_seed: int,
    objective,
    *,
    updates: int,
    batch_size: int,
    device: torch.device | str,
) -> HeadTrainingResult:
    module.to(device=device, dtype=torch.float32).train()
    optimizer = make_adamw(module.parameters())
    plan = auxiliary_batch_indices(len(records), training_seed, updates=updates, batch_size=batch_size)
    losses: list[float] = []
    for indices in plan:
        loss = objective(module, _batch(records, indices, normalization, device))
        _finite_step(loss, module, optimizer)
        losses.append(float(loss.detach().cpu()))
    module.eval()
    return HeadTrainingResult(module, updates, tuple(losses))


def train_transition_g(
    records: tuple[CandidateStateRecord, ...], normalization: StateNormalization, training_seed: int,
    *, updates: int = 1_000, batch_size: int = 64, device: torch.device | str = "cpu",
) -> HeadTrainingResult:
    torch.manual_seed(60_000 + training_seed)
    module = TransitionG()
    def objective(head, values):
        pre, true_next, frames, _, _ = values
        return transition_mse(head(torch.cat((pre, frames), dim=-1)), true_next)
    return _run(module, records, normalization, training_seed, objective, updates=updates, batch_size=batch_size, device=device)


def train_physical_d(
    records: tuple[CandidateStateRecord, ...], normalization: StateNormalization, training_seed: int,
    *, updates: int = 1_000, batch_size: int = 64, device: torch.device | str = "cpu",
) -> HeadTrainingResult:
    """Train D exclusively on normalized true-next states, never on G output."""
    torch.manual_seed(70_000 + training_seed)
    module = PhysicalD()
    def objective(head, values):
        _, true_next, _, physical, _ = values
        return physical_group_ce(head(true_next), physical)
    return _run(module, records, normalization, training_seed, objective, updates=updates, batch_size=batch_size, device=device)


def train_direct_c(
    records: tuple[CandidateStateRecord, ...], normalization: StateNormalization, training_seed: int,
    *, updates: int = 1_000, batch_size: int = 64, device: torch.device | str = "cpu",
) -> HeadTrainingResult:
    torch.manual_seed(80_000 + training_seed)
    module = DirectC()
    def objective(head, values):
        pre, _, frames, _, unsafe = values
        return direct_unsafe_bce(head(torch.cat((pre, frames), dim=-1)), unsafe)
    return _run(module, records, normalization, training_seed, objective, updates=updates, batch_size=batch_size, device=device)
