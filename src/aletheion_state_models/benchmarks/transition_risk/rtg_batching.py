"""Frozen PCG64 permutation and cyclic batch plans for ATTR-RTG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np

from .rtg_config import TRAINING_SEEDS

ItemT = TypeVar("ItemT")
BatchNamespace = Literal["backbone", "auxiliary"]
_NAMESPACE_OFFSETS = {"backbone": 40_000, "auxiliary": 50_000}


@dataclass(frozen=True)
class BatchPlan:
    namespace: BatchNamespace
    training_seed: int
    batch_size: int
    updates: int
    permutation: tuple[int, ...]
    batches: tuple[tuple[int, ...], ...]


def make_batch_plan(
    item_count: int,
    training_seed: int,
    namespace: BatchNamespace,
    *,
    batch_size: int | None = None,
    updates: int = 1_000,
) -> BatchPlan:
    """Permute once with NumPy PCG64, then cycle without reshuffling."""
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unregistered ATTR-RTG seed: {training_seed}")
    if item_count < 1 or updates < 1:
        raise ValueError("batch planning requires positive item_count and updates")
    registered_size = 4 if namespace == "backbone" else 64
    size = registered_size if batch_size is None else batch_size
    if size != registered_size:
        raise ValueError(f"{namespace} batch size must be {registered_size}")
    rng = np.random.Generator(np.random.PCG64(_NAMESPACE_OFFSETS[namespace] + training_seed))
    permutation = tuple(int(index) for index in rng.permutation(item_count))
    stream = tuple(permutation[index % item_count] for index in range(updates * size))
    batches = tuple(
        stream[offset : offset + size] for offset in range(0, len(stream), size)
    )
    return BatchPlan(namespace, training_seed, size, updates, permutation, batches)


def materialize_batches(items: tuple[ItemT, ...], plan: BatchPlan) -> tuple[tuple[ItemT, ...], ...]:
    if len(items) != len(plan.permutation):
        raise ValueError("items differ from the batch plan population")
    return tuple(tuple(items[index] for index in batch) for batch in plan.batches)
