"""Paired-budget training and validation phases for ATTR-TG1."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .trajectory_checkpoint import (
    atomic_write_json,
    checkpoint_metadata,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
)
from .trajectory_dataset import collate_trajectory_episodes
from .trajectory_evaluation import write_records_jsonl
from .trajectory_models import TrajectoryModel
from .trajectory_protocol_io import (
    BATCH_SIZE,
    LEARNING_RATE,
    UPDATES,
    WEIGHT_DECAY,
    TrajectoryPaths,
    common_episodes,
    validate_identity,
    verify_preseal,
)
from .trajectory_runtime import evaluate_free_running
from .trajectory_training import train_trajectory_step

ModelFactory = Callable[[str | Path, str, int, int], tuple[TrajectoryModel, dict]]


def train_one(
    paths: TrajectoryPaths,
    protocol,
    factory: ModelFactory,
    device: torch.device,
    arm: str,
    seed: int,
) -> Path:
    """Run the registered 1000 updates and freeze one terminal checkpoint."""
    validate_identity(arm, seed)
    verify_preseal(paths)
    destination = paths.checkpoint(arm, seed)
    result_path = paths.result(arm, seed)
    if paths.seal.exists() or destination.exists() or result_path.exists():
        raise RuntimeError("training is immutable and must precede the seal")
    model, metadata = factory(paths.root, arm, seed, UPDATES)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        common_episodes(paths, protocol, "train"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collate_trajectory_episodes,
        drop_last=True,
    )
    iterator = iter(loader)
    trace = []
    for update in range(UPDATES):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        metrics = train_trajectory_step(model, optimizer, batch)
        if update in {0, 99, 199, 399, 599, 799, 999}:
            trace.append({"update": update + 1, **metrics})
    save_terminal_checkpoint(
        destination, model.adapter, model.head, checkpoint_metadata(arm, seed)
    )
    atomic_write_json(
        result_path,
        {
            "arm": arm,
            "optimizer_seed": seed,
            "updates": UPDATES,
            "metadata": metadata,
            "training_trace": trace,
            "test_opened": False,
        },
    )
    return destination


def validate_one(
    paths: TrajectoryPaths,
    protocol,
    factory: ModelFactory,
    device: torch.device,
    arm: str,
    seed: int,
) -> Path:
    """Write complete K256 validation records for one terminal checkpoint."""
    validate_identity(arm, seed)
    verify_preseal(paths)
    checkpoint = paths.checkpoint(arm, seed)
    destination = paths.prediction(arm, seed, "validation")
    if not checkpoint.is_file() or paths.seal.exists() or destination.exists():
        raise RuntimeError("validation requires one unsealed, unevaluated checkpoint")
    model, _ = factory(paths.root, arm, seed, UPDATES)
    load_terminal_checkpoint(checkpoint, model.adapter, model.head, device=device)
    records = evaluate_free_running(
        model,
        common_episodes(paths, protocol, "validation"),
        arm=arm,
        seed=seed,
        split="validation",
        samples=256,
        device=device,
    )
    return write_records_jsonl(destination, records)


__all__ = ["ModelFactory", "train_one", "validate_one"]
