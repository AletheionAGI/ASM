"""Validation seal, one-shot test opening, and sealed evaluation for ATTR-TG1."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .trajectory_checkpoint import (
    CheckpointPaths,
    atomic_write_json,
    digest_files,
    load_terminal_checkpoint,
)
from .trajectory_evaluation import read_records_jsonl, write_records_jsonl
from .trajectory_manifests import ARMS, OPTIMIZER_SEEDS
from .trajectory_protocol_io import (
    UPDATES,
    TrajectoryPaths,
    code_paths,
    common_episodes,
    data_paths,
    verify_preseal,
)
from .trajectory_runtime import evaluate_free_running, generate_split
from .trajectory_seal import (
    create_trajectory_seal,
    open_trajectory_seal,
    read_trajectory_preseal,
    write_trajectory_seal,
)


def checkpoint_matrix(paths: TrajectoryPaths) -> CheckpointPaths:
    return {
        (arm, seed): paths.checkpoint(arm, seed)
        for arm in ARMS
        for seed in OPTIMIZER_SEEDS
    }


def seal_validation(paths: TrajectoryPaths, protocol) -> Path:
    """Seal only after all ten complete, aligned validation outputs."""
    verify_preseal(paths)
    validation = common_episodes(paths, protocol, "validation")
    expected = {
        (episode.world_id, episode.episode_id, anchor)
        for episode in validation
        for anchor in range(episode.step_positions.numel())
    }
    for arm in ARMS:
        for seed in OPTIMIZER_SEEDS:
            records = read_records_jsonl(paths.prediction(arm, seed, "validation"))
            actual = {
                (row.identity.world_id, row.identity.episode_id, row.identity.anchor)
                for row in records
            }
            if actual != expected or any(
                row.identity.arm != arm
                or row.identity.seed != seed
                or row.identity.split != "validation"
                or row.physical_sample_count != 256
                for row in records
            ):
                raise ValueError("validation matrix is incomplete or misaligned")
    preseal = read_trajectory_preseal(paths.preseal)
    seal = create_trajectory_seal(preseal, checkpoint_matrix(paths))
    return write_trajectory_seal(seal, paths.seal)


def open_fresh_test(paths: TrajectoryPaths) -> tuple:
    """Consume the checkpoint seal once, then materialize fresh test episodes."""
    verify_preseal(paths)
    if any(paths.predictions.glob("test_*/*.jsonl")):
        raise FileExistsError("test prediction exists before the registered open")
    specs = open_trajectory_seal(
        paths.seal,
        checkpoint_matrix(paths),
        code_paths(paths),
        data_paths(paths),
    )
    manifest = [
        {"name": item.name, "episodes": len(generate_split(item, test_opened=True))}
        for item in specs
    ]
    atomic_write_json(paths.artifacts / "trajectory_test_open_event.json", manifest)
    return specs


def evaluate_fresh_test(
    paths: TrajectoryPaths, protocol, factory, device: torch.device
):
    """Evaluate every test record once; partial output invalidates the test."""
    receipt = paths.seal.with_suffix(".json.opened")
    opened = paths.artifacts / "trajectory_test_open_event.json"
    if not receipt.is_file() or not opened.is_file():
        raise RuntimeError("test evaluation requires the one-shot open")
    specs = tuple(item for item in protocol.splits if item.name.startswith("test_"))
    outputs = [
        paths.prediction(arm, seed, spec.name)
        for arm in ARMS
        for seed in OPTIMIZER_SEEDS
        for spec in specs
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("test output exists; patch/resume is forbidden")
    made = []
    for arm in ARMS:
        for seed in OPTIMIZER_SEEDS:
            model, _ = factory(paths.root, arm, seed, UPDATES)
            load_terminal_checkpoint(
                paths.checkpoint(arm, seed), model.adapter, model.head, device=device
            )
            for spec in specs:
                records = evaluate_free_running(
                    model,
                    generate_split(spec, test_opened=True),
                    arm=arm,
                    seed=seed,
                    split=spec.name,
                    samples=256,
                    device=device,
                )
                made.append(
                    write_records_jsonl(paths.prediction(arm, seed, spec.name), records)
                )
    all_predictions = [
        paths.prediction(arm, seed, split)
        for arm in ARMS
        for seed in OPTIMIZER_SEEDS
        for split in ("validation", "test_id", "test_shift", "test_ood")
    ]
    digests = digest_files(
        {
            path.relative_to(paths.predictions).as_posix(): path
            for path in all_predictions
        }
    )
    manifest = {
        "seal": paths.seal.name,
        "files": [asdict(item) for item in digests],
    }
    atomic_write_json(paths.artifacts / "trajectory_prediction_manifest.json", manifest)
    return tuple(made)


__all__ = [
    "checkpoint_matrix",
    "evaluate_fresh_test",
    "open_fresh_test",
    "seal_validation",
]
