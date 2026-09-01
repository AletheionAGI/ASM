"""Dataset generation and free-running Monte Carlo runtime for ATTR-TG1."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .trajectory_checkpoint import atomic_write_json
from .trajectory_dataset import (
    collate_trajectory_episodes,
    make_trajectory_episodes,
    make_worlds,
)
from .trajectory_evaluation import (
    HORIZONS,
    TrajectoryIdentity,
    TrajectoryRecord,
    validate_record,
)
from .trajectory_head import FIELD_CARDINALITIES
from .trajectory_manifests import SplitManifest
from .trajectory_training import move_trajectory_batch


def dynamic_family(family: str) -> str:
    """Map sealed split-family names to the existing HazardWorld generator."""
    if family in {"common_fixed", "id"}:
        return "baseline"
    if family in {"shift", "ood"}:
        return family
    raise ValueError(f"unknown trajectory family: {family}")


def generate_split(spec: SplitManifest, *, test_opened: bool = False):
    """Generate one split; test generation is impossible before the one-shot open."""
    is_test = spec.name.startswith("test_")
    if is_test and not test_opened:
        raise PermissionError("test data cannot be generated before the seal is opened")
    worlds = make_worlds(
        spec.world_count, spec.seed, dynamic_family=dynamic_family(spec.family)
    )
    return make_trajectory_episodes(worlds, spec.episodes_per_world, spec.seed)


def episode_digest(episodes: Sequence) -> str:
    """Hash all deterministic supervision without serializing tensors to JSON."""
    if not episodes:
        raise ValueError("cannot digest an empty trajectory dataset")
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode.episode_id.encode())
        digest.update(episode.world_id.encode())
        digest.update(str(episode.failure_delay).encode())
        for value in (
            episode.input_ids,
            episode.step_positions,
            episode.plan_actions,
            episode.trap_cells,
            episode.valid_mask,
            episode.unsafe_truth,
            *episode.targets.as_dict().values(),
        ):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def write_common_manifests(
    directory: str | Path, splits: Sequence[SplitManifest]
) -> dict[str, Path]:
    """Generate and write deterministic train/validation manifests only."""
    if {item.name for item in splits} != {"train", "validation"} or len(splits) != 2:
        raise ValueError(
            "preseal data generation requires exactly train and validation"
        )
    root = Path(directory)
    result = {}
    for spec in splits:
        episodes = generate_split(spec)
        payload = {
            "split": asdict(spec),
            "episodes": len(episodes),
            "sha256": episode_digest(episodes),
        }
        result[spec.name] = atomic_write_json(
            root / f"{spec.name}.manifest.json", payload
        )
    return result


def crn_uniforms(
    identity: TrajectoryIdentity,
    batch: int,
    steps: int,
    samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Create arm-independent deterministic uniforms, distinct for every field."""
    if samples < 1:
        raise ValueError("samples must be positive")
    common = f"{identity.seed}:{identity.split}:{identity.world_id}:{identity.episode_id}:{identity.anchor}"
    shapes = {"trap_cells": (samples, batch, steps, 3)} | {
        name: (samples, batch, steps, 8) for name in FIELD_CARDINALITIES
    }
    values = {}
    for name, shape in shapes.items():
        seed = int.from_bytes(
            hashlib.sha256(f"ATTR-TG1:{common}:{name}".encode()).digest()[:8], "big"
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        values[name] = torch.rand(
            shape, generator=generator, device=device, dtype=dtype
        )
    return values


def _teacher_forced_nll(
    model, context: torch.Tensor, batch: Mapping[str, object]
) -> dict[str, torch.Tensor]:
    predictions = model.head(
        context, batch["plan_actions"], batch["targets"], teacher_forcing=True
    )
    losses = {}
    trap_targets = batch["trap_cells"]
    trap_logits = predictions["trap_cells"]
    trap = F.cross_entropy(
        trap_logits.flatten(0, -2), trap_targets.flatten(), reduction="none"
    )
    losses["trap_cells"] = (
        trap.reshape(*trap_targets.shape)
        .mean(-1, keepdim=True)
        .expand(*trap_targets.shape[:2], 8)
    )
    for name in FIELD_CARDINALITIES:
        logits, target = predictions[name], batch["targets"][name]
        value = F.cross_entropy(
            logits.flatten(0, -2), target.flatten(), reduction="none"
        )
        losses[name] = value.reshape_as(target)
    losses["joint"] = losses["trap_cells"] + torch.stack(
        tuple(losses[name] for name in FIELD_CARDINALITIES)
    ).sum(0)
    return losses


def _record_nll(
    losses: Mapping[str, torch.Tensor], row: int, anchor: int, valid: torch.Tensor
):
    output = {}
    for horizon in HORIZONS:
        mask = valid[:horizon]
        output[horizon] = {
            name: float(value[row, anchor, :horizon][mask].mean().detach().cpu())
            if mask.any()
            else 0.0
            for name, value in losses.items()
        }
    return output


def evaluate_free_running(
    model,
    episodes: Sequence,
    *,
    arm: str,
    seed: int,
    split: str,
    samples: int = 256,
    device: str | torch.device = "cpu",
    origin_chunk_size: int = 64,
) -> tuple[TrajectoryRecord, ...]:
    """Evaluate K free-running physical rollouts with one causal backbone pass."""
    if not episodes or samples < 1 or origin_chunk_size < 1:
        raise ValueError(
            "evaluation episodes, samples, and chunk size must be positive"
        )
    model = model.to(device)
    model.eval()
    batch = move_trajectory_batch(collate_trajectory_episodes(episodes), device)
    records = []
    with torch.no_grad():
        context = model.encode_steps(batch["input_ids"], batch["step_positions"])
        losses = _teacher_forced_nll(model, context, batch)
        origins = [
            (row, anchor, episode)
            for row, episode in enumerate(episodes)
            for anchor in range(episode.step_positions.numel())
        ]
        for offset in range(0, len(origins), origin_chunk_size):
            chunk = origins[offset : offset + origin_chunk_size]
            identities = [
                TrajectoryIdentity(
                    seed, arm, split, episode.world_id, episode.episode_id, anchor
                )
                for row, anchor, episode in chunk
            ]
            selected_context = torch.stack(
                [context[row, anchor] for row, anchor, episode in chunk]
            )
            selected_plans = torch.stack(
                [batch["plan_actions"][row, anchor] for row, anchor, episode in chunk]
            )
            repeated_context = selected_context.unsqueeze(0).expand(samples, -1, -1)
            repeated_plans = selected_plans.unsqueeze(0).expand(samples, -1, -1)
            streams = [
                crn_uniforms(
                    identity,
                    1,
                    1,
                    samples,
                    device=context.device,
                    dtype=context.dtype,
                )
                for identity in identities
            ]
            uniforms = {
                name: torch.stack([stream[name][:, 0, 0] for stream in streams], dim=1)
                for name in streams[0]
            }
            draws = model.head.sample(repeated_context, repeated_plans, uniforms)
            traps = draws["trap_cells"]
            duplicate = (
                (traps[..., 0] == traps[..., 1])
                | (traps[..., 0] == traps[..., 2])
                | (traps[..., 1] == traps[..., 2])
            )
            delays = torch.tensor(
                [episode.failure_delay for row, anchor, episode in chunk],
                device=context.device,
            ).view(1, -1, 1)
            collision = (draws["agent_cell"].unsqueeze(-1) == traps.unsqueeze(-2)).any(
                -1
            ) | (draws["agent_cell"] == draws["moving_hazard_cell"])
            delayed = (draws["low_energy_steps"] >= delays) & (
                draws["recovery_left"] == 0
            )
            terminals = draws["safe_terminal"].bool()
            prior_terminal = (terminals.long().cumsum(-1) - terminals.long()) > 0
            unsafe = ((collision | delayed) & ~prior_terminal) | duplicate.unsqueeze(-1)
            for column, (row, anchor, episode) in enumerate(chunk):
                risks = {
                    horizon: float(
                        unsafe[:, column, :horizon].any(-1).float().mean().cpu()
                    )
                    for horizon in HORIZONS
                }
                valid = batch["valid_mask"][row, anchor]
                truth = tuple(
                    bool(value) for value in batch["unsafe_truth"][row, anchor].cpu()
                )
                record = TrajectoryRecord(
                    identities[column],
                    risks,
                    truth,
                    _record_nll(losses, row, anchor, valid),
                    samples,
                )
                validate_record(record)
                records.append(record)
    return tuple(records)


__all__ = [
    "crn_uniforms",
    "dynamic_family",
    "episode_digest",
    "evaluate_free_running",
    "generate_split",
    "write_common_manifests",
]
