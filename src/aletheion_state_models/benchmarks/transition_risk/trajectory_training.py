"""Physical-only objective and training primitives for ATTR-TG1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .trajectory_head import FIELD_CARDINALITIES


@dataclass(frozen=True)
class TrajectoryLoss:
    total: torch.Tensor
    trap: torch.Tensor
    fields: torch.Tensor
    per_field: dict[str, torch.Tensor]


def _categorical_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    classes = logits.shape[-1]
    losses = F.cross_entropy(
        logits.reshape(-1, classes), targets.long().reshape(-1), reduction="none"
    ).reshape(targets.shape)
    weights = mask.to(dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def trajectory_loss(
    predictions: Mapping[str, torch.Tensor], batch: Mapping[str, object]
) -> TrajectoryLoss:
    """Compute only trap CE plus mean physical-field CE on valid future steps."""
    targets = batch.get("targets")
    if not isinstance(targets, Mapping):
        raise TypeError("batch['targets'] must be a mapping")
    required = set(FIELD_CARDINALITIES)
    missing = required - set(targets)
    if missing:
        raise KeyError(f"missing trajectory targets: {sorted(missing)}")
    if "trap_cells" not in batch or "step_mask" not in batch:
        raise KeyError("batch requires trap_cells and step_mask")
    step_mask = batch["step_mask"].bool()
    valid_mask = targets.get("valid_mask", batch.get("valid_mask"))
    if valid_mask is None:
        raise KeyError("batch requires valid_mask")
    valid = valid_mask.bool() & step_mask.unsqueeze(-1)
    trap_targets = batch["trap_cells"]
    trap_mask = step_mask.unsqueeze(-1).expand_as(trap_targets)
    trap = _categorical_loss(predictions["trap_cells"], trap_targets, trap_mask)
    per_field = {
        name: _categorical_loss(predictions[name], targets[name], valid)
        for name in FIELD_CARDINALITIES
    }
    fields = torch.stack(tuple(per_field.values())).mean()
    return TrajectoryLoss(trap + fields, trap, fields, per_field)


def move_trajectory_batch(
    batch: Mapping[str, object], device: torch.device | str
) -> dict:
    """Move the tensor batch, including its nested target mapping."""
    moved = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[name] = value.to(device)
        elif name == "targets" and isinstance(value, Mapping):
            moved[name] = {key: tensor.to(device) for key, tensor in value.items()}
        else:
            moved[name] = value
    return moved


def forward_trajectory_batch(
    model: nn.Module,
    batch: Mapping[str, object],
    *,
    teacher_forcing: bool | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Run the common batch contract through a TrajectoryModel."""
    device = next(model.parameters()).device
    moved = move_trajectory_batch(batch, device)
    predictions = model(
        moved["input_ids"],
        moved["step_positions"],
        moved["plan_actions"],
        moved["targets"],
        teacher_forcing=teacher_forcing,
    )
    return predictions, moved


def train_trajectory_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, object],
    *,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Apply one finite teacher-forced update and report physical losses."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions, moved = forward_trajectory_batch(model, batch, teacher_forcing=True)
    losses = trajectory_loss(predictions, moved)
    if not torch.isfinite(losses.total):
        raise FloatingPointError("non-finite ATTR-TG1 loss")
    losses.total.backward()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("non-finite ATTR-TG1 gradients")
    optimizer.step()
    return {
        "loss": float(losses.total.detach()),
        "trap_ce": float(losses.trap.detach()),
        "field_ce": float(losses.fields.detach()),
        "gradient_norm": float(gradient_norm.detach()),
    }


__all__ = [
    "TrajectoryLoss",
    "forward_trajectory_batch",
    "move_trajectory_batch",
    "train_trajectory_step",
    "trajectory_loss",
]
