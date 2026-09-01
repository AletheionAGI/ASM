"""Paired CPU/GPU training primitives for the ATTR benchmark."""

from __future__ import annotations
from dataclasses import dataclass, asdict
import random
import time
from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F
from .dataset import HazardEpisode, collate_episodes, gather_step_representations
from .metrics import basic_risk_metrics, recall_at_false_positive_rate
from .model_heads import TransitionRiskHeads


@dataclass(frozen=True)
class ATTRLossWeights:
    next_state: float = 1.0
    hazard: float = 1.0
    severity: float = 0.25
    time_to_hazard: float = 0.25


@dataclass(frozen=True)
class ATTRArmResult:
    arm: str
    backbone_parameters: int
    head_parameters: int
    total_parameters: int
    trainable_parameters: int
    final_train_loss: float
    validation_auprc_h8: float
    validation_brier_h8: float
    validation_recall_at_fpr: float
    validation_threshold: float
    arm_train_elapsed_sec: float
    updates_per_second: float

    def to_dict(self):
        return asdict(self)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    return (values * expanded).sum() / expanded.expand_as(values).sum().clamp_min(1)


def attr_loss(
    predictions, batch, weights: ATTRLossWeights = ATTRLossWeights()
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["step_mask"].float()
    next_pred = predictions["next_state"]
    scale = next_pred["log_scale"].exp()
    nll = (
        0.5 * ((batch["next_states"] - next_pred["mean"]) / scale).square()
        + next_pred["log_scale"]
    )
    next_loss = _masked_mean(nll, mask)
    hazard_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            predictions["hazard_logits"], batch["hazard_labels"], reduction="none"
        ),
        mask,
    )
    severity_loss = _masked_mean(
        F.smooth_l1_loss(
            predictions["severity"]["severity"], batch["severity"], reduction="none"
        ),
        mask,
    )
    time_mask = mask * (batch["time_to_hazard"] > 0)
    time_loss = _masked_mean(
        F.smooth_l1_loss(
            torch.log1p(predictions["severity"]["time_to_hazard"]),
            torch.log1p(batch["time_to_hazard"]),
            reduction="none",
        ),
        time_mask,
    )
    total = (
        weights.next_state * next_loss
        + weights.hazard * hazard_loss
        + weights.severity * severity_loss
        + weights.time_to_hazard * time_loss
    )
    return total, {
        "next_state": float(next_loss.detach()),
        "hazard": float(hazard_loss.detach()),
        "severity": float(severity_loss.detach()),
        "time_to_hazard": float(time_loss.detach()),
    }


def _forward(adapter, heads, batch, device):
    moved = {key: value.to(device) for key, value in batch.items()}
    representations = adapter(moved["input_ids"])
    steps = gather_step_representations(representations, moved["step_positions"])
    return heads(steps), moved


def _iter_batches(episodes: list[HazardEpisode], batch_size: int, rng: random.Random):
    indices = list(range(len(episodes)))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield collate_episodes(
            [episodes[index] for index in indices[start : start + batch_size]]
        )


def evaluate_arm(
    adapter,
    heads,
    episodes: list[HazardEpisode],
    device: torch.device,
    batch_size: int = 8,
    horizon_index: int = 2,
):
    adapter.eval()
    heads.eval()
    labels = []
    scores = []
    with torch.no_grad():
        for batch in _iter_batches(episodes, batch_size, random.Random(0)):
            predictions, moved = _forward(adapter, heads, batch, device)
            mask = moved["step_mask"].bool()
            probabilities = predictions["hazard_logits"].sigmoid()[..., horizon_index]
            labels.extend(
                moved["hazard_labels"][..., horizon_index][mask].cpu().tolist()
            )
            scores.extend(probabilities[mask].cpu().tolist())
    metrics = basic_risk_metrics(labels, scores)
    recall, threshold = recall_at_false_positive_rate(labels, scores, max_fpr=0.05)
    return metrics, recall, threshold


def train_arm(
    arm: str,
    adapter,
    heads: TransitionRiskHeads,
    train_episodes: list[HazardEpisode],
    validation_episodes: list[HazardEpisode],
    *,
    updates: int = 20,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 17,
    device: str = "cpu",
) -> ATTRArmResult:
    """Train one arm with the exact common objective and validation calibration."""
    torch.manual_seed(seed)
    target = torch.device(device)
    adapter.to(target)
    heads.to(target)
    adapter.train()
    heads.train()
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(heads.parameters()), lr=learning_rate
    )
    rng = random.Random(seed)
    loss_value = float("nan")
    completed = 0
    started = time.perf_counter()
    while completed < updates:
        for batch in _iter_batches(train_episodes, batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            predictions, moved = _forward(adapter, heads, batch, target)
            loss, _ = attr_loss(predictions, moved)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(heads.parameters()), 1.0
            )
            optimizer.step()
            loss_value = float(loss.detach())
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed >= updates:
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-9)
                remaining = (updates - completed) / max(rate, 1e-9)
                print(
                    {
                        "attr_arm": arm,
                        "update": completed,
                        "updates": updates,
                        "loss": loss_value,
                        "updates_per_second": rate,
                        "estimated_remaining_sec": remaining,
                    },
                    flush=True,
                )
            if completed >= updates:
                break
    train_elapsed = time.perf_counter() - started
    metrics, recall, threshold = evaluate_arm(
        adapter, heads, validation_episodes, target, batch_size
    )
    return ATTRArmResult(
        arm,
        count_parameters(adapter.model),
        count_parameters(heads),
        count_parameters(adapter) + count_parameters(heads),
        sum(
            parameter.numel()
            for parameter in adapter.parameters()
            if parameter.requires_grad
        )
        + sum(
            parameter.numel()
            for parameter in heads.parameters()
            if parameter.requires_grad
        ),
        loss_value,
        metrics.auprc,
        metrics.brier,
        recall,
        threshold,
        train_elapsed,
        updates / max(train_elapsed, 1e-9),
    )


__all__ = [
    "ATTRArmResult",
    "ATTRLossWeights",
    "attr_loss",
    "count_parameters",
    "evaluate_arm",
    "train_arm",
]
