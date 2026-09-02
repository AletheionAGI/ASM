"""Exact terminal-only next-byte training loop for ATTR-RTG backbones."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .rtg_training_data import (
    BehavioralEpisode,
    backbone_batch_indices,
    collate_ce_episodes,
)


@dataclass(frozen=True)
class BackboneTrainingResult:
    model: nn.Module
    terminal_update: int
    losses: tuple[float, ...]


def make_adamw(parameters, *, lr: float = 3e-4) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


def _logits(model: nn.Module, input_ids: torch.Tensor, update: int) -> torch.Tensor:
    parameters = inspect.signature(model.forward).parameters
    kwargs = {}
    if "global_step" in parameters:
        kwargs.update(global_step=update, collect_diagnostics=False)
    output = model(input_ids, **kwargs)
    logits = output.get("logits") if isinstance(output, dict) else output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("backbone must return [batch,time,vocab] logits")
    return logits


def _finite_step(loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if loss.ndim or not torch.isfinite(loss):
        raise FloatingPointError("non-finite or non-scalar training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("non-finite or absent training gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite gradient clipping norm")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("optimizer produced non-finite parameters")


def train_backbone(
    model: nn.Module,
    episodes: tuple[BehavioralEpisode, ...],
    training_seed: int,
    *,
    updates: int = 1_000,
    batch_size: int = 4,
    sequence_length: int = 64,
    device: torch.device | str = "cpu",
) -> BackboneTrainingResult:
    """Train only explicit CE and return only the terminal in-memory model."""
    if updates < 1 or batch_size < 1 or sequence_length != 64:
        raise ValueError("updates/batch must be positive and sequence_length must be 64")
    model.to(device=device, dtype=torch.float32)
    model.train()
    optimizer = make_adamw(model.parameters())
    plan = backbone_batch_indices(len(episodes), training_seed, updates=updates, batch_size=batch_size)
    losses: list[float] = []
    for update, indices in enumerate(plan, start=1):
        batch = collate_ce_episodes([episodes[index] for index in indices], sequence_length=sequence_length)
        input_ids, targets = batch.input_ids.to(device), batch.targets.to(device)
        logits = _logits(model, input_ids, update)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        _finite_step(loss, model, optimizer)
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return BackboneTrainingResult(model, updates, tuple(losses))
