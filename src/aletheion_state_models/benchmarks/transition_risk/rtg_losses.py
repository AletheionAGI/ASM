"""Explicit frozen training objectives for ATTR-RTG."""

from __future__ import annotations

import torch
from torch.nn import functional as F

PHYSICAL_CARDINALITIES = (81, 81, 81, 81, 81, 3, 3, 64, 4, 4, 2)
PHYSICAL_OFFSETS = (0, 81, 162, 243, 324, 405, 408, 411, 475, 479, 483, 485)


def masked_next_byte_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute next-byte CE within each episode; exclude padding and final bytes."""
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("logits/input_ids shapes must be [batch,time,vocab]/[batch,time]")
    if lengths.shape != (input_ids.shape[0],):
        raise ValueError("lengths must contain one logical length per episode")
    if torch.any(lengths < 2) or torch.any(lengths > input_ids.shape[1]):
        raise ValueError("logical episode lengths must be in [2,time]")
    positions = torch.arange(input_ids.shape[1] - 1, device=input_ids.device)
    mask = positions.unsqueeze(0) < (lengths.to(input_ids.device) - 1).unsqueeze(1)
    losses = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape_as(mask)
    return losses[mask].mean()


def transition_mse(predicted_next: torch.Tensor, true_next: torch.Tensor) -> torch.Tensor:
    if predicted_next.shape != true_next.shape or predicted_next.shape[-1] != 28:
        raise ValueError("G transition tensors must share a final dimension of 28")
    return F.mse_loss(predicted_next, true_next)


def physical_group_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the unweighted mean of the eleven registered categorical CEs."""
    if logits.shape[-1] != 485 or targets.shape != logits.shape[:-1] + (11,):
        raise ValueError("D logits/targets do not match the registered physical schema")
    losses = []
    for index, classes in enumerate(PHYSICAL_CARDINALITIES):
        start, stop = PHYSICAL_OFFSETS[index : index + 2]
        target = targets[..., index].long()
        if torch.any(target < 0) or torch.any(target >= classes):
            raise ValueError(f"physical target {index} is outside its cardinality")
        losses.append(F.cross_entropy(logits[..., start:stop].reshape(-1, classes), target.reshape(-1)))
    return torch.stack(losses).mean()


def direct_unsafe_bce(logits: torch.Tensor, unsafe: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1:] != (1,) or unsafe.shape != logits.shape[:-1]:
        raise ValueError("C logits and unsafe labels do not align")
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), unsafe.to(logits.dtype))
