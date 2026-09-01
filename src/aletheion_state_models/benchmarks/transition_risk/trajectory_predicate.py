"""Frozen physical failure predicate and categorical CRN sampling."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .trajectory_types import TARGET_CARDINALITIES, TrajectoryTargets


def physical_unsafe(
    agent_cell: torch.Tensor,
    trap_cells: torch.Tensor,
    moving_hazard_cell: torch.Tensor,
    low_energy_steps: torch.Tensor,
    recovery_left: torch.Tensor,
    safe_terminal: torch.Tensor,
    failure_delay: int,
) -> torch.Tensor:
    """Evaluate collision or delayed energy failure, excluding safe terminals."""
    collision = (agent_cell.unsqueeze(-1) == trap_cells).any(dim=-1)
    collision |= agent_cell == moving_hazard_cell
    delayed = (low_energy_steps >= failure_delay) & (recovery_left == 0)
    terminals = safe_terminal.bool()
    prior_terminal = (terminals.long().cumsum(dim=-1) - terminals.long()) > 0
    return (collision | delayed) & ~prior_terminal


def unsafe_from_targets(
    targets: TrajectoryTargets,
    trap_cells: torch.Tensor,
    valid_mask: torch.Tensor,
    failure_delay: int,
) -> torch.Tensor:
    """Reconstruct branch truth from physical targets only."""
    traps = trap_cells.unsqueeze(1).expand(-1, valid_mask.shape[1], -1)
    return (
        physical_unsafe(
            targets.agent_cell,
            traps,
            targets.moving_hazard_cell,
            targets.low_energy_steps,
            targets.recovery_left,
            targets.safe_terminal,
            failure_delay,
        )
        & valid_mask
    )


def horizon_unsafe(
    unsafe_truth: torch.Tensor,
    valid_mask: torch.Tensor,
    horizons: tuple[int, ...] = (1, 4, 8),
) -> torch.Tensor:
    """Return whether failure occurs in each valid branch prefix."""
    if unsafe_truth.shape != valid_mask.shape:
        raise ValueError("truth and validity shapes must match")
    columns = []
    for horizon in horizons:
        if horizon < 1 or horizon > unsafe_truth.shape[-1]:
            raise ValueError("horizon is outside the planned trajectory")
        columns.append(
            (unsafe_truth[..., :horizon] & valid_mask[..., :horizon]).any(-1)
        )
    return torch.stack(columns, dim=-1)


def categorical_inverse_cdf(
    probabilities: torch.Tensor, uniforms: torch.Tensor
) -> torch.Tensor:
    """Draw categorical values by inverse CDF, suitable for common random numbers."""
    if probabilities.ndim < 1 or probabilities.shape[-1] < 2:
        raise ValueError("probabilities need a categorical dimension")
    if uniforms.shape != probabilities.shape[:-1]:
        raise ValueError("uniforms must match all non-categorical dimensions")
    if torch.any(probabilities < 0):
        raise ValueError("probabilities cannot be negative")
    totals = probabilities.sum(-1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError("each categorical distribution needs positive mass")
    cdf = (probabilities / totals).cumsum(-1)
    clipped = uniforms.clamp(
        0.0,
        torch.nextafter(
            torch.ones((), device=uniforms.device, dtype=uniforms.dtype),
            torch.zeros((), device=uniforms.device, dtype=uniforms.dtype),
        ),
    )
    return (clipped.unsqueeze(-1) > cdf).sum(-1).long()


def sample_trajectory_risk(
    distributions: Mapping[str, torch.Tensor],
    trap_cells: torch.Tensor,
    valid_mask: torch.Tensor,
    failure_delay: int,
    *,
    samples: int = 256,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Estimate prefix risk from categorical physical targets, never hazard labels."""
    missing = set(TARGET_CARDINALITIES) - set(distributions)
    if missing:
        raise ValueError(f"missing target distributions: {sorted(missing)}")
    shape = valid_mask.shape
    draws = {}
    # A field-specific CRN is shared across alternatives in the batch.
    for field, cardinality in TARGET_CARDINALITIES.items():
        probs = distributions[field]
        if probs.shape != (*shape, cardinality):
            raise ValueError(f"invalid distribution shape for {field}")
        uniforms = torch.rand(
            (samples, 1, shape[-1]),
            device=probs.device,
            dtype=probs.dtype,
            generator=generator,
        ).expand(samples, shape[0], shape[-1])
        draws[field] = categorical_inverse_cdf(
            probs.unsqueeze(0).expand(samples, *probs.shape), uniforms
        )
    traps = trap_cells.unsqueeze(0).unsqueeze(2).expand(samples, -1, shape[-1], -1)
    unsafe = physical_unsafe(
        draws["agent_cell"],
        traps,
        draws["moving_hazard_cell"],
        draws["low_energy_steps"],
        draws["recovery_left"],
        draws["safe_terminal"],
        failure_delay,
    ) & valid_mask.unsqueeze(0)
    return unsafe.cumsum(-1).clamp(max=1).float().mean(0)
