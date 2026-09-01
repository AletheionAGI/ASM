"""Reproducible Phase 0 invariant experiments for ASM-VR."""

from __future__ import annotations

from typing import Any

import torch

from .diagnostics import diagnose_cycle
from .frame import FrameState
from .information_probe import linear_information_probe
from .projector import hard_projector_matrix, soft_access_filter
from .state import VariableRankState
from .transport import transport_state


def _mask(width: int, rank: int) -> torch.Tensor:
    mask = torch.zeros(width, dtype=torch.bool)
    mask[:rank] = True
    return mask


def _cycle_map(point: torch.Tensor, frame: FrameState) -> torch.Tensor:
    state = VariableRankState(point, _mask(8, 8), 8, frame)
    for rank in (3, 5, 8):
        state = transport_state(state, frame, _mask(8, rank))
    return state.effective_coordinates


def run_phase0_experiments(
    *,
    samples: int = 2048,
    seed: int = 2026,
    relative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Run projector, information-probe, and controlled-cycle checks.

    The information experiment uses independent Gaussian coordinates. After an
    8-to-3 collapse, the effective state contains only the first three values;
    the five discarded values are probed both from that state and from a
    declared external archive containing them.
    """
    if samples < 32:
        raise ValueError("samples must be at least 32")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dtype = torch.float64
    basis = torch.eye(8, dtype=dtype)
    frame = FrameState(basis)

    rank_three = _mask(8, 3)
    projector = hard_projector_matrix(frame, rank_three)
    projector_error = torch.linalg.matrix_norm(projector @ projector - projector).item()

    soft_weights = torch.linspace(0.1, 0.9, 8, dtype=dtype)
    probe_vector = torch.arange(1, 9, dtype=dtype)
    soft_once = soft_access_filter(probe_vector, frame, soft_weights)
    soft_twice = soft_access_filter(soft_once, frame, soft_weights)
    soft_non_idempotence = torch.linalg.vector_norm(soft_twice - soft_once).item()

    ambient = torch.randn(samples, 8, generator=generator, dtype=dtype)
    effective = ambient[:, :3]
    discarded = ambient[:, 3:]
    effective_probe = linear_information_probe(effective, discarded, seed=seed)
    external_probe = linear_information_probe(discarded, discarded, seed=seed)

    point = torch.randn(8, generator=generator, dtype=dtype, requires_grad=True)
    cycle = diagnose_cycle(
        loop_map=lambda value: _cycle_map(value, frame),
        point=point,
        relative_tolerance=relative_tolerance,
    )
    passed = (
        projector_error < 1e-10
        and soft_non_idempotence > 1e-3
        and effective_probe.recovery_score < 0.05
        and external_probe.recovery_score > 0.99
        and cycle.numerical_rank <= 3
        and cycle.rank_deficit >= 5
    )
    return {
        "passed": passed,
        "samples": samples,
        "projector_idempotence_error": projector_error,
        "soft_filter_non_idempotence": soft_non_idempotence,
        "effective_probe_recovery": effective_probe.recovery_score,
        "external_memory_probe_recovery": external_probe.recovery_score,
        "cycle_numerical_rank": cycle.numerical_rank,
        "cycle_rank_bound": 3,
        "cycle_rank_deficit": cycle.rank_deficit,
        "cycle_frobenius_deviation": cycle.frobenius_deviation.item(),
        "cycle_minimum_dissipation_eigenvalue": (
            cycle.minimum_dissipation_eigenvalue.item()
        ),
    }


__all__ = ["run_phase0_experiments"]
