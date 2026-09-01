"""Reproducible ASM-VR Phase 1 no-bypass experiment."""

from __future__ import annotations

from typing import Any

import torch

from .frame import FrameState
from .intrinsic_dynamics import IntrinsicCollapseDynamics
from .state import VariableRankState


def _mask(width: int, rank: int) -> torch.Tensor:
    mask = torch.zeros(width, dtype=torch.bool)
    mask[:rank] = True
    return mask


def _initial_state(values: torch.Tensor, frame: FrameState) -> VariableRankState:
    mask = torch.ones(frame.frame_width, dtype=torch.bool, device=values.device)
    return VariableRankState(values, mask, frame.frame_width, frame)


def _configure_dynamics(seed: int, frame: FrameState) -> IntrinsicCollapseDynamics:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dynamics = IntrinsicCollapseDynamics(frame, input_dimension=4, output_dimension=6)
        for parameter in dynamics.parameters():
            torch.nn.init.uniform_(parameter, -0.25, 0.25)
    return dynamics


def run_phase1_experiments(
    *,
    seed: int = 2026,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Check paired-state indistinguishability through future inputs.

    The pair differs only in coordinates removed by the first 8-to-3 collapse.
    It then receives identical inputs and hard masks through re-expansion. The
    final-output Jacobian is also checked on the discarded complement.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.float64
    frame = FrameState(torch.eye(8, dtype=dtype))
    dynamics = _configure_dynamics(seed, frame)
    shared = torch.randn(3, generator=generator, dtype=dtype)
    first_tail = torch.randn(5, generator=generator, dtype=dtype)
    second_tail = torch.randn(5, generator=generator, dtype=dtype)
    first = torch.cat((shared, first_tail))
    second = torch.cat((shared, second_tail))
    inputs = [torch.randn(4, generator=generator, dtype=dtype) for _ in range(3)]
    masks = [_mask(8, rank) for rank in (3, 5, 8)]

    first_state = _initial_state(first, frame)
    second_state = _initial_state(second, frame)
    maximum_state_difference = 0.0
    maximum_output_difference = 0.0
    collapsed_difference = float("nan")
    for index, (mask, token_input) in enumerate(zip(masks, inputs, strict=True)):
        first_result = dynamics.step(first_state, mask, token_input)
        second_result = dynamics.step(second_state, mask, token_input)
        state_difference = torch.max(torch.abs(
            first_result.state.effective_coordinates
            - second_result.state.effective_coordinates
        )).item()
        output_difference = torch.max(torch.abs(
            first_result.output - second_result.output
        )).item()
        if index == 0:
            collapsed_difference = state_difference
        maximum_state_difference = max(maximum_state_difference, state_difference)
        maximum_output_difference = max(maximum_output_difference, output_difference)
        first_state = first_result.state
        second_state = second_result.state

    def final_output(initial: torch.Tensor) -> torch.Tensor:
        state = _initial_state(initial, frame)
        result = None
        for mask, token_input in zip(masks, inputs, strict=True):
            result = dynamics.step(state, mask, token_input)
            state = result.state
        assert result is not None
        return result.output

    point = first.detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(final_output, point, vectorize=True)
    complement_norm = torch.linalg.matrix_norm(jacobian[:, 3:]).item()
    initial_discarded_difference = torch.linalg.vector_norm(first[3:] - second[3:]).item()
    passed = (
        initial_discarded_difference > 0
        and collapsed_difference <= tolerance
        and maximum_state_difference <= tolerance
        and maximum_output_difference <= tolerance
        and complement_norm <= tolerance
        and first_state.transition_memory is None
        and second_state.transition_memory is None
    )
    return {
        "passed": passed,
        "initial_discarded_difference": initial_discarded_difference,
        "collapsed_state_difference": collapsed_difference,
        "maximum_future_state_difference": maximum_state_difference,
        "maximum_future_output_difference": maximum_output_difference,
        "discarded_complement_jacobian_norm": complement_norm,
        "transition_memory_enabled": False,
        "ranks": [3, 5, 8],
    }


__all__ = ["run_phase1_experiments"]
