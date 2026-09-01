import pytest
import torch

from aletheion_state_models.geometry.variable_rank import (
    FrameState,
    IntrinsicCollapseDynamics,
    VariableRankState,
    run_phase1_experiments,
)


def _state(values: torch.Tensor, frame: FrameState) -> VariableRankState:
    mask = torch.ones(frame.frame_width, dtype=torch.bool)
    return VariableRankState(values, mask, frame.frame_width, frame)


def test_phase1_pair_remains_indistinguishable_after_future_inputs():
    result = run_phase1_experiments(seed=31415)

    assert result["passed"]
    assert result["initial_discarded_difference"] > 0
    assert result["collapsed_state_difference"] == pytest.approx(0.0, abs=1e-12)
    assert result["maximum_future_state_difference"] == pytest.approx(0.0, abs=1e-12)
    assert result["maximum_future_output_difference"] == pytest.approx(0.0, abs=1e-12)
    assert result["discarded_complement_jacobian_norm"] == pytest.approx(0.0, abs=1e-12)
    assert result["transition_memory_enabled"] is False


def test_forcing_cannot_read_coordinates_discarded_in_same_step():
    dtype = torch.float64
    frame = FrameState(torch.eye(4, dtype=dtype))
    dynamics = IntrinsicCollapseDynamics(frame, input_dimension=1, output_dimension=2)
    with torch.no_grad():
        dynamics.forcing.weight.fill_(1.0)
        dynamics.forcing.bias.zero_()
        dynamics.emitter.weight.fill_(1.0)
        dynamics.emitter.bias.zero_()
    mask = torch.tensor([True, True, False, False])
    token = torch.zeros(1, dtype=dtype)
    first = _state(torch.tensor([1.0, 2.0, 100.0, 200.0], dtype=dtype), frame)
    second = _state(torch.tensor([1.0, 2.0, -100.0, -200.0], dtype=dtype), frame)

    first_result = dynamics.step(first, mask, token)
    second_result = dynamics.step(second, mask, token)

    assert torch.equal(first_result.state.effective_coordinates, second_result.state.effective_coordinates)
    assert torch.equal(first_result.forcing, second_result.forcing)
    assert torch.equal(first_result.output, second_result.output)


def test_phase1_dynamics_rejects_a_different_frame_or_bad_input():
    dtype = torch.float64
    frame = FrameState(torch.eye(3, dtype=dtype))
    other_frame = FrameState(torch.eye(3, dtype=dtype))
    dynamics = IntrinsicCollapseDynamics(frame, input_dimension=2, output_dimension=2)
    state = _state(torch.ones(3, dtype=dtype), other_frame)
    mask = torch.tensor([True, False, False])

    with pytest.raises(ValueError, match="fixed frame"):
        dynamics.step(state, mask, torch.zeros(2, dtype=dtype))
