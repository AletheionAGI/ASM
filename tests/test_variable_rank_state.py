from dataclasses import fields
import pytest
import torch

from aletheion_state_models.geometry.variable_rank import (
    FrameState,
    LearnedOrthonormalFrame,
    VariableRankState,
)


def _frame(width: int = 4) -> FrameState:
    return FrameState(torch.eye(width, dtype=torch.float64))


def test_variable_rank_state_exposes_only_effective_state():
    frame = _frame()
    mask = torch.tensor([True, False, True, False])
    state = VariableRankState(
        effective_coordinates=torch.tensor([[2.0, -1.0]], dtype=torch.float64),
        active_mask=mask,
        rank=2,
        frame_state=frame,
    )

    assert [field.name for field in fields(state)] == [
        "effective_coordinates",
        "active_mask",
        "rank",
        "frame_state",
        "transition_memory",
    ]
    assert not hasattr(state, "full_state")
    assert state.effective_coordinates.shape[-1] == state.rank


def test_variable_rank_state_rejects_inconsistent_or_hidden_full_state():
    frame = _frame()
    mask = torch.tensor([True, False, True, False])
    coordinates = torch.ones(1, 2, dtype=torch.float64)

    with pytest.raises(ValueError, match="rank must equal"):
        VariableRankState(coordinates, mask, 1, frame)
    with pytest.raises(ValueError, match="last effective-coordinate"):
        VariableRankState(torch.ones(1, 4, dtype=torch.float64), mask, 2, frame)
    with pytest.raises(ValueError, match="transition_memory must be None"):
        VariableRankState(coordinates, mask, 2, frame, {"full_state": coordinates})


def test_zero_rank_state_is_valid_without_ambient_payload():
    frame = _frame()
    state = VariableRankState(
        torch.empty(3, 0, dtype=torch.float64),
        torch.zeros(4, dtype=torch.bool),
        0,
        frame,
    )
    assert state.rank == 0
    assert state.effective_coordinates.numel() == 0


def test_learned_frame_is_orthonormal_and_differentiable():
    module = LearnedOrthonormalFrame(ambient_dimension=6, frame_width=4, dtype=torch.float64)
    frame = module()
    identity = torch.eye(4, dtype=torch.float64)
    assert torch.allclose(frame.basis.T @ frame.basis, identity, atol=1e-10)

    frame.basis.square().sum().backward()
    assert module.unconstrained_basis.grad is not None
    assert torch.isfinite(module.unconstrained_basis.grad).all()
