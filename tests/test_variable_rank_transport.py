import torch

from aletheion_state_models.geometry.variable_rank import (
    FrameState,
    VariableRankState,
    diagnose_cycle,
    padded_transport_operator,
    transport_state,
)


def _mask(rank: int) -> torch.Tensor:
    mask = torch.zeros(8, dtype=torch.bool)
    mask[:rank] = True
    return mask


def _state(values: torch.Tensor, frame: FrameState) -> VariableRankState:
    return VariableRankState(values, _mask(8), 8, frame)


def _cycle(point: torch.Tensor, frame: FrameState) -> torch.Tensor:
    state = _state(point, frame)
    for rank in (3, 5, 8):
        state = transport_state(state, frame, _mask(rank))
    return state.effective_coordinates


def test_transport_discards_coordinates_and_does_not_resurrect_them():
    frame = FrameState(torch.eye(8, dtype=torch.float64))
    initial = torch.arange(1, 9, dtype=torch.float64)

    collapsed = transport_state(_state(initial, frame), frame, _mask(3))
    expanded = transport_state(collapsed, frame, _mask(8))

    assert collapsed.effective_coordinates.tolist() == [1.0, 2.0, 3.0]
    assert expanded.effective_coordinates.tolist() == [1.0, 2.0, 3.0, 0, 0, 0, 0, 0]


def test_states_differing_only_in_discarded_coordinates_become_indistinguishable():
    frame = FrameState(torch.eye(8, dtype=torch.float64))
    first = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
    second = torch.tensor([1.0, 2.0, 3.0, -4.0, -5.0, -6.0, -7.0, -8.0], dtype=torch.float64)

    first_state = transport_state(_state(first, frame), frame, _mask(3))
    second_state = transport_state(_state(second, frame), frame, _mask(3))
    for rank in (5, 8):
        first_state = transport_state(first_state, frame, _mask(rank))
        second_state = transport_state(second_state, frame, _mask(rank))

    assert torch.equal(first_state.effective_coordinates, second_state.effective_coordinates)


def test_forcing_is_explicit_and_separate_from_transport():
    frame = FrameState(torch.eye(8, dtype=torch.float64))
    collapsed = transport_state(_state(torch.arange(8, dtype=torch.float64), frame), frame, _mask(3))
    forcing = torch.tensor([0.0, 0.0, 0.0, 11.0, 12.0], dtype=torch.float64)

    expanded = transport_state(collapsed, frame, _mask(5), forcing=forcing)

    assert expanded.effective_coordinates.tolist() == [0.0, 1.0, 2.0, 11.0, 12.0]


def test_controlled_8_to_3_to_5_to_8_cycle_has_jacobian_rank_at_most_three():
    frame = FrameState(torch.eye(8, dtype=torch.float64))
    point = torch.randn(8, dtype=torch.float64, requires_grad=True)
    diagnostics = diagnose_cycle(loop_map=lambda value: _cycle(value, frame), point=point)

    assert diagnostics.holonomy.shape == (8, 8)
    assert diagnostics.numerical_rank == 3
    assert diagnostics.rank_deficit == 5
    assert not torch.allclose(diagnostics.holonomy, torch.eye(8, dtype=torch.float64))
    assert diagnostics.minimum_dissipation_eigenvalue.item() >= -1e-10


def test_transport_uses_conjugate_overlap_for_complex_frames():
    basis = torch.eye(2, dtype=torch.complex128)
    basis[:, 1] *= 1j
    mask = torch.ones(2, dtype=torch.bool)
    operator = padded_transport_operator(basis, basis, mask, mask)
    assert torch.allclose(operator, torch.eye(2, dtype=torch.complex128))
