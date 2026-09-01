import torch

from aletheion_state_models.geometry.variable_rank import (
    FrameState,
    hard_project,
    hard_projector_matrix,
    project_effective_coordinates,
    reconstruct_ambient_state,
    soft_access_filter,
)


def test_hard_projection_is_idempotent_and_has_requested_rank():
    basis, _ = torch.linalg.qr(torch.randn(7, 5, dtype=torch.float64))
    frame = FrameState(basis)
    mask = torch.tensor([True, False, True, True, False])
    projector = hard_projector_matrix(frame, mask)

    assert torch.allclose(projector @ projector, projector, atol=1e-10)
    assert torch.allclose(projector.T, projector, atol=1e-10)
    assert torch.linalg.matrix_rank(projector, atol=1e-10, rtol=0).item() == 3


def test_compact_coordinates_reconstruct_only_the_effective_component():
    frame = FrameState(torch.eye(5, dtype=torch.float64))
    mask = torch.tensor([True, False, True, False, False])
    ambient = torch.tensor([[1.0, 9.0, 3.0, 8.0, 7.0]], dtype=torch.float64)

    coordinates = project_effective_coordinates(ambient, frame, mask)
    reconstructed = reconstruct_ambient_state(coordinates, frame, mask)

    assert coordinates.tolist() == [[1.0, 3.0]]
    assert reconstructed.tolist() == [[1.0, 0.0, 3.0, 0.0, 0.0]]
    assert torch.equal(hard_project(ambient, frame, mask), reconstructed)


def test_soft_access_filter_is_not_mislabeled_as_a_projector():
    frame = FrameState(torch.eye(4, dtype=torch.float64))
    state = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    intensities = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)

    once = soft_access_filter(state, frame, intensities)
    twice = soft_access_filter(once, frame, intensities)

    assert not torch.allclose(twice, once)
