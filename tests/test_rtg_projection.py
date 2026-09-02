import torch

from aletheion_state_models.benchmarks.transition_risk.rtg_projection import (
    make_registered_projection,
    project_state,
)


def test_registered_projections_are_deterministic_and_orthonormal():
    for kind, shape in (("asm", (28, 28)), ("transformer", (32, 28))):
        first = make_registered_projection(kind)
        second = make_registered_projection(kind)
        assert first.shape == shape
        assert first.dtype == torch.float32
        assert torch.equal(first, second)
        assert torch.allclose(first.T @ first, torch.eye(28), atol=2e-6, rtol=0)


def test_projection_preserves_batch_axes():
    values = torch.ones(2, 3, 32)
    assert project_state(values, make_registered_projection("transformer")).shape == (2, 3, 28)
