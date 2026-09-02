import torch

from aletheion_state_models.benchmarks.transition_risk.rtg_heads import (
    DirectC,
    PhysicalD,
    TransitionG,
    build_registered_heads,
    validate_head_budgets,
)


def test_registered_heads_have_exact_shapes_budgets_and_zero_biases():
    g, d, c = TransitionG(), PhysicalD(), DirectC()
    validate_head_budgets(g, d, c)
    assert g(torch.zeros(2, 60)).shape == (2, 28)
    assert d(torch.zeros(2, 28)).shape == (2, 485)
    assert c(torch.zeros(2, 60)).shape == (2, 1)
    for model in (g, d, c):
        assert torch.count_nonzero(model.input.bias) == 0
        assert torch.count_nonzero(model.output.bias) == 0
        assert len(tuple(model.parameters())) == 4


def test_registered_head_seed_namespaces_are_reproducible():
    first = build_registered_heads(29)
    second = build_registered_heads(29)
    for left, right in zip(first, second, strict=True):
        assert all(
            torch.equal(a, b)
            for a, b in zip(left.parameters(), right.parameters(), strict=True)
        )
