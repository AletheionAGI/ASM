from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from aletheion_state_models.variants import build_zero_choice, zero_choice_config
from drm_language_emitter import DRMConfig
from drm_language_emitter.asm_z_core import solve_spd_metric


def tiny_config(**overrides) -> DRMConfig:
    values = {
        "vocab_size": 23, "d_token": 6, "d_state": 8, "hidden_size": 12, "metric_rank": 3,
        "n_flow_steps": 1, "dropout": 0.0, "bounded_state": False,
        "use_drm_geometry": True, "use_direction_field": False,
        "use_relational_metric": False, "use_metric_naturalization": False,
        "sequence_mode": "asm_z", "compact_streaming_inference": True,
        "asm_z_eta": 0.07, "asm_z_lambda": 0.02,
        "asm_z_metric_d_min": 0.2, "asm_z_metric_d_max": 1.7,
        "asm_z_metric_u_bound": 0.6,
    }
    values.update(overrides)
    return DRMConfig(**values)


def test_builder_removes_every_catalog_and_bypass() -> None:
    model = build_zero_choice(DRMConfig(d_state=8, d_token=6, hidden_size=12, metric_rank=3))
    assert model.direction_field is None
    assert model.metric is None
    assert model.flow is None
    assert model.direct_transition is None
    assert model.updater is None
    assert model.risk is None
    assert model.addressable_memory is None
    assert model.selective_memory is None
    source = inspect.getsource(type(model.asm_z_core))
    for forbidden in ("DirectionField", "DRMFlow", "candidate", "a_i", "c_i"):
        assert forbidden not in source


def test_one_step_matches_exact_dense_equation() -> None:
    torch.manual_seed(4)
    model = build_zero_choice(tiny_config()).eval()
    state = torch.randn(2, 8)
    token = torch.randn(2, 6)
    next_state, geometry = model.asm_z_core(state, token)
    dense = torch.linalg.solve(geometry.metric, geometry.gradient.unsqueeze(-1)).squeeze(-1)
    expected = state - model.config.asm_z_eta * dense
    assert torch.allclose(next_state, expected, atol=2e-6, rtol=2e-5)
    woodbury = solve_spd_metric(geometry.diagonal, geometry.low_rank, geometry.gradient)
    assert torch.allclose(woodbury, dense, atol=2e-6, rtol=2e-5)


def test_metric_is_spd_bounded_and_low_rank_has_gauge_symmetry() -> None:
    torch.manual_seed(8)
    model = build_zero_choice(tiny_config()).eval()
    state, token = torch.randn(4, 8), torch.randn(4, 6)
    diagonal, low_rank = model.asm_z_core.metric(state, token)
    assert torch.all(diagonal > model.config.asm_z_metric_d_min)
    assert torch.all(diagonal < model.config.asm_z_metric_d_max)
    assert torch.all(low_rank.norm(dim=(1, 2)) <= model.config.asm_z_metric_u_bound + 1e-6)
    eigenvalues = torch.linalg.eigvalsh(
        torch.diag_embed(diagonal) + low_rank @ low_rank.transpose(-1, -2)
    )
    assert torch.all(eigenvalues[:, 0] >= model.config.asm_z_metric_d_min)
    eigenvalue_max = model.config.asm_z_metric_d_max + model.config.asm_z_metric_u_bound**2
    assert torch.all(eigenvalues[:, -1] <= eigenvalue_max + 1e-6)
    assert torch.all(
        eigenvalues[:, -1] / eigenvalues[:, 0]
        <= eigenvalue_max / model.config.asm_z_metric_d_min + 1e-5
    )
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = low_rank @ orthogonal
    assert torch.allclose(low_rank @ low_rank.transpose(-1, -2), rotated @ rotated.transpose(-1, -2), atol=1e-6)


def test_woodbury_solve_gradcheck_and_gradgradcheck() -> None:
    torch.manual_seed(10)
    diagonal = (torch.rand(1, 3, dtype=torch.float64) + 0.4).requires_grad_()
    low_rank = (0.1 * torch.randn(1, 3, 2, dtype=torch.float64)).requires_grad_()
    gradient = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)

    def solve(diagonal_value, low_rank_value, gradient_value):
        return solve_spd_metric(diagonal_value, low_rank_value, gradient_value)

    inputs = (diagonal, low_rank, gradient)
    assert torch.autograd.gradcheck(solve, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(solve, inputs, eps=1e-6, atol=2e-5, rtol=2e-4)


def test_woodbury_residual_is_finite_for_adversarial_bounded_geometry() -> None:
    diagonal = torch.tensor(
        [[0.100001, 1.999999, 0.100002, 1.999998]], dtype=torch.float64
    )
    raw_u = torch.tensor(
        [[[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]],
        dtype=torch.float64,
    )
    low_rank = 0.999999 * raw_u / raw_u.norm(dim=(1, 2), keepdim=True)
    gradient = torch.tensor([[1e3, -1e-3, 2e2, -4e-2]], dtype=torch.float64)
    solution = solve_spd_metric(diagonal, low_rank, gradient)
    metric = torch.diag_embed(diagonal) + low_rank @ low_rank.transpose(-1, -2)
    dense = torch.linalg.solve(metric, gradient.unsqueeze(-1)).squeeze(-1)
    residual = (metric @ solution.unsqueeze(-1)).squeeze(-1) - gradient
    assert torch.isfinite(solution).all()
    assert torch.isfinite(residual).all()
    assert torch.allclose(solution, dense, atol=1e-9, rtol=1e-9)
    assert residual.norm() <= 1e-9 + 1e-11 * gradient.norm()


def test_potential_and_metric_are_input_conditioned() -> None:
    model = build_zero_choice(tiny_config()).eval()
    state = torch.randn(2, 8)
    first_token = torch.zeros(2, 6)
    second_token = torch.ones(2, 6)
    first = model.asm_z_core.geometry(state, first_token)
    second = model.asm_z_core.geometry(state, second_token)
    assert not torch.allclose(first.potential, second.potential)
    assert not torch.allclose(first.metric, second.metric)


def test_total_potential_gradient_includes_lambda_times_state() -> None:
    model = build_zero_choice(tiny_config(asm_z_lambda=0.3)).eval()
    state = torch.randn(2, 8, requires_grad=True)
    token = torch.randn(2, 6)
    learned = model.asm_z_core.potential.net(torch.cat((state, token), dim=-1)).squeeze(-1)
    learned_gradient = torch.autograd.grad(learned.sum(), state)[0]
    total_gradient = model.asm_z_core.geometry(state, token).gradient
    assert torch.allclose(total_gradient, learned_gradient + 0.3 * state, atol=1e-7, rtol=1e-6)


def test_end_to_end_backward_reaches_potential_and_metric() -> None:
    torch.manual_seed(12)
    model = build_zero_choice(tiny_config()).train()
    ids = torch.randint(0, model.config.vocab_size, (2, 5))
    output = model(ids, targets=ids)
    output["loss"].backward()
    potential_grads = [p.grad for p in model.asm_z_core.potential.parameters()]
    metric_grads = [p.grad for p in model.asm_z_core.metric.parameters()]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in potential_grads)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in metric_grads)


def test_every_registered_parameter_is_gradient_active() -> None:
    torch.manual_seed(15)
    model = build_zero_choice(tiny_config()).train()
    ids = torch.arange(10).reshape(2, 5) % model.config.vocab_size
    model(ids, targets=(ids + 1) % model.config.vocab_size)["loss"].backward()
    inactive = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
        or parameter.grad.abs().sum() == 0
    ]
    assert inactive == []


def test_constant_potential_produces_no_token_dependent_transition() -> None:
    class ConstantPotential(nn.Module):
        def forward(self, state, token):
            return state.sum(-1) * 0.0

    model = build_zero_choice(tiny_config(asm_z_lambda=0.0)).eval()
    model.asm_z_core.potential = ConstantPotential()
    state = torch.randn(2, 8)
    first, _ = model.asm_z_core(state, torch.randn(2, 6))
    second, _ = model.asm_z_core(state, torch.randn(2, 6))
    assert torch.equal(first, state)
    assert torch.equal(second, state)


def test_recurrence_keeps_gradient_path_to_first_token() -> None:
    torch.manual_seed(17)
    model = build_zero_choice(tiny_config()).train()
    ids = torch.tensor([[1, 2, 3, 4]])
    states = model(ids, return_states=True)["states"]
    states[:, -1].square().sum().backward()
    first_token_gradient = model.token_embedding.embedding.weight.grad[1]
    assert torch.isfinite(first_token_gradient).all()
    assert first_token_gradient.abs().sum() > 0


def test_causal_prefix_and_compact_streaming_recurrence_parity() -> None:
    torch.manual_seed(18)
    model = build_zero_choice(tiny_config()).eval()
    prefix = torch.randint(0, model.config.vocab_size, (2, 4))
    suffix = torch.randint(0, model.config.vocab_size, (2, 3))
    prefix_logits = model(prefix)["logits"]
    full = model(torch.cat((prefix, suffix), dim=1))["logits"]
    assert torch.allclose(prefix_logits, full[:, : prefix.shape[1]], atol=1e-6)
    streamed_logits, state = model.prefill(prefix)
    assert state.compact and state.input_ids.shape[1] == 0
    step_logits, next_state = model.decode_step(suffix[:, :1], state)
    assert torch.allclose(streamed_logits, full[:, : prefix.shape[1]], atol=1e-6)
    assert torch.allclose(step_logits, full[:, prefix.shape[1]], atol=1e-6)
    assert next_state.input_ids.shape[1] == 0
    assert next_state.tokens_seen == prefix.shape[1] + 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asm_z_eta", float("nan")),
        ("asm_z_metric_d_min", float("nan")),
        ("asm_z_metric_u_bound", float("inf")),
        ("asm_z_lambda", float("inf")),
    ],
)
def test_zero_choice_config_rejects_nonfinite_fields(field: str, value: float) -> None:
    data = tiny_config().to_dict()
    data[field] = value
    with pytest.raises(ValueError, match="must be finite"):
        DRMConfig.from_dict(data)


def test_zero_choice_config_rejects_nonpositive_eta() -> None:
    data = tiny_config().to_dict()
    data["asm_z_eta"] = 0.0
    try:
        DRMConfig.from_dict(data)
    except ValueError as error:
        assert "asm_z_eta > 0" in str(error)
    else:
        raise AssertionError("zero eta was accepted")


def test_zero_choice_config_rejects_direction_catalog() -> None:
    configured = zero_choice_config(DRMConfig(d_state=8, d_token=6, hidden_size=12))
    assert configured.sequence_mode == "asm_z"
    data = configured.to_dict()
    data["use_direction_field"] = True
    try:
        DRMConfig.from_dict(data)
    except ValueError as error:
        assert "forbids legacy direction" in str(error)
    else:
        raise AssertionError("direction catalog was accepted")
