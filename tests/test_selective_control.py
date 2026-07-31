import json

import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from scripts.run_drm_fix_ablation import resolve_variant


def _tiny_control_config() -> DRMConfig:
    return DRMConfig(
        vocab_size=32,
        d_token=16,
        d_state=16,
        hidden_size=32,
        sequence_mode="directional_block_cumsum",
        directional_cumsum_step_mode="velocity",
        directional_cumsum_block_size=4,
        directional_local_mixer="causal_conv",
        directional_local_mixer_hidden_size=16,
        directional_local_mixer_kernel_size=3,
        directional_local_mixer_layers=1,
        token_state_residual=True,
        selective_memory=True,
        selective_memory_hidden_size=24,
        use_drm_geometry=False,
        use_direction_field=False,
        use_relational_metric=False,
        instantiate_disabled_risk=False,
        lambda_action=0.0,
        lambda_dim_sparsity=0.0,
        lambda_dim_entropy=0.0,
        lambda_dim_variance=0.0,
        lambda_metric_reg=0.0,
        lambda_metric_diversity=0.0,
        lambda_active_fraction=0.0,
        lambda_condition=0.0,
        lambda_metric_u_floor=0.0,
        lambda_metric_u_target=0.0,
    )


def test_selective_control_does_not_instantiate_drm_geometry():
    model = DRMEmitterModel(_tiny_control_config())

    assert model.direction_field is None
    assert model.metric is None
    assert model.flow is None
    assert model.risk is None
    parameter_names = {name for name, _ in model.named_parameters()}
    assert not any(name.startswith("direction_field.") for name in parameter_names)
    assert not any(name.startswith("metric.") for name in parameter_names)
    assert not any(name.startswith("flow.") for name in parameter_names)
    assert not any(name.startswith("risk.") for name in parameter_names)


def test_selective_control_is_causal_and_has_finite_gradients():
    torch.manual_seed(7)
    model = DRMEmitterModel(_tiny_control_config())
    first = torch.randint(0, 32, (2, 8))
    second = first.clone()
    second[:, 5:] = torch.randint(0, 32, (2, 3))

    first_out = model(first, first, collect_diagnostics=False)
    second_out = model(second, second, collect_diagnostics=False)

    assert torch.allclose(
        first_out["logits"][:, :5],
        second_out["logits"][:, :5],
        atol=1e-6,
        rtol=1e-6,
    )
    first_out["loss"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_ssm_control_is_parameter_matched_to_j():
    matrix = json.loads(
        open("configs/drm_fix_ablation_variants.json", encoding="utf-8").read()
    )
    j = DRMEmitterModel(resolve_variant(matrix, "J")[0])
    control = DRMEmitterModel(resolve_variant(matrix, "SSM_CONTROL")[0])
    j_count = sum(parameter.numel() for parameter in j.parameters())
    control_count = sum(parameter.numel() for parameter in control.parameters())

    assert abs(j_count - control_count) / j_count < 0.0001
