import json

import torch

from drm_language_emitter.model import DRMEmitterModel
from scripts.run_drm_fix_ablation import resolve_variant


def _matrix():
    return json.loads(
        open("configs/drm_fix_ablation_variants.json", encoding="utf-8").read()
    )


def test_geometry_component_variants_instantiate_expected_modules():
    matrix = _matrix()
    full = DRMEmitterModel(resolve_variant(matrix, "J")[0])
    no_metric = DRMEmitterModel(resolve_variant(matrix, "J_NO_METRIC")[0])
    no_direction = DRMEmitterModel(resolve_variant(matrix, "J_NO_DIRECTION")[0])
    direct_control = DRMEmitterModel(
        resolve_variant(matrix, "J_DIRECT_CONTROL")[0]
    )
    direct_control_matched = DRMEmitterModel(
        resolve_variant(matrix, "J_DIRECT_CONTROL_MATCHED")[0]
    )
    no_naturalization = DRMEmitterModel(
        resolve_variant(matrix, "J_NO_NATURALIZATION")[0]
    )

    assert full.direction_field is not None
    assert full.flow is not None
    assert full.metric is not None
    assert full.direct_transition is None

    assert no_metric.direction_field is not None
    assert no_metric.flow is not None
    assert no_metric.metric is None

    assert no_direction.direction_field is None
    assert no_direction.flow is None
    assert no_direction.direct_transition is not None
    assert no_direction.metric is not None

    for control in (direct_control, direct_control_matched):
        assert control.direction_field is None
        assert control.flow is None
        assert control.direct_transition is not None
        assert control.metric is None
        assert not control.config.use_metric_naturalization

    no_direction_parameters = sum(
        parameter.numel() for parameter in no_direction.parameters()
    )
    matched_parameters = sum(
        parameter.numel() for parameter in direct_control_matched.parameters()
    )
    assert abs(no_direction_parameters - matched_parameters) <= 1_000

    assert no_naturalization.metric is not None
    assert not no_naturalization.config.use_metric_naturalization


def test_geometry_component_variants_are_causal_finite_and_trainable():
    matrix = _matrix()
    for name in (
        "J_NO_METRIC",
        "J_NO_DIRECTION",
        "J_DIRECT_CONTROL",
        "J_DIRECT_CONTROL_MATCHED",
        "J_METRIC_SUBSPACE",
        "J_METRIC_ORTHONORMAL_DIRECTION",
        "J_NO_NATURALIZATION",
    ):
        config, _ = resolve_variant(matrix, name)
        config.vocab_size = 32
        config.d_token = 16
        config.d_state = 16
        config.n_directions = 4
        config.metric_rank = 2
        config.hidden_size = 32
        config.direction_basis_size = 4
        config.metric_u_basis_size = 4
        config.directional_cumsum_block_size = 4
        config.directional_local_mixer_hidden_size = 16
        config.selective_memory_hidden_size = 16
        config = config.validated_copy()
        model = DRMEmitterModel(config)
        first = torch.randint(0, 32, (2, 8))
        second = first.clone()
        second[:, 5:] = torch.randint(0, 32, (2, 3))

        first_out = model(first, first, collect_diagnostics=False)
        second_out = model(second, second, collect_diagnostics=False)

        assert torch.isfinite(first_out["loss"])
        assert torch.allclose(
            first_out["logits"][:, :5],
            second_out["logits"][:, :5],
            atol=1e-6,
            rtol=1e-6,
        )
        first_out["loss"].backward()
        active_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert active_gradients
        assert all(torch.isfinite(gradient).all() for gradient in active_gradients)


def test_metric_first_directional_variants_preserve_direction_span():
    matrix = _matrix()
    for name in ("J_METRIC_SUBSPACE", "J_METRIC_ORTHONORMAL_DIRECTION"):
        config, _ = resolve_variant(matrix, name)
        config.vocab_size = 32
        config.d_token = 8
        config.d_state = 12
        config.n_directions = 4
        config.metric_rank = 2
        config.hidden_size = 16
        config.direction_basis_size = 0
        config.metric_u_basis_size = 0
        config.metric_naturalization_strength = 1.0
        config.metric_naturalization_warmup_steps = 0
        config = config.validated_copy()
        model = DRMEmitterModel(config)

        directions = torch.randn(2, 4, 12)
        gates = torch.sigmoid(torch.randn(2, 4))
        coefficients = torch.randn(2, 3, 4)
        metric_diag = torch.rand(2, 12) + 0.5
        metric_u = torch.randn(2, 12, 2) * 0.1
        movement = model._compose_metric_and_directions(
            directions,
            gates,
            coefficients,
            metric_diag,
            metric_u,
            global_step=None,
        )

        reconstructed = []
        for batch_index in range(2):
            solution = torch.linalg.lstsq(
                directions[batch_index].transpose(0, 1),
                movement[batch_index].transpose(0, 1),
            ).solution
            reconstructed.append(
                torch.matmul(
                    directions[batch_index].transpose(0, 1),
                    solution,
                ).transpose(0, 1)
            )
        reconstructed_movement = torch.stack(reconstructed)
        assert torch.isfinite(movement).all()
        assert torch.allclose(
            movement,
            reconstructed_movement,
            atol=1e-4,
            rtol=1e-4,
        )


def test_metric_orthonormal_direction_handles_degenerate_basis():
    matrix = _matrix()
    config, _ = resolve_variant(matrix, "J_METRIC_ORTHONORMAL_DIRECTION")
    config.d_token = 8
    config.d_state = 8
    config.n_directions = 4
    config.metric_rank = 2
    config.hidden_size = 16
    config.direction_basis_size = 0
    config.metric_u_basis_size = 0
    config.metric_naturalization_strength = 1.0
    config.metric_naturalization_warmup_steps = 0
    config.metric_damping = 1e-6
    config = config.validated_copy()
    model = DRMEmitterModel(config)

    repeated = torch.randn(2, 1, 8).expand(-1, 4, -1).clone()
    movement = model._compose_metric_and_directions(
        repeated,
        torch.ones(2, 4),
        torch.randn(2, 3, 4),
        torch.full((2, 8), 1e-4),
        torch.randn(2, 8, 2) * 10.0,
        global_step=None,
    )

    assert torch.isfinite(movement).all()
