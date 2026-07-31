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

    assert no_naturalization.metric is not None
    assert not no_naturalization.config.use_metric_naturalization


def test_geometry_component_variants_are_causal_finite_and_trainable():
    matrix = _matrix()
    for name in ("J_NO_METRIC", "J_NO_DIRECTION", "J_NO_NATURALIZATION"):
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
