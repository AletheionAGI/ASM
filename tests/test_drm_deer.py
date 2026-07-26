import torch

from drm_language_emitter import DRMConfig, DRMEmitterModel
from drm_language_emitter.deer import anderson_solve, causal_anderson_solve, cumulative_delta_warmstart, fixed_point_solve, sequential_rollout


def deer_tiny_config() -> DRMConfig:
    return DRMConfig(
        vocab_size=17,
        d_token=8,
        d_state=12,
        n_directions=4,
        metric_rank=2,
        hidden_size=16,
        max_seq_len=16,
        sequence_mode="directional_candidates",
        directional_candidate_scale=0.01,
        directional_candidate_temperature=1.0,
        geodesic_anchor_weight=1.0,
        geodesic_metric_weight=0.1,
        geodesic_risk_weight=0.0,
    )


def test_drm_directional_transition_fixed_point_matches_sequential_rollout():
    torch.manual_seed(23)
    model = DRMEmitterModel(deer_tiny_config())
    input_ids = torch.randint(0, 17, (2, 8))
    token_embeddings = model.token_embedding(input_ids)
    z0 = model.initializer(input_ids.shape[0], input_ids.device)

    expected = sequential_rollout(model.directional_transition, z0, token_embeddings)
    solved, residuals = fixed_point_solve(
        model.directional_transition,
        z0,
        token_embeddings,
        iterations=input_ids.shape[1],
    )

    assert residuals[-1] < residuals[0]
    assert torch.allclose(solved, expected, atol=1e-5)


def test_drm_directional_transition_anderson_reduces_residual():
    torch.manual_seed(29)
    model = DRMEmitterModel(deer_tiny_config())
    input_ids = torch.randint(0, 17, (1, 8))
    token_embeddings = model.token_embedding(input_ids)
    z0 = model.initializer(input_ids.shape[0], input_ids.device)

    expected = sequential_rollout(model.directional_transition, z0, token_embeddings)
    solved, residuals = anderson_solve(
        model.directional_transition,
        z0,
        token_embeddings,
        iterations=8,
        history_size=4,
        ridge=1e-3,
    )
    final_image, _ = fixed_point_solve(
        model.directional_transition,
        z0,
        token_embeddings,
        iterations=1,
        initial_trajectory=solved,
    )

    assert residuals[-1] < residuals[0]
    assert (final_image - solved).norm() < residuals[0]
    assert (solved - expected).abs().max() < 0.25


def test_drm_directional_transition_causal_anderson_preserves_prefix():
    torch.manual_seed(30)
    model = DRMEmitterModel(deer_tiny_config())
    input_ids = torch.randint(0, 17, (1, 8))
    changed = input_ids.clone()
    changed[:, 4:] = (changed[:, 4:] + 5) % 17
    token_embeddings = model.token_embedding(input_ids)
    changed_embeddings = model.token_embedding(changed)
    z0 = model.initializer(input_ids.shape[0], input_ids.device)

    solved, residuals = causal_anderson_solve(
        model.directional_transition,
        z0,
        token_embeddings,
        iterations=4,
        history_size=4,
        ridge=1e-3,
    )
    solved_changed, _ = causal_anderson_solve(
        model.directional_transition,
        z0,
        changed_embeddings,
        iterations=4,
        history_size=4,
        ridge=1e-3,
    )

    assert residuals[-1] < residuals[0]
    assert torch.allclose(solved[:, :4], solved_changed[:, :4], atol=1e-6)


def test_drm_cumulative_warmstart_improves_anderson_accuracy():
    torch.manual_seed(31)
    model = DRMEmitterModel(deer_tiny_config())
    input_ids = torch.randint(0, 17, (1, 16))
    token_embeddings = model.token_embedding(input_ids)
    z0 = model.initializer(input_ids.shape[0], input_ids.device)

    expected = sequential_rollout(model.directional_transition, z0, token_embeddings)
    warmstart = cumulative_delta_warmstart(model.directional_transition, z0, token_embeddings)
    repeated = z0.unsqueeze(1).expand_as(warmstart)
    solved, residuals = anderson_solve(
        model.directional_transition,
        z0,
        token_embeddings,
        iterations=4,
        history_size=4,
        ridge=1e-3,
        initial_trajectory=warmstart,
    )

    assert warmstart.shape == expected.shape
    assert torch.isfinite(warmstart).all()
    assert (warmstart - expected).norm() < (repeated - expected).norm()
    assert residuals[-1] < residuals[0]
    assert (solved - expected).abs().max() < 0.005


def test_directional_block_cumsum_is_causal_within_block():
    torch.manual_seed(37)
    config = deer_tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 8
    model = DRMEmitterModel(config)
    model.eval()
    input_ids = torch.randint(0, 17, (1, 8))
    changed = input_ids.clone()
    changed[:, 4:] = (changed[:, 4:] + 5) % 17

    with torch.no_grad():
        base = model(input_ids, return_states=True, collect_diagnostics=False)
        perturbed = model(changed, return_states=True, collect_diagnostics=False)

    assert torch.allclose(base["states"][:, :4], perturbed["states"][:, :4], atol=1e-6)
    assert torch.allclose(base["logits"][:, :4], perturbed["logits"][:, :4], atol=1e-6)


def test_directional_cumsum_pure_is_causal_across_full_sequence():
    torch.manual_seed(39)
    config = deer_tiny_config()
    config.sequence_mode = "directional_cumsum"
    model = DRMEmitterModel(config)
    model.eval()
    input_ids = torch.randint(0, 17, (1, 8))
    changed = input_ids.clone()
    changed[:, 4:] = (changed[:, 4:] + 5) % 17

    with torch.no_grad():
        base = model(input_ids, return_states=True, collect_diagnostics=False)
        perturbed = model(changed, return_states=True, collect_diagnostics=False)

    assert torch.allclose(base["states"][:, :4], perturbed["states"][:, :4], atol=1e-6)
    assert torch.allclose(base["logits"][:, :4], perturbed["logits"][:, :4], atol=1e-6)


def test_directional_block_anderson_is_causal_within_block():
    torch.manual_seed(41)
    config = deer_tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 8
    config.directional_anderson_iterations = 2
    config.directional_anderson_history_size = 4
    config.directional_anderson_ridge = 1e-3
    model = DRMEmitterModel(config)
    model.eval()
    input_ids = torch.randint(0, 17, (1, 8))
    changed = input_ids.clone()
    changed[:, 4:] = (changed[:, 4:] + 5) % 17

    with torch.no_grad():
        base = model(input_ids, return_states=True, collect_diagnostics=False)
        perturbed = model(changed, return_states=True, collect_diagnostics=False)

    assert torch.allclose(base["states"][:, :4], perturbed["states"][:, :4], atol=1e-6)
    assert torch.allclose(base["logits"][:, :4], perturbed["logits"][:, :4], atol=1e-6)


def test_directional_block_fixed_point_is_causal_within_block():
    torch.manual_seed(43)
    config = deer_tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 8
    config.directional_fixed_point_iterations = 2
    model = DRMEmitterModel(config)
    model.eval()
    input_ids = torch.randint(0, 17, (1, 8))
    changed = input_ids.clone()
    changed[:, 4:] = (changed[:, 4:] + 5) % 17

    with torch.no_grad():
        base = model(input_ids, return_states=True, collect_diagnostics=False)
        perturbed = model(changed, return_states=True, collect_diagnostics=False)

    assert torch.allclose(base["states"][:, :4], perturbed["states"][:, :4], atol=1e-6)
    assert torch.allclose(base["logits"][:, :4], perturbed["logits"][:, :4], atol=1e-6)
