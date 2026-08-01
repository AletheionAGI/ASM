import torch

from drm_language_emitter import DRMConfig, DRMEmitterModel


def tiny_config() -> DRMConfig:
    return DRMConfig(vocab_size=17, d_token=8, d_state=12, n_directions=4, metric_rank=2, hidden_size=16, max_seq_len=8)


def test_forward_cpu_shapes_and_loss_finite():
    model = DRMEmitterModel(tiny_config())
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y)
    assert out["logits"].shape == (2, 6, 17)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["aux_losses"]["ce"])


def test_metric_rank_zero_does_not_add_u_floor_loss():
    config = DRMConfig(
        vocab_size=17,
        d_token=8,
        d_state=12,
        n_directions=4,
        metric_rank=0,
        hidden_size=16,
        max_seq_len=8,
        lambda_metric_u_floor=1.0,
    )
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y)
    assert out["diagnostics"]["metric_u_floor_loss"].item() == 0.0
    assert "metric_u_floor" not in out["aux_losses"]


def test_compiled_forward_failure_falls_back_to_eager():
    model = DRMEmitterModel(tiny_config())

    def broken_compiled_forward(*args, **kwargs):
        raise RuntimeError("compile backend unavailable")

    model._compiled_forward = broken_compiled_forward
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y)
    assert out["logits"].shape == (2, 6, 17)
    assert model._compiled_forward is None


def test_geometry_update_interval_keeps_forward_finite():
    config = tiny_config()
    config.geometry_update_interval = 3
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y, collect_diagnostics=False)
    assert out["logits"].shape == (2, 6, 17)
    assert torch.isfinite(out["loss"])


def test_factorized_geometry_heads_keep_forward_finite():
    config = tiny_config()
    config.direction_basis_size = 3
    config.metric_u_basis_size = 3
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y, collect_diagnostics=False)
    assert out["logits"].shape == (2, 6, 17)
    assert torch.isfinite(out["loss"])


def test_bptt_truncate_interval_keeps_backward_finite():
    config = tiny_config()
    config.bptt_truncate_interval = 2
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y, collect_diagnostics=False)
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_geodesic_step_forward_without_targets_is_finite():
    config = tiny_config()
    config.sequence_mode = "geodesic_step"
    config.geodesic_solver_steps = 1
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    out = model(x, collect_diagnostics=False)
    assert out["logits"].shape == (2, 6, 17)
    assert torch.isfinite(out["loss"])


def test_geodesic_step_forward_works_under_no_grad():
    config = tiny_config()
    config.sequence_mode = "geodesic_step"
    config.geodesic_solver_steps = 1
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    with torch.no_grad():
        out = model(x, collect_diagnostics=False)
    assert out["logits"].shape == (2, 6, 17)
    assert torch.isfinite(out["loss"])


def test_geodesic_step_backward_is_finite():
    config = tiny_config()
    config.sequence_mode = "geodesic_step"
    config.geodesic_solver_steps = 1
    config.geodesic_lr = 0.001
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y, collect_diagnostics=False)
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_candidates_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_candidates"
    config.directional_candidate_temperature = 0.7
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y, collect_diagnostics=False)
    assert out["logits"].shape == (2, 6, 17)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_cumsum_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_cumsum"
    config.directional_candidate_scale = 0.01
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    assert out["logits"].shape == (2, 6, 17)
    assert out["states"].shape == (2, 6, config.d_state)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_block_cumsum_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 2
    config.directional_candidate_scale = 0.01
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    assert out["logits"].shape == (2, 6, 17)
    assert out["states"].shape == (2, 6, config.d_state)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_block_endpoint_correction_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 3
    config.directional_candidate_scale = 0.01
    config.directional_endpoint_correction_weight = 0.5
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    assert out["states"].shape == (2, 6, config.d_state)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_block_inner_cumsum_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 4
    config.directional_cumsum_inner_block_size = 2
    config.directional_candidate_scale = 0.01
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    assert out["states"].shape == (2, 6, config.d_state)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_block_anderson_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 3
    config.directional_candidate_scale = 0.01
    config.directional_anderson_iterations = 2
    config.directional_anderson_history_size = 2
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (1, 6))
    y = torch.randint(0, 17, (1, 6))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    assert out["states"].shape == (1, 6, config.d_state)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_superblock_cumsum_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_superblock_cumsum"
    config.directional_superblock_size = 6
    config.directional_superblock_local_size = 3
    config.directional_candidate_scale = 0.01
    config.directional_anderson_iterations = 1
    config.directional_anderson_history_size = 2
    config.directional_anderson_transition_mode = "velocity"
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (1, 6))
    y = torch.randint(0, 17, (1, 6))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    assert out["logits"].shape == (1, 6, 17)
    assert out["states"].shape == (1, 6, config.d_state)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_block_endpoint_anderson_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 3
    config.directional_candidate_scale = 0.01
    config.directional_anderson_iterations = 2
    config.directional_anderson_history_size = 2
    config.directional_anderson_transition_mode = "velocity"
    config.directional_anderson_scope = "endpoint"
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (1, 6))
    y = torch.randint(0, 17, (1, 6))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    assert out["states"].shape == (1, 6, config.d_state)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_block_consistency_loss_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 3
    config.directional_candidate_scale = 0.01
    config.lambda_block_consistency = 0.1
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (1, 6))
    y = torch.randint(0, 17, (1, 6))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    assert "block_consistency" in out["aux_losses"]
    assert torch.isfinite(out["aux_losses"]["block_consistency"])
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_sampled_block_consistency_loss_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 6
    config.directional_cumsum_step_mode = "velocity"
    config.directional_candidate_scale = 0.01
    config.lambda_sampled_block_consistency = 0.1
    config.sampled_block_consistency_interval = 1
    config.sampled_block_consistency_local_size = 3
    config.sampled_block_consistency_teacher_mode = "candidate"
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (1, 6))
    y = torch.randint(0, 17, (1, 6))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    assert "sampled_block_consistency" in out["aux_losses"]
    assert torch.isfinite(out["aux_losses"]["sampled_block_consistency"])
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_local_mixer_forward_and_backward_are_finite():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 6
    config.directional_cumsum_step_mode = "velocity"
    config.directional_candidate_scale = 0.01
    config.directional_local_mixer = "causal_conv"
    config.directional_local_mixer_hidden_size = 16
    config.directional_local_mixer_kernel_size = 3
    config.directional_local_mixer_layers = 1
    config.directional_local_mixer_scale = 0.1
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (1, 6))
    y = torch.randint(0, 17, (1, 6))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    assert out["logits"].shape == (1, 6, 17)
    assert out["states"].shape == (1, 6, config.d_state)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_directional_local_mixer_preserves_prefix_causality():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 6
    config.directional_cumsum_step_mode = "velocity"
    config.directional_candidate_scale = 0.01
    config.directional_local_mixer = "causal_conv"
    config.directional_local_mixer_hidden_size = 16
    config.directional_local_mixer_kernel_size = 3
    config.directional_local_mixer_layers = 1
    config.directional_local_mixer_scale = 0.1
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (1, 6))
    changed = x.clone()
    changed[:, 4:] = torch.randint(0, 17, (1, 2))
    out = model(x, return_states=True, collect_diagnostics=False)
    changed_out = model(changed, return_states=True, collect_diagnostics=False)
    assert torch.allclose(out["logits"][:, :4], changed_out["logits"][:, :4], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out["states"][:, :4], changed_out["states"][:, :4], atol=1e-5, rtol=1e-5)


def test_token_state_residual_is_causal_and_receives_gradients():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 6
    config.directional_cumsum_step_mode = "velocity"
    config.token_state_residual = True
    config.token_state_residual_scale = 0.2
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (1, 6))
    y = torch.randint(0, 17, (1, 6))
    changed = x.clone()
    changed[:, 4:] = torch.randint(0, 17, (1, 2))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    changed_out = model(changed, return_states=True, collect_diagnostics=False)
    assert torch.allclose(out["logits"][:, :4], changed_out["logits"][:, :4], atol=1e-5, rtol=1e-5)
    out["loss"].backward()
    assert model.token_state_residual.projection.weight.grad is not None


def test_dilated_local_mixer_and_refinement_are_causal_and_finite():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 6
    config.directional_cumsum_step_mode = "velocity"
    config.directional_local_mixer = "causal_conv"
    config.directional_local_mixer_hidden_size = 16
    config.directional_local_mixer_kernel_size = 3
    config.directional_local_mixer_layers = 2
    config.directional_local_mixer_dilation_growth = 2
    config.directional_refinement_layers = 1
    config.directional_refinement_scale = 0.1
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (1, 6))
    y = torch.randint(0, 17, (1, 6))
    changed = x.clone()
    changed[:, 4:] = torch.randint(0, 17, (1, 2))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    changed_out = model(changed, return_states=True, collect_diagnostics=False)
    assert torch.isfinite(out["loss"])
    assert torch.allclose(out["logits"][:, :4], changed_out["logits"][:, :4], atol=1e-5, rtol=1e-5)
    out["loss"].backward()
    refinement_grads = [
        parameter.grad
        for parameter in model.refinement_layers.parameters()
        if parameter.grad is not None
    ]
    assert refinement_grads
    assert all(torch.isfinite(gradient).all() for gradient in refinement_grads)


def test_optional_risk_does_not_change_other_component_initialization():
    with_risk_parameters = tiny_config()
    with_risk_parameters.seed = 7
    with_risk_parameters.directional_local_mixer = "causal_conv"
    without_risk_parameters = with_risk_parameters.validated_copy()
    without_risk_parameters.instantiate_disabled_risk = False

    first = DRMEmitterModel(with_risk_parameters)
    second = DRMEmitterModel(without_risk_parameters)
    first_state = first.state_dict()
    second_state = second.state_dict()
    common = sorted(set(first_state) & set(second_state))
    assert common
    assert all(torch.equal(first_state[name], second_state[name]) for name in common)


def test_selective_memory_is_causal_and_receives_finite_gradients():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 6
    config.directional_cumsum_step_mode = "velocity"
    config.selective_memory = True
    config.selective_memory_hidden_size = 8
    config.selective_memory_scale = 0.1
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (1, 6))
    y = torch.randint(0, 17, (1, 6))
    changed = x.clone()
    changed[:, 4:] = torch.randint(0, 17, (1, 2))
    out = model(x, y, return_states=True, collect_diagnostics=False)
    changed_out = model(changed, return_states=True, collect_diagnostics=False)
    assert torch.isfinite(out["loss"])
    assert torch.allclose(out["logits"][:, :4], changed_out["logits"][:, :4], atol=1e-5, rtol=1e-5)
    out["loss"].backward()
    memory_grads = [
        parameter.grad
        for parameter in model.selective_memory.parameters()
        if parameter.grad is not None
    ]
    assert memory_grads
    assert all(torch.isfinite(gradient).all() for gradient in memory_grads)


def test_blockwise_gate_diagnostics_use_individual_gates():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 4
    config.directional_cumsum_step_mode = "velocity"
    model = DRMEmitterModel(config)
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output[1].detach())

    handle = model.direction_field.register_forward_hook(capture)
    output = model(torch.randint(0, 17, (2, 4)), collect_diagnostics=True)
    handle.remove()
    gates = captured[0]
    diagnostics = output["diagnostics"]
    assert torch.allclose(
        diagnostics["hard_active_fraction_025"],
        (gates > 0.25).float().mean(),
    )
    assert torch.allclose(
        diagnostics["hard_active_fraction_075"],
        (gates > 0.75).float().mean(),
    )
    assert torch.allclose(diagnostics["gate_min"], gates.min())
    assert torch.allclose(diagnostics["gate_max"], gates.max())


def test_disabling_blockwise_diagnostics_does_not_change_logits():
    config = tiny_config()
    config.sequence_mode = "directional_block_cumsum"
    config.directional_cumsum_block_size = 4
    config.directional_cumsum_step_mode = "velocity"
    config.lambda_action = 0.0
    config.lambda_dim_sparsity = 0.0
    config.lambda_dim_entropy = 0.0
    config.lambda_dim_variance = 0.0
    config.lambda_metric_reg = 0.0
    config.lambda_active_fraction = 0.0
    config.lambda_condition = 0.0
    config.lambda_metric_u_floor = 0.0
    config.lambda_metric_u_target = 0.0
    config.lambda_blindspot = 0.0
    model = DRMEmitterModel(config).eval()
    tokens = torch.randint(0, 17, (2, 8))
    with_diagnostics = model(tokens, collect_diagnostics=True)["logits"]
    without_diagnostics = model(tokens, collect_diagnostics=False)["logits"]
    assert torch.equal(with_diagnostics, without_diagnostics)
