import torch

from aletheion_state_models.geometry.variable_rank import (
    InputHardRankController,
    phase2_rank_curriculum,
    rank_regularization,
)
from aletheion_state_models.variants import (
    build_variable_rank_phase1,
    build_variable_rank_phase2,
)
from drm_language_emitter import DRMConfig


def _config() -> DRMConfig:
    return DRMConfig(
        vocab_size=23,
        d_token=8,
        d_state=8,
        n_directions=4,
        metric_rank=2,
        hidden_size=12,
        sequence_mode="directional_block_cumsum",
        directional_cumsum_step_mode="velocity",
        directional_cumsum_block_size=2,
        bounded_state=False,
        variable_rank_min_rank=2,
        variable_rank_target_fraction=0.5,
        variable_rank_warmup_steps=2,
        variable_rank_budget_ramp_steps=3,
        variable_rank_hardening_steps=4,
        seed=91,
    )


def test_ste_projection_is_hard_in_forward_and_soft_in_backward():
    controller = InputHardRankController(3, 4, estimator="ste", minimum_rank=1)
    controller.train()
    token = torch.randn(2, 3)
    observation = controller(token, temperature=1.5)

    assert observation.projection_mask is not None
    assert observation.soft_gates is not None
    assert torch.equal(
        observation.projection_mask.detach(), observation.active_mask.float()
    )
    assert observation.projection_mask.requires_grad

    loss = (observation.projection_mask * torch.arange(1.0, 5.0)).sum()
    loss.backward()
    assert controller.score_head.weight.grad is not None
    assert torch.isfinite(controller.score_head.weight.grad).all()
    assert torch.count_nonzero(controller.score_head.weight.grad) > 0


def test_eval_uses_only_the_hard_projection_mask():
    controller = InputHardRankController(3, 4, estimator="ste", minimum_rank=1).eval()
    observation = controller(torch.randn(2, 3))

    assert observation.projection_mask is None
    assert observation.soft_gates is not None
    assert observation.active_mask.dtype is torch.bool


def test_rank_regularization_matches_known_values_and_one_block_switch():
    gates = torch.tensor([[[0.25, 0.75], [0.75, 0.25]]], requires_grad=True)
    losses = rank_regularization(gates, target_rank=1.0)

    assert losses.budget.item() == 0.0
    assert torch.allclose(losses.binary, torch.tensor(0.1875))
    assert torch.allclose(losses.switch, torch.tensor(0.5))
    single = rank_regularization(gates[:, :1], target_rank=1.0)
    assert single.switch.item() == 0.0
    (losses.budget + losses.binary + losses.switch + single.switch).backward()
    assert torch.isfinite(gates.grad).all()


def test_rank_curriculum_has_warmup_ramp_hardening_and_steady_state():
    config = _config()
    warmup = phase2_rank_curriculum(0, config)
    ramp = phase2_rank_curriculum(3, config)
    hardening = phase2_rank_curriculum(7, config)
    steady = phase2_rank_curriculum(20, config)

    assert warmup.target_rank == config.d_state
    assert warmup.budget_weight == warmup.binary_weight == warmup.switch_weight == 0
    assert config.d_state * config.variable_rank_target_fraction < ramp.target_rank < config.d_state
    assert 0 < ramp.budget_weight < config.lambda_variable_rank_budget
    assert hardening.target_rank == steady.target_rank == 4.0
    assert hardening.temperature > steady.temperature
    assert steady.temperature == config.variable_rank_temperature_final
    assert steady.binary_weight == config.lambda_variable_rank_binary


def test_phase2_builder_preserves_phase1_parameter_contract_and_opens_gates():
    phase1 = build_variable_rank_phase1(_config())
    phase2 = build_variable_rank_phase2(_config())

    assert phase2.config.variable_rank_mode == "phase2_input_ste"
    assert phase1.state_dict().keys() == phase2.state_dict().keys()
    assert phase2.variable_rank_core.controller.estimator == "ste"
    assert torch.count_nonzero(phase2.variable_rank_core.controller.score_head.weight) == 0
    scores = torch.sigmoid(phase2.variable_rank_core.controller.score_head.bias)
    assert torch.allclose(
        scores,
        torch.full_like(scores, phase2.config.variable_rank_open_probability),
    )


def test_phase2_forward_adds_rank_losses_and_trains_controller():
    model = build_variable_rank_phase2(_config()).train()
    tokens = torch.randint(0, model.config.vocab_size, (4, 6))
    targets = torch.randint(0, model.config.vocab_size, (4, 6))
    output = model(tokens, targets=targets, global_step=6, collect_diagnostics=False)

    assert output["variable_rank_soft_gates"].shape == (4, 3, 8)
    assert output["variable_rank_masks"].dtype is torch.bool
    assert "variable_rank_budget" in output["aux_losses"]
    assert "variable_rank_regularization" in output["aux_losses"]
    output["loss"].backward()
    gradient = model.variable_rank_core.controller.score_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_phase2_training_requires_explicit_global_step():
    model = build_variable_rank_phase2(_config()).train()
    tokens = torch.randint(0, model.config.vocab_size, (2, 4))
    try:
        model(tokens, targets=tokens)
    except ValueError as error:
        assert "global_step" in str(error)
    else:
        raise AssertionError("Phase 2 training accepted a missing global_step")


def test_phase2_eval_streaming_matches_full_forward_with_hard_cache():
    model = build_variable_rank_phase2(_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 7))
    expected = model(tokens, collect_diagnostics=False)["logits"]
    first, state = model.prefill(tokens[:, :1])
    observed = [first]
    for position in range(1, tokens.shape[1]):
        step, state = model.decode_step(tokens[:, position], state)
        observed.append(step.unsqueeze(1))
        assert state.variable_rank_state.active_mask.dtype is torch.bool
        inactive = state.variable_rank_state.effective_coordinates[
            ~state.variable_rank_state.active_mask
        ]
        assert torch.count_nonzero(inactive) == 0
    assert torch.allclose(torch.cat(observed, dim=1), expected, atol=1e-6, rtol=1e-6)


def test_hard_controller_masks_are_prefix_nested():
    controller = InputHardRankController(2, 6, estimator="hard", minimum_rank=1)
    with torch.no_grad():
        controller.score_head.weight.copy_(
            torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.5, 0.0], [-0.5, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
        )
        controller.score_head.bias.zero_()
    decisions = controller(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    for mask, rank in zip(decisions.active_mask, decisions.ranks, strict=True):
        assert torch.all(mask[:rank])
        assert not torch.any(mask[rank:])
