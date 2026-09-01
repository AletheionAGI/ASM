from types import MethodType

import pytest
import torch

from aletheion_state_models.geometry.variable_rank import VariableRankBatchState
from aletheion_state_models.variants import (
    build_relational_state,
    build_variable_rank_phase1,
)
from drm_language_emitter import DRMConfig, DRMEmitterModel


def _base_config(*, block_size: int = 2) -> DRMConfig:
    return DRMConfig(
        vocab_size=19,
        d_token=8,
        d_state=8,
        n_directions=4,
        metric_rank=2,
        hidden_size=12,
        sequence_mode="directional_block_cumsum",
        directional_cumsum_step_mode="velocity",
        directional_cumsum_block_size=block_size,
        bounded_state=False,
        seed=77,
    )


def _fixed_prefix_mask(model: DRMEmitterModel, rank: int) -> None:
    controller = model.variable_rank_core.controller
    with torch.no_grad():
        controller.score_head.weight.zero_()
        controller.score_head.bias.fill_(-20.0)
        controller.score_head.bias[:rank].fill_(20.0)


def _paired_initial(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            [1.0, 2.0, 3.0, -10.0, -20.0, -30.0, -40.0, -50.0],
        ],
        dtype=dtype,
    )


def test_phase1_builder_is_opt_in_and_disables_every_declared_bypass():
    base = _base_config()
    model = build_variable_rank_phase1(base)

    assert base.variable_rank_mode == "off"
    assert model.config.variable_rank_mode == "phase1_input_hard"
    assert model.local_mixer is None
    assert model.token_state_residual is None
    assert model.selective_memory is None
    assert model.addressable_memory is None
    assert len(model.refinement_layers) == 0
    assert model.config.compact_streaming_inference


def test_phase1_config_rejects_bypass_and_non_block_modes():
    data = _base_config().to_dict()
    data.update(
        variable_rank_mode="phase1_input_hard",
        compact_streaming_inference=True,
        use_direction_field=False,
    )
    data["token_state_residual"] = True
    with pytest.raises(ValueError, match="forbids"):
        DRMConfig.from_dict(data)

    data["token_state_residual"] = False
    data["sequence_mode"] = "directional_cumsum"
    with pytest.raises(ValueError, match="directional_block_cumsum"):
        DRMConfig.from_dict(data)


def test_default_off_does_not_add_parameters_or_change_logits():
    config = _base_config()
    implicit = build_relational_state(config).eval()
    explicit_config = DRMConfig.from_dict(config.to_dict() | {"variable_rank_mode": "off"})
    explicit = build_relational_state(explicit_config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 5))

    assert implicit.variable_rank_core is None
    assert explicit.variable_rank_core is None
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    assert torch.equal(
        implicit(tokens, collect_diagnostics=False)["logits"],
        explicit(tokens, collect_diagnostics=False)["logits"],
    )


def test_full_rank_phase1_matches_no_bypass_asm_r_forward():
    base = _base_config()
    asm_r = build_relational_state(base).eval()
    phase1 = build_variable_rank_phase1(
        DRMConfig.from_dict(base.to_dict() | {"variable_rank_threshold": 0.0})
    ).eval()
    shared = asm_r.state_dict()
    missing, unexpected = phase1.load_state_dict(shared, strict=False)
    assert all(name.startswith("variable_rank_core.") for name in missing)
    assert not unexpected
    tokens = torch.randint(0, base.vocab_size, (2, 5))

    expected = asm_r(tokens, collect_diagnostics=False)["logits"]
    observed = phase1(tokens, collect_diagnostics=False)["logits"]

    assert torch.equal(observed, expected)


def test_real_asm_r_logits_forget_discarded_initial_coordinates():
    model = build_variable_rank_phase1(_base_config()).eval()
    _fixed_prefix_mask(model, 3)
    tokens = torch.randint(0, model.config.vocab_size, (1, 6)).expand(2, -1)
    embeddings = model.token_embedding(tokens)

    output = model._forward_directional_cumsum(
        _paired_initial(), embeddings, None, True, None, False
    )

    assert torch.equal(output["logits"][0], output["logits"][1])
    assert torch.equal(output["states"][0], output["states"][1])
    assert torch.equal(output["variable_rank_masks"][0], output["variable_rank_masks"][1])


def test_real_future_logits_have_zero_jacobian_on_discarded_complement():
    model = build_variable_rank_phase1(_base_config()).eval()
    _fixed_prefix_mask(model, 3)
    tokens = torch.randint(0, model.config.vocab_size, (1, 4))
    embeddings = model.token_embedding(tokens).detach()

    def logits_from_initial(initial: torch.Tensor) -> torch.Tensor:
        output = model._forward_directional_cumsum(
            initial.unsqueeze(0), embeddings, None, False, None, False
        )
        return output["logits"].flatten()

    point = torch.randn(model.config.d_state, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(logits_from_initial, point, vectorize=True)

    assert torch.count_nonzero(jacobian[:, 3:]).item() == 0


def test_compact_prefill_decode_matches_forward_and_never_uses_fallback():
    model = build_variable_rank_phase1(_base_config(block_size=3)).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    expected = model(tokens, collect_diagnostics=False)["logits"]
    first, state = model.prefill(tokens[:, :2])
    observed = [first]

    def forbidden_prefill(*_args, **_kwargs):
        raise AssertionError("decode attempted the full-prefix fallback")

    model.prefill = forbidden_prefill
    for position in range(2, tokens.shape[1]):
        step, state = model.decode_step(tokens[:, position], state)
        observed.append(step.unsqueeze(1))
        assert state.uses_block_cache
        assert state.completed_state is None
        assert state.input_ids.numel() == 0
        assert isinstance(state.variable_rank_state, VariableRankBatchState)
        cached = state.variable_rank_state
        assert torch.count_nonzero(cached.effective_coordinates[~cached.active_mask]) == 0
        assert state.block_tokens.shape[1] < state.block_size

    assert torch.allclose(torch.cat(observed, dim=1), expected, atol=1e-6, rtol=1e-6)


def test_real_cache_discards_pair_difference_before_reexpansion():
    model = build_variable_rank_phase1(_base_config(block_size=2)).eval()
    _fixed_prefix_mask(model, 3)
    initial = _paired_initial()

    def paired_initializer(_module, batch_size: int, _device: torch.device) -> torch.Tensor:
        assert batch_size == 2
        return initial.clone()

    model.initializer.forward = MethodType(paired_initializer, model.initializer)
    tokens = torch.randint(0, model.config.vocab_size, (1, 4)).expand(2, -1)
    first_logits, state = model.prefill(tokens[:, :1])

    assert torch.equal(first_logits[0], first_logits[1])
    assert torch.equal(
        state.variable_rank_state.effective_coordinates[0],
        state.variable_rank_state.effective_coordinates[1],
    )
    closed_logits, state = model.decode_step(tokens[:, 1], state)
    assert torch.equal(closed_logits[0], closed_logits[1])

    _fixed_prefix_mask(model, model.config.d_state)
    for position in (2, 3):
        logits, state = model.decode_step(tokens[:, position], state)
        assert torch.equal(logits[0], logits[1])
        assert torch.equal(
            state.variable_rank_state.effective_coordinates[0],
            state.variable_rank_state.effective_coordinates[1],
        )


def test_controller_is_per_example_and_independent_of_recurrent_state():
    model = build_variable_rank_phase1(_base_config()).eval()
    controller = model.variable_rank_core.controller
    first_tokens = torch.stack((torch.zeros(8), torch.ones(8)))
    with torch.no_grad():
        controller.score_head.weight.fill_(0.25)
        controller.score_head.bias.copy_(torch.linspace(-1.0, 1.0, 8))
    decision = controller(first_tokens)

    assert decision.active_mask.dtype is torch.bool
    assert torch.equal(decision.ranks, decision.active_mask.sum(dim=-1))
    assert not torch.equal(decision.active_mask[0], decision.active_mask[1])


def test_future_tokens_and_diagnostics_do_not_change_past_phase1_outputs():
    model = build_variable_rank_phase1(_base_config()).eval()
    prefix = torch.randint(0, model.config.vocab_size, (2, 4))
    first_suffix = torch.randint(0, model.config.vocab_size, (2, 2))
    second_suffix = torch.randint(0, model.config.vocab_size, (2, 2))
    first_tokens = torch.cat((prefix, first_suffix), dim=1)
    second_tokens = torch.cat((prefix, second_suffix), dim=1)

    first = model(first_tokens, collect_diagnostics=True)
    second = model(second_tokens, collect_diagnostics=False)
    no_diagnostics = model(first_tokens, collect_diagnostics=False)

    assert torch.equal(first["logits"][:, : prefix.shape[1]], second["logits"][:, : prefix.shape[1]])
    assert torch.equal(
        first["variable_rank_masks"][:, : prefix.shape[1]],
        second["variable_rank_masks"][:, : prefix.shape[1]],
    )
    assert torch.equal(first["logits"], no_diagnostics["logits"])
