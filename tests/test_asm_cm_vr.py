import torch
import pytest
from drm_language_emitter import DRMConfig
from drm_language_emitter.fast_weight_memory import FastWeightMemoryState
from drm_language_emitter.rank_aware_memory import apply_rank_aware_memory
from aletheion_state_models.benchmarks.phase3a_training import measure_streaming_error
from aletheion_state_models.benchmarks.phase3a_variants import phase3a_config
from aletheion_state_models.variants import (
    build_compact_durable_fast_weight,
    build_compact_memory_adaptive_rank,
    build_compact_memory_variable_rank,
)


def _observation(model, batch=2):
    tokens = torch.randn(batch, 4, model.config.d_token)
    return model.variable_rank_core.observe_block(tokens)


def test_cm_vr_builder_freezes_exact_fixed32_and_aligns_value_payload():
    model = build_compact_memory_variable_rank(phase3a_config(17))
    observation = _observation(model)
    assert torch.all(observation.ranks == 32)
    assert model.addressable_memory.d_value == model.config.d_state == 64
    assert model.addressable_memory.d_key == model.config.addressable_memory_dim
    assert model.config.variable_rank_memory_policy == "project_io"
    assert model.config.fast_weight_durable_memory
    assert all(not parameter.requires_grad for parameter in model.variable_rank_core.controller.parameters())


def test_cm_vr_config_keeps_unprojected_memory_forbidden():
    config = build_compact_memory_variable_rank(phase3a_config(17)).config.to_dict()
    config["variable_rank_memory_policy"] = "forbid"
    with pytest.raises(ValueError, match="unprojected addressable memory"):
        DRMConfig.from_dict(config)
    config["variable_rank_memory_policy"] = "project_io"
    config["addressable_memory_backend"] = "slots"
    with pytest.raises(ValueError, match="state-aligned fast-weight"):
        DRMConfig.from_dict(config)


def test_rank_aware_memory_removes_inactive_state_and_payload_bypass():
    torch.manual_seed(4); model = build_compact_memory_variable_rank(phase3a_config(17)).eval(); memory = model.addressable_memory
    with torch.no_grad(): memory.read_output.weight.normal_()
    states = torch.randn(2, 3, 64); altered_states = states.clone(); altered_states[..., 32:] += 50
    tokens = torch.randn(2, 3, 64); initial = memory.initial_state(2, states.device, states.dtype)
    altered_memory = FastWeightMemoryState(initial.matrix.clone(), initial.consolidated.clone(), initial.previous_token.clone())
    altered_memory.matrix[..., 32:] = 30; altered_memory.consolidated[..., 32:] = -40
    observation = _observation(model)
    first = apply_rank_aware_memory(memory, states, tokens, initial, variable_rank_core=model.variable_rank_core, rank_observation=observation)
    second = apply_rank_aware_memory(memory, altered_states, tokens, altered_memory, variable_rank_core=model.variable_rank_core, rank_observation=observation)
    assert torch.equal(first[0], second[0]); assert torch.equal(first[1].matrix, second[1].matrix); assert torch.equal(first[1].consolidated, second[1].consolidated)
    assert torch.count_nonzero(first[0][..., 32:]) == 0
    assert torch.count_nonzero(first[1].matrix[..., 32:]) == 0
    assert torch.count_nonzero(first[1].consolidated[..., 32:]) == 0
    raw_first = memory.forward_sequence(states, tokens, initial)[1]
    raw_second = memory.forward_sequence(altered_states, tokens, altered_memory)[1]
    assert not torch.equal(raw_first.matrix, raw_second.matrix)


def test_cm_vr_inactive_state_and_memory_jacobians_are_zero():
    torch.manual_seed(5); model = build_compact_memory_variable_rank(phase3a_config(17)).eval(); memory = model.addressable_memory
    states = torch.randn(1, 2, 64, requires_grad=True); tokens = torch.randn(1, 2, 64); base = memory.initial_state(1, states.device, states.dtype)
    matrix = base.matrix.detach().requires_grad_(True); consolidated = base.consolidated.detach().requires_grad_(True); initial = FastWeightMemoryState(matrix, consolidated, base.previous_token)
    output, next_memory, _ = apply_rank_aware_memory(memory, states, tokens, initial, variable_rank_core=model.variable_rank_core, rank_observation=_observation(model,1))
    objective = output[..., :32].sum() + next_memory.matrix[..., :32].sum() + next_memory.consolidated[..., :32].sum()
    state_grad, matrix_grad, consolidated_grad = torch.autograd.grad(objective, (states, matrix, consolidated))
    assert torch.count_nonzero(state_grad[..., 32:]) == 0
    assert torch.count_nonzero(matrix_grad[..., 32:]) == 0
    assert torch.count_nonzero(consolidated_grad[..., 32:]) == 0


def test_cm_vr_full64_matches_state_aligned_cm_control():
    base = phase3a_config(17); data = base.to_dict() | {"addressable_memory_value_dim": base.d_state}; aligned = DRMConfig.from_dict(data)
    legacy = build_compact_durable_fast_weight(aligned).eval(); full = build_compact_memory_variable_rank(base, fixed_rank=64).eval()
    shared = {name: value for name, value in full.state_dict().items() if name in legacy.state_dict()}; legacy.load_state_dict(shared, strict=True)
    assert sum(p.numel() for p in legacy.parameters()) == sum(p.numel() for p in full.parameters() if p.requires_grad)
    tokens = torch.randint(0, base.vocab_size, (2, 65))
    with torch.no_grad(): legacy_logits = legacy(tokens, collect_diagnostics=False)["logits"]; full_logits = full(tokens, collect_diagnostics=False)["logits"]
    assert torch.allclose(legacy_logits, full_logits, atol=1e-6, rtol=1e-6)


def test_cm_vr_fixed32_streaming_parity_and_projected_cache():
    model = build_compact_memory_variable_rank(phase3a_config(17)).eval()
    corpus = torch.randint(0, model.config.vocab_size, (2048,), dtype=torch.uint8)
    error = measure_streaming_error(model, corpus, torch.device("cpu"))
    tokens = torch.randint(0, model.config.vocab_size, (1, 65))
    assert error <= 1e-5
    with torch.no_grad(): _, state = model.prefill(tokens[:, :33])
    for position in range(33, 65):
        with torch.no_grad(): _, state = model.decode_step(tokens[0, position : position + 1], state)
    assert torch.count_nonzero(state.addressable_memory.matrix[..., 32:]) == 0
    assert torch.count_nonzero(state.addressable_memory.consolidated[..., 32:]) == 0
    assert torch.count_nonzero(state.variable_rank_state.effective_coordinates[..., 32:]) == 0


def test_cm_vr_memory_shrink_then_grow_cannot_restore_discarded_payload():
    model = build_compact_memory_variable_rank(phase3a_config(17)).eval(); memory = model.addressable_memory
    state = torch.randn(1, 64); token = torch.randn(1, 64); first = memory.initial_state(1, state.device, state.dtype)
    second = FastWeightMemoryState(first.matrix.clone(), first.consolidated.clone(), first.previous_token.clone())
    second.matrix[..., 32:] = 1e6; second.consolidated[..., 32:] = -1e6
    mask32 = torch.arange(64).unsqueeze(0) < 32; mask64 = torch.ones(1, 64, dtype=torch.bool)
    out_a, memory_a, _ = memory.step(state, token, first, value_mask=mask32)
    out_b, memory_b, _ = memory.step(state, token, second, value_mask=mask32)
    assert torch.equal(out_a, out_b); assert torch.equal(memory_a.matrix, memory_b.matrix)
    future = torch.randn(1, 64)
    grown_a = memory.step(out_a, future, memory_a, value_mask=mask64)
    grown_b = memory.step(out_b, future, memory_b, value_mask=mask64)
    assert torch.equal(grown_a[0], grown_b[0]); assert torch.equal(grown_a[1].matrix, grown_b[1].matrix); assert torch.equal(grown_a[1].consolidated, grown_b[1].consolidated)


def test_cm_vr_full_stream_parity_crosses_each_block_boundary():
    model = build_compact_memory_variable_rank(phase3a_config(17)).eval(); tokens = torch.randint(0, model.config.vocab_size, (1, 65))
    with torch.no_grad(): expected = model(tokens, collect_diagnostics=False)["logits"]
    for prompt in (1, 31, 32, 33):
        with torch.no_grad(): prefix, state = model.prefill(tokens[:, :prompt])
        observed = [prefix]
        for position in range(prompt, tokens.shape[1]):
            with torch.no_grad(): current, state = model.decode_step(tokens[:, position], state)
            observed.append(current.unsqueeze(1))
        actual = torch.cat(observed, dim=1)
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cm_vr_adaptive_builder_keeps_controller_trainable_and_memory_strict():
    model = build_compact_memory_adaptive_rank(phase3a_config(17), target_rank=32)
    observation = _observation(model)
    assert torch.all(observation.ranks == 64)
    assert all(parameter.requires_grad for parameter in model.variable_rank_core.controller.parameters())
    assert model.config.variable_rank_target_fraction == .5
    assert model.config.variable_rank_memory_policy == "project_io"
    assert model.config.lambda_variable_rank_budget > 0
