from types import MethodType

import pytest
import torch

from aletheion_state_models.variants import (
    build_relational_state,
    build_variable_rank_phase3a1,
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
        directional_local_mixer="causal_conv",
        directional_local_mixer_hidden_size=12,
        directional_local_mixer_kernel_size=2,
        token_state_residual=True,
        selective_memory=True,
        selective_memory_hidden_size=12,
        bounded_state=False,
        dropout=0.0,
        seed=123,
    )


def _fix_rank(model, rank: int) -> None:
    controller = model.variable_rank_core.controller
    with torch.no_grad():
        controller.score_head.weight.zero_()
        controller.score_head.bias.fill_(-20.0)
        controller.score_head.bias[:rank].fill_(20.0)


def test_phase3a1_builder_enables_only_requested_projected_routes():
    model = build_variable_rank_phase3a1(
        _config(), mixer=True, residual=False, selective_memory=True
    )

    assert model.config.variable_rank_mode == "phase3a1_projected"
    assert model.config.variable_rank_scaffold_projection
    assert model.local_mixer is not None
    assert model.token_state_residual is None
    assert model.selective_memory is not None
    assert model.addressable_memory is None


def test_projected_scaffold_flag_is_rejected_without_variable_rank():
    data = _config().to_dict() | {"variable_rank_scaffold_projection": True}
    with pytest.raises(ValueError, match="requires variable rank"):
        DRMConfig.from_dict(data)


def test_full_rank_projected_scaffold_matches_same_asm_r_components():
    base = _config()
    asm_r = build_relational_state(base).eval()
    projected = build_variable_rank_phase3a1(
        base, mixer=True, residual=True, selective_memory=True
    ).eval()
    missing, unexpected = projected.load_state_dict(asm_r.state_dict(), strict=False)
    assert all(name.startswith("variable_rank_core.") for name in missing)
    assert not unexpected
    _fix_rank(projected, base.d_state)
    tokens = torch.randint(0, base.vocab_size, (2, 6))

    expected = asm_r(tokens, collect_diagnostics=False)["logits"]
    actual = projected(tokens, collect_diagnostics=False)["logits"]

    assert torch.equal(actual, expected)



def test_full_rank_projected_selective_core_matches_asm_s():
    from aletheion_state_models.variants import build_selective_state
    base = _config()
    asm_s = build_selective_state(base, memory_hidden_size=12).eval()
    projected = build_variable_rank_phase3a1(
        base,
        mixer=True,
        residual=True,
        selective_memory=True,
        relational_core=False,
    ).eval()
    missing, unexpected = projected.load_state_dict(asm_s.state_dict(), strict=False)
    assert all(name.startswith("variable_rank_core.") for name in missing)
    assert not unexpected
    _fix_rank(projected, base.d_state)
    tokens = torch.randint(0, base.vocab_size, (2, 6))
    assert torch.equal(
        projected(tokens, collect_diagnostics=False)["logits"],
        asm_s(tokens, collect_diagnostics=False)["logits"],
    )
    assert not projected.config.use_relational_metric
    assert not projected.config.use_metric_naturalization


def test_full_rank_projected_relational_selective_core_matches_asm_rs():
    from aletheion_state_models.variants import build_relational_selective_state
    base = _config()
    asm_rs = build_relational_selective_state(base, memory_hidden_size=12).eval()
    projected = build_variable_rank_phase3a1(
        base, mixer=True, residual=True, selective_memory=True,
        relational_core=True,
    ).eval()
    missing, unexpected = projected.load_state_dict(asm_rs.state_dict(), strict=False)
    assert all(name.startswith("variable_rank_core.") for name in missing)
    assert not unexpected
    _fix_rank(projected, base.d_state)
    tokens = torch.randint(0, base.vocab_size, (2, 6))
    assert torch.equal(
        projected(tokens, collect_diagnostics=False)["logits"],
        asm_rs(tokens, collect_diagnostics=False)["logits"],
    )

def test_projection_cleans_sentinels_between_every_scaffold_component():
    model = build_variable_rank_phase3a1(
        _config(), mixer=True, residual=True, selective_memory=True
    ).eval()
    _fix_rank(model, 4)
    inactive = torch.tensor([False, False, False, False, True, True, True, True])
    checks = []

    def inject(_module, _inputs, output):
        changed = output.clone()
        changed[..., inactive] = 1_000.0
        return changed

    def mixer_input(_module, inputs):
        checks.append("mixer")
        assert torch.count_nonzero(inputs[0][..., inactive]) == 0
        assert torch.count_nonzero(inputs[1][..., inactive]) == 0
        assert torch.count_nonzero(inputs[3][..., inactive]) == 0

    def residual_input(_module, inputs):
        checks.append("residual")
        assert torch.count_nonzero(inputs[0][..., inactive]) == 0

    def selective_input(_module, inputs):
        checks.append("selective")
        assert torch.count_nonzero(inputs[1][..., inactive]) == 0

    handles = [
        model.local_mixer.register_forward_pre_hook(mixer_input),
        model.local_mixer.register_forward_hook(inject),
        model.token_state_residual.register_forward_pre_hook(residual_input),
        model.token_state_residual.register_forward_hook(inject),
        model.selective_memory.register_forward_pre_hook(selective_input),
        model.selective_memory.register_forward_hook(inject),
    ]
    try:
        output = model(torch.randint(0, model.config.vocab_size, (2, 6)), return_states=True, collect_diagnostics=False)
    finally:
        for handle in handles:
            handle.remove()

    assert checks == ["mixer", "residual", "selective"] * 3
    assert torch.count_nonzero(output["states"][..., inactive]) == 0


def test_projected_scaffold_forgets_discarded_initial_pair_and_jacobian():
    model = build_variable_rank_phase3a1(
        _config(), mixer=True, residual=True, selective_memory=True
    ).eval()
    _fix_rank(model, 3)
    tokens = torch.randint(0, model.config.vocab_size, (1, 6)).expand(2, -1)
    embeddings = model.token_embedding(tokens)
    initial = torch.tensor(
        [[1, 2, 3, 10, 20, 30, 40, 50], [1, 2, 3, -10, -20, -30, -40, -50]],
        dtype=torch.float32,
    )
    output = model._forward_directional_cumsum(initial, embeddings, None, True, None, False)
    assert torch.equal(output["logits"][0], output["logits"][1])
    assert torch.equal(output["states"][0], output["states"][1])

    def future_logits(point):
        result = model._forward_directional_cumsum(
            point.unsqueeze(0), embeddings[:1], None, False, None, False
        )
        return result["logits"].flatten()

    point = initial[0].detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(future_logits, point, vectorize=True)
    assert torch.count_nonzero(jacobian[:, 3:]) == 0
    assert torch.count_nonzero(jacobian[:, :3]) > 0


def test_projected_scaffold_streaming_cache_matches_forward():
    model = build_variable_rank_phase3a1(
        _config(), mixer=True, residual=True, selective_memory=True
    ).eval()
    _fix_rank(model, 4)
    tokens = torch.randint(0, model.config.vocab_size, (2, 7))
    expected = model(tokens, collect_diagnostics=False)["logits"]
    first, state = model.prefill(tokens[:, :1])
    observed = [first]
    for position in range(1, tokens.shape[1]):
        current, state = model.decode_step(tokens[:, position], state)
        observed.append(current.unsqueeze(1))
        cached = state.variable_rank_state
        assert torch.count_nonzero(cached.effective_coordinates[~cached.active_mask]) == 0
    assert torch.allclose(torch.cat(observed, dim=1), expected, atol=1e-6, rtol=1e-6)



def test_all_stage_a_factorial_arms_resolve_the_requested_modules():
    from aletheion_state_models.benchmarks.phase3a1_variants import (
        STAGE_A_COMPONENTS,
        build_stage_a_variant,
    )
    for name, expected in STAGE_A_COMPONENTS.items():
        model, rank = build_stage_a_variant(name, 17)
        actual = (
            model.local_mixer is not None,
            model.token_state_residual is not None,
            model.selective_memory is not None,
        )
        assert actual == expected
        assert rank == 64

def _synthetic_result(variant: str, seed: int, ce: float, rank: float = 64.0):
    from aletheion_state_models.benchmarks.phase3a_training import Phase3ARunResult
    return Phase3ARunResult(
        variant=variant, seed=seed, steps=2, tokens_seen=128, best_step=2,
        validation_ce=ce, validation_ppl=20.0, test_ce=ce + 0.01, test_ppl=20.0,
        mean_rank=rank, rank_std=2.0 if "adaptive" in variant else 0.0,
        rank_min=16.0 if "adaptive" in variant else rank,
        rank_max=64.0 if "adaptive" in variant else rank,
        rank_ce_correlation=0.2, controller_gradient_fraction=1.0 if "adaptive" in variant else 0.0,
        tokens_per_second=1000.0, peak_memory_mb=10.0, parameter_count=100,
        streaming_error=1e-6, finite=True,
        history=[{"step": 2.0, "tokens": 128.0, "train_loss": ce, "validation_ce": ce, "mean_rank": rank}],
    )


def test_phase3a1_summaries_select_scaffold_and_separate_scientific_gates(tmp_path):
    from aletheion_state_models.benchmarks.phase3a1_plots import render_stage_a, render_stage_b
    from aletheion_state_models.benchmarks.phase3a1_summary import summarize_stage_a, summarize_stage_b
    from aletheion_state_models.benchmarks.phase3a1_variants import STAGE_A_COMPONENTS, STAGE_B_VARIANTS
    stage_a_results = []
    for seed in (17, 29, 43):
        for name, flags in STAGE_A_COMPONENTS.items():
            stage_a_results.append(_synthetic_result(name, seed, 3.0 - 0.1 * sum(flags)))
    stage_a = summarize_stage_a(stage_a_results)
    assert stage_a["selection"]["variant"] == "all_projected"
    assert stage_a["gates"]["quality_recovery"]
    stage_b_results = []
    ranks = {"selected_full": 64.0, "selected_fixed_16": 16.0, "selected_fixed_32": 32.0, "selected_fixed_48": 48.0, "selected_adaptive_32": 30.0}
    ces = {"selected_full": 3.0, "selected_fixed_16": 3.2, "selected_fixed_32": 3.1, "selected_fixed_48": 3.05, "selected_adaptive_32": 3.05}
    for seed in (17, 29, 43):
        for name in STAGE_B_VARIANTS:
            stage_b_results.append(_synthetic_result(name, seed, ces[name], ranks[name]))
    stage_b = summarize_stage_b(stage_b_results, stage_a["selection"], 0.8)
    assert stage_b["operational_passed"]
    assert "adaptive_frontier_advantage" in stage_b["gates"]
    render_stage_a(stage_a, tmp_path / "a"); render_stage_b(stage_b, tmp_path / "b")
    assert (tmp_path / "a/index.html").exists()
    assert (tmp_path / "b/quality_vs_mean_rank.png").exists()
