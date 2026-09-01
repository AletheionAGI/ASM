import xml.etree.ElementTree as ET
import torch
from aletheion_state_models.benchmarks.phase3a_training import Phase3ARunResult
from aletheion_state_models.benchmarks.phase3a2_plots import render_phase3a2_charts
from aletheion_state_models.benchmarks.phase3a2_summary import select_common_policy, summarize_phase3a2
from aletheion_state_models.benchmarks.phase3a2_variants import BASES, PHASE3A2_VARIANTS, RANK_ARMS, build_phase3a2_variant


def test_phase3a2_selective_core_is_parameter_matched_and_geometry_free():
    relational, _ = build_phase3a2_variant("vr_r_full", 17)
    selective, _ = build_phase3a2_variant("vr_s_full", 17)
    r_count = sum(parameter.numel() for parameter in relational.parameters())
    s_count = sum(parameter.numel() for parameter in selective.parameters())
    assert r_count == 223_814
    assert s_count == 223_738
    assert abs(s_count / r_count - 1) < .001
    assert selective.local_mixer is not None
    assert selective.token_state_residual is not None
    assert selective.selective_memory is not None
    assert selective.metric is None
    assert not selective.config.use_relational_metric
    assert not selective.config.use_metric_naturalization


def test_phase3a2_fixed_arms_have_exact_prefix_rank():
    for base in BASES:
        for arm, rank in (("full", 64), ("fixed_16", 16), ("fixed_32", 32), ("fixed_48", 48)):
            model, resolved = build_phase3a2_variant(f"{base}_{arm}", 17)
            model.eval(); output = model(torch.randint(0, 256, (2, 32)), collect_diagnostics=False)
            assert resolved == rank
            assert torch.all(output["variable_rank_ranks"] == rank)
            assert torch.all(output["variable_rank_masks"][..., :rank])
            assert not torch.any(output["variable_rank_masks"][..., rank:])


def _result(variant, seed):
    base, arm = variant[:4], variant[5:]
    rank = {"full": 64.0, "fixed_16": 16.0, "fixed_32": 32.0, "fixed_48": 48.0, "adaptive_32": 31.0}[arm]
    fixed_ce = {"full": 2.50, "fixed_16": 2.66, "fixed_32": 2.58, "fixed_48": 2.53, "adaptive_32": 2.57}[arm]
    ce = fixed_ce - (.03 if base == "vr_s" else 0)
    return Phase3ARunResult(
        variant=variant, seed=seed, steps=10, tokens_seen=1000, best_step=10,
        validation_ce=ce-.01, validation_ppl=12.0, test_ce=ce, test_ppl=12.0,
        mean_rank=rank, rank_std=3.0 if arm == "adaptive_32" else 0.0,
        rank_min=16.0 if arm == "adaptive_32" else rank,
        rank_max=64.0 if arm == "adaptive_32" else rank,
        rank_ce_correlation=.2, controller_gradient_fraction=1.0 if arm == "adaptive_32" else 0.0,
        tokens_per_second=1500.0 if base == "vr_s" else 1000.0,
        peak_memory_mb=70.0, parameter_count=223_738 if base == "vr_s" else 223_814,
        streaming_error=1e-5, finite=True,
        history=[{"step": 10.0, "tokens": 1000.0, "train_loss": ce, "validation_ce": ce-.01, "mean_rank": rank}],
    )


def test_phase3a2_summary_selection_and_charts(tmp_path):
    results = [_result(variant, seed) for seed in (17, 29, 43) for variant in PHASE3A2_VARIANTS]
    policy = select_common_policy(results)
    assert policy["common_policy"] == "full"
    summary = summarize_phase3a2(results, {"vr_r": .7, "vr_s": .7})
    assert summary["technical_passed"]
    assert summary["base_selection"]["promoted"] == "vr_s"
    render_phase3a2_charts(summary, tmp_path)
    pngs = list(tmp_path.glob("*.png")); svgs = list(tmp_path.glob("*.svg"))
    assert len(pngs) == len(svgs) == 7
    for png in pngs: assert png.read_bytes().startswith(b"\x89PNG")
    for svg in svgs: ET.parse(svg)
    assert (tmp_path / "index.html").read_text().count("<img") == 7
