import torch
from aletheion_state_models.benchmarks.purpose_variants import (
    PURPOSE_VARIANTS, build_purpose_variant, parameter_inventory,
)


def test_pmcs64_parameter_match_is_within_preregistered_tolerance():
    inventory = parameter_inventory()
    cm_total = inventory["asm_cm"]["total"]
    assert max(abs(row["total"] / cm_total - 1.0) for row in inventory.values()) <= 0.001
    assert inventory["asm_vr_s_full"]["trainable"] == inventory["asm_vr_s_fixed_32"]["trainable"]


def test_pmcs64_builders_have_expected_memory_and_rank_policies():
    cm, _ = build_purpose_variant("asm_cm", 17)
    full, _ = build_purpose_variant("asm_vr_s_full", 17)
    fixed, _ = build_purpose_variant("asm_vr_s_fixed_32", 17)
    assert cm.config.addressable_memory_backend == "fast_weight"
    assert cm.config.fast_weight_durable_memory
    assert not full.config.use_relational_metric and not fixed.config.use_relational_metric
    assert all(not parameter.requires_grad for parameter in full.variable_rank_core.controller.parameters())
    assert all(not parameter.requires_grad for parameter in fixed.variable_rank_core.controller.parameters())


def test_pmcs64_full_and_fixed_masks_are_exact_and_fixed_projection_has_no_bypass():
    for variant, expected_rank in (("asm_vr_s_full", 64), ("asm_vr_s_fixed_32", 32)):
        model, _ = build_purpose_variant(variant, 17); model.eval()
        controller = model.variable_rank_core.controller
        decision = controller(torch.randn(2, 64))
        assert torch.all(decision.ranks == expected_rank)
        projected = model.variable_rank_core.project(torch.ones(2, 64), decision.active_mask)
        assert torch.all(projected[:, expected_rank:] == 0)


def test_pmcs64_stream_summary_preserves_failed_long_probe():
    from aletheion_state_models.benchmarks.purpose_summary import _stream
    rows = [{"variant": "x", "seed": 17, "streaming": [
        {"length": 4096, "seed": 17, "tokens_per_second": 10.0, "retained_state_bytes": 320, "cuda_peak_mb": 1.0, "latency_ms_p95": 2.0},
        {"length": 32768, "seed": 17, "status": "failed", "failed_at_position": 5000, "error": "non-finite"},
    ]}]
    summary = _stream(rows)
    assert summary["by_length"]["32768"]["successful"] == 0
    assert summary["by_length"]["32768"]["tokens_per_second_mean"] is None
