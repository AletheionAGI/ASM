"""Aggregation and acceptance gates for ASM-VR Phase 3A."""

from __future__ import annotations

from dataclasses import asdict
from statistics import mean, pstdev

from .phase3a_training import Phase3ARunResult
from .phase3a_variants import PHASE3A_VARIANTS


def summarize_phase3a(results: list[Phase3ARunResult]) -> dict[str, object]:
    """Aggregate paired seeds and evaluate the frozen small-scale gates."""
    grouped: dict[str, list[Phase3ARunResult]] = {}
    for result in results:
        grouped.setdefault(result.variant, []).append(result)
    variants = {}
    for variant in PHASE3A_VARIANTS:
        rows = grouped.get(variant, [])
        if not rows:
            continue
        variants[variant] = {
            "runs": len(rows),
            "validation_ce_mean": mean(row.validation_ce for row in rows),
            "validation_ce_std": pstdev(row.validation_ce for row in rows),
            "test_ce_mean": mean(row.test_ce for row in rows),
            "test_ce_std": pstdev(row.test_ce for row in rows),
            "mean_rank": mean(row.mean_rank for row in rows),
            "rank_std_mean": mean(row.rank_std for row in rows),
            "rank_ce_correlation_mean": mean(
                row.rank_ce_correlation for row in rows
            ),
            "tokens_per_second_mean": mean(row.tokens_per_second for row in rows),
            "peak_memory_mb_mean": mean(row.peak_memory_mb for row in rows),
            "parameter_count": rows[0].parameter_count,
        }
    adaptive = sorted(grouped.get("vr_adaptive_32", []), key=lambda row: row.seed)
    fixed = sorted(grouped.get("vr_fixed_32", []), key=lambda row: row.seed)
    paired_quality = []
    if len(adaptive) == len(fixed):
        paired_quality = [
            {
                "seed": left.seed,
                "adaptive_minus_fixed32_test_ce": left.test_ce - right.test_ce,
            }
            for left, right in zip(adaptive, fixed, strict=True)
            if left.seed == right.seed
        ]
    expected_tokens = results[0].tokens_seen if results else 0
    gates = {
        "complete_matrix": len(results) == len(PHASE3A_VARIANTS) * 3
        and all(len(grouped.get(variant, [])) == 3 for variant in PHASE3A_VARIANTS),
        "all_runs_finite": bool(results) and all(row.finite for row in results),
        "matched_token_budget": bool(results)
        and all(row.tokens_seen == expected_tokens for row in results),
        "streaming_parity": bool(results)
        and all(row.streaming_error <= 0.05 for row in results),
        "adaptive_budget": bool(adaptive)
        and all(28.8 <= row.mean_rank <= 35.2 for row in adaptive),
        "adaptive_varies": bool(adaptive)
        and all(row.rank_std > 1.0 for row in adaptive),
        "controller_receives_gradient": bool(adaptive)
        and all(row.controller_gradient_fraction >= 0.9 for row in adaptive),
        "quality_near_fixed32": len(paired_quality) == 3
        and mean(item["adaptive_minus_fixed32_test_ce"] for item in paired_quality)
        <= 0.05
        and all(
            item["adaptive_minus_fixed32_test_ce"] <= 0.10
            for item in paired_quality
        ),
        "capacity_below_full": bool(adaptive)
        and all(row.mean_rank <= 0.8 * 64 for row in adaptive),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "variants": variants,
        "paired_quality": paired_quality,
        "runs": [asdict(result) for result in results],
        "claims": {
            "small_scale_language_gate": True,
            "scaling_confirmation": False,
            "hardware_speedup": False,
        },
    }


__all__ = ["summarize_phase3a"]
