"""Scientific aggregation for the ASM-VR-R versus ASM-VR-S experiment."""
from __future__ import annotations
from dataclasses import asdict
from statistics import mean, pstdev
from .phase3a2_variants import BASES, PHASE3A2_VARIANTS, RANK_ARMS


def _aggregate(runs):
    output = {"runs": [asdict(run) for run in runs], "seeds": [run.seed for run in runs]}
    for field in ("validation_ce", "test_ce", "mean_rank", "rank_std", "rank_ce_correlation", "controller_gradient_fraction", "tokens_per_second", "peak_memory_mb", "parameter_count", "streaming_error"):
        values = [float(getattr(run, field)) for run in runs]
        output[f"{field}_mean"] = mean(values); output[f"{field}_std"] = pstdev(values)
    return output


def _fixed_frontier(variants, base, rank):
    points = sorted((variants[f"{base}_{arm}"]["mean_rank_mean"], variants[f"{base}_{arm}"]["test_ce_mean"]) for arm in RANK_ARMS if arm != "adaptive_32")
    for (left_rank, left_ce), (right_rank, right_ce) in zip(points, points[1:]):
        if left_rank <= rank <= right_rank:
            weight = (rank - left_rank) / (right_rank - left_rank)
            return left_ce + weight * (right_ce - left_ce)
    return min(points, key=lambda point: abs(point[0] - rank))[1]


def _adaptive_gates(variants, grouped, base):
    adaptive = variants[f"{base}_adaptive_32"]; fixed = variants[f"{base}_fixed_32"]
    adaptive_runs = grouped[f"{base}_adaptive_32"]; fixed_by_seed = {run.seed: run for run in grouped[f"{base}_fixed_32"]}
    deltas = [{"seed": run.seed, "adaptive_minus_fixed32_test_ce": run.test_ce - fixed_by_seed[run.seed].test_ce} for run in adaptive_runs]
    frontier = _fixed_frontier(variants, base, adaptive["mean_rank_mean"]); gap = adaptive["test_ce_mean"] - frontier
    less_rank = [variants[f"{base}_{arm}"] for arm in RANK_ARMS if arm != "adaptive_32" and variants[f"{base}_{arm}"]["mean_rank_mean"] <= adaptive["mean_rank_mean"]]
    gates = {
        "budget": 28.8 <= adaptive["mean_rank_mean"] <= 35.2,
        "variation": min(run.rank_std for run in adaptive_runs) > 1.0,
        "controller_gradient": min(run.controller_gradient_fraction for run in adaptive_runs) >= .9,
        "near_fixed32": mean(item["adaptive_minus_fixed32_test_ce"] for item in deltas) <= .05 and max(item["adaptive_minus_fixed32_test_ce"] for item in deltas) <= .10,
        "pareto": not any(value["test_ce_mean"] <= adaptive["test_ce_mean"] - .01 for value in less_rank),
        "frontier_advantage": gap <= -.02,
    }
    return {"gates": gates, "paired_fixed32": deltas, "fixed_frontier_ce": frontier, "adaptive_minus_frontier_ce": gap}



def select_common_policy(results):
    """Freeze one rank policy using validation only, before test is opened."""
    scores = {
        arm: mean(run.validation_ce for run in results if run.variant.endswith(f"_{arm}"))
        for arm in RANK_ARMS
    }
    ranks = {
        arm: mean(run.mean_rank for run in results if run.variant.endswith(f"_{arm}"))
        for arm in RANK_ARMS
    }
    best = min(scores.values()); candidates = [arm for arm in RANK_ARMS if scores[arm] <= best + .01]
    selected = min(candidates, key=lambda arm: ranks[arm])
    return {"common_policy": selected, "scores": scores, "mean_ranks": ranks, "rule": "within 0.01 nat of best across bases, choose lower rank"}

def summarize_phase3a2(results, thresholds):
    grouped = {name: sorted((run for run in results if run.variant == name), key=lambda run: run.seed) for name in PHASE3A2_VARIANTS}
    variants = {name: _aggregate(runs) for name, runs in grouped.items()}; comparisons = {}
    for arm in RANK_ARMS:
        r_by_seed = {run.seed: run for run in grouped[f"vr_r_{arm}"]}
        comparisons[arm] = [{"seed": run.seed, "s_minus_r_test_ce": run.test_ce - r_by_seed[run.seed].test_ce, "s_over_r_throughput": run.tokens_per_second / r_by_seed[run.seed].tokens_per_second} for run in grouped[f"vr_s_{arm}"]]
    parameter_delta = abs(variants["vr_s_full"]["parameter_count_mean"] / variants["vr_r_full"]["parameter_count_mean"] - 1)
    full_quality = mean(item["s_minus_r_test_ce"] for item in comparisons["full"])
    fixed_quality = mean(item["s_minus_r_test_ce"] for item in comparisons["fixed_32"])
    speed_ratio = mean(item["s_over_r_throughput"] for arm in RANK_ARMS for item in comparisons[arm])
    base_gates = {
        "complete_matrix": all(len(runs) == 3 for runs in grouped.values()),
        "finite": all(run.finite for run in results),
        "streaming": max(run.streaming_error for run in results) <= 1e-4,
        "parameter_match": parameter_delta <= .01,
        "s_full_quality_noninferior": full_quality <= .05 and max(item["s_minus_r_test_ce"] for item in comparisons["full"]) <= .10,
        "s_fixed32_quality_noninferior": fixed_quality <= .05 and max(item["s_minus_r_test_ce"] for item in comparisons["fixed_32"]) <= .10,
        "s_throughput_advantage": speed_ratio >= 1.25,
        "s_quality_superior": (
            full_quality <= -.02 and fixed_quality <= -.02
            and all(item["s_minus_r_test_ce"] < 0 for arm in ("full", "fixed_32") for item in comparisons[arm])
        ),
    }
    efficiency_path = base_gates["s_full_quality_noninferior"] and base_gates["s_fixed32_quality_noninferior"] and base_gates["s_throughput_advantage"]
    promote_s = base_gates["s_quality_superior"] or efficiency_path
    adaptive = {base: _adaptive_gates(variants, grouped, base) for base in BASES}
    policy_selection = select_common_policy(results)
    technical = ("complete_matrix", "finite", "streaming", "parameter_match")
    return {
        "experiment": "3A.2", "variants": variants, "thresholds": thresholds,
        "paired_base_comparisons": comparisons,
        "base_effects": {"s_minus_r_full_test_ce": full_quality, "s_minus_r_fixed32_test_ce": fixed_quality, "s_over_r_throughput": speed_ratio, "parameter_fraction_delta": parameter_delta},
        "base_gates": base_gates, "adaptive": adaptive,
        "validation_policy_selection": policy_selection,
        "base_selection": {"promoted": "vr_s" if promote_s else "vr_r", "quality_path": base_gates["s_quality_superior"], "efficiency_path": efficiency_path, "rule": "promote S by >=0.02 nat consistent quality gain, or noninferior quality with >=1.25x throughput"},
        "technical_passed": all(base_gates[name] for name in technical),
        "scientific_base_passed": promote_s,
    }


__all__ = ["select_common_policy", "summarize_phase3a2"]
