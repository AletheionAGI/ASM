"""Aggregation and scientific gates for ASM-VR Phase 3A.1."""
from __future__ import annotations
from dataclasses import asdict
import math
from statistics import mean, pstdev
from .phase3a1_variants import STAGE_A_COMPONENTS, STAGE_A_VARIANTS, STAGE_B_VARIANTS
from .phase3a_training import Phase3ARunResult


def _group(results: list[Phase3ARunResult], variants: tuple[str, ...]):
    grouped = {}
    for variant in variants:
        runs = sorted((run for run in results if run.variant == variant), key=lambda run: run.seed)
        if runs:
            grouped[variant] = runs
    return grouped


def _aggregate(runs: list[Phase3ARunResult]) -> dict:
    fields = (
        "validation_ce", "test_ce", "mean_rank", "rank_std",
        "rank_ce_correlation", "controller_gradient_fraction",
        "tokens_per_second", "peak_memory_mb", "parameter_count", "streaming_error",
    )
    output = {"seeds": [run.seed for run in runs], "runs": [asdict(run) for run in runs]}
    for field in fields:
        values = [float(getattr(run, field)) for run in runs]
        output[f"{field}_mean"] = mean(values)
        output[f"{field}_std"] = pstdev(values)
    return output


def _factorial_effects(grouped: dict[str, list[Phase3ARunResult]]) -> dict[str, dict]:
    names = ("mixer", "residual", "selective", "mixer:residual", "mixer:selective", "residual:selective", "mixer:residual:selective")
    terms = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
    effects = {}
    seeds = sorted({run.seed for runs in grouped.values() for run in runs})
    for name, term in zip(names, terms, strict=True):
        paired = []
        for seed in seeds:
            contrast = 0.0
            for variant, flags in STAGE_A_COMPONENTS.items():
                run = next(item for item in grouped[variant] if item.seed == seed)
                sign = math.prod(1 if flags[index] else -1 for index in term)
                contrast += sign * run.validation_ce
            paired.append(contrast / 4.0)
        effects[name] = {"validation_ce_effect_mean": mean(paired), "by_seed": paired}
    return effects


def summarize_stage_a(results: list[Phase3ARunResult]) -> dict:
    grouped = _group(results, STAGE_A_VARIANTS)
    variants = {name: _aggregate(runs) for name, runs in grouped.items()}
    complete = set(grouped) == set(STAGE_A_VARIANTS) and all(len(runs) == 3 for runs in grouped.values())
    minimum = min(value["validation_ce_mean"] for value in variants.values())
    candidates = [name for name, value in variants.items() if value["validation_ce_mean"] <= minimum + 0.02]
    selected = min(candidates, key=lambda name: (sum(STAGE_A_COMPONENTS[name]), variants[name]["validation_ce_mean"]))
    strict_runs = {run.seed: run for run in grouped["strict"]}
    selected_runs = {run.seed: run for run in grouped[selected]}
    deltas = {str(seed): selected_runs[seed].validation_ce - strict_runs[seed].validation_ce for seed in strict_runs}
    gain = -mean(deltas.values())
    gates = {
        "complete_matrix": complete,
        "finite_runs": all(run.finite for run in results),
        "streaming_parity": max(run.streaming_error for run in results) <= 1e-4,
        "quality_recovery": gain >= 0.05 and sum(delta <= -0.05 for delta in deltas.values()) >= 2,
    }
    return {
        "stage": "3A.1-A", "variants": variants, "factorial_effects": _factorial_effects(grouped),
        "selection": {"variant": selected, "components": STAGE_A_COMPONENTS[selected], "validation_gain_vs_strict": gain, "paired_deltas": deltas, "rule": "within 0.02 nat of best, prefer fewer components"},
        "gates": gates, "operational_passed": all(gates[name] for name in ("complete_matrix", "finite_runs", "streaming_parity")),
    }


def _interpolated_fixed_ce(variants: dict, rank: float) -> float:
    points = sorted((variants[name]["mean_rank_mean"], variants[name]["test_ce_mean"]) for name in STAGE_B_VARIANTS if "adaptive" not in name)
    for (left_rank, left_ce), (right_rank, right_ce) in zip(points, points[1:]):
        if left_rank <= rank <= right_rank:
            weight = (rank - left_rank) / (right_rank - left_rank)
            return left_ce + weight * (right_ce - left_ce)
    return min(points, key=lambda point: abs(point[0] - rank))[1]


def summarize_stage_b(results: list[Phase3ARunResult], selection: dict, threshold: float) -> dict:
    grouped = _group(results, STAGE_B_VARIANTS)
    variants = {name: _aggregate(runs) for name, runs in grouped.items()}
    adaptive = variants["selected_adaptive_32"]
    fixed32 = variants["selected_fixed_32"]
    frontier_ce = _interpolated_fixed_ce(variants, adaptive["mean_rank_mean"])
    fixed_with_less_rank = [value for name, value in variants.items() if "adaptive" not in name and value["mean_rank_mean"] <= adaptive["mean_rank_mean"]]
    pareto = not any(value["test_ce_mean"] <= adaptive["test_ce_mean"] - 0.01 for value in fixed_with_less_rank)
    advantage = adaptive["test_ce_mean"] - frontier_ce
    paired = []
    fixed_by_seed = {run.seed: run for run in grouped["selected_fixed_32"]}
    for run in grouped["selected_adaptive_32"]:
        paired.append({"seed": run.seed, "adaptive_minus_fixed32_test_ce": run.test_ce - fixed_by_seed[run.seed].test_ce})
    gates = {
        "complete_matrix": set(grouped) == set(STAGE_B_VARIANTS) and all(len(runs) == 3 for runs in grouped.values()),
        "finite_runs": all(run.finite for run in results),
        "streaming_parity": max(run.streaming_error for run in results) <= 1e-4,
        "adaptive_budget": 28.8 <= adaptive["mean_rank_mean"] <= 35.2,
        "adaptive_variation": min(run.rank_std for run in grouped["selected_adaptive_32"]) > 1.0,
        "quality_near_fixed32": mean(item["adaptive_minus_fixed32_test_ce"] for item in paired) <= 0.05 and max(item["adaptive_minus_fixed32_test_ce"] for item in paired) <= 0.10,
        "controller_gradient": min(run.controller_gradient_fraction for run in grouped["selected_adaptive_32"]) >= 0.9,
        "pareto": pareto,
        "adaptive_frontier_advantage": advantage <= -0.02,
    }
    integrity = ("complete_matrix", "finite_runs", "streaming_parity", "adaptive_budget", "adaptive_variation", "quality_near_fixed32", "controller_gradient")
    return {
        "stage": "3A.1-B", "selection": selection, "calibrated_threshold": threshold,
        "variants": variants, "paired_quality": paired,
        "fixed_frontier_ce_at_adaptive_rank": frontier_ce,
        "adaptive_minus_fixed_frontier_ce": advantage,
        "gates": gates, "operational_passed": all(gates[name] for name in integrity),
        "scientific_passed": gates["pareto"] and gates["adaptive_frontier_advantage"],
    }


__all__ = ["summarize_stage_a", "summarize_stage_b"]
