"""Aggregate the ASM-VR-RS full-rank comparison."""
from dataclasses import asdict
from statistics import mean, pstdev


def _aggregate(runs):
    output = {"runs": [asdict(run) for run in runs]}
    for field in ("validation_ce", "test_ce", "tokens_per_second", "peak_memory_mb", "parameter_count", "streaming_error"):
        values = [float(getattr(run, field)) for run in runs]; output[f"{field}_mean"] = mean(values); output[f"{field}_std"] = pstdev(values)
    return output


def summarize_phase3a3(results, phase3a2):
    rs = _aggregate(sorted(results, key=lambda run: run.seed)); variants = {
        "vr_r_full": phase3a2["variants"]["vr_r_full"],
        "vr_s_full": phase3a2["variants"]["vr_s_full"],
        "vr_rs_full": rs,
    }
    baselines = {name: {run["seed"]: run for run in value["runs"]} for name, value in variants.items() if name != "vr_rs_full"}
    paired = {name: [{"seed": run.seed, "rs_minus_base_test_ce": run.test_ce - baselines[name][run.seed]["test_ce"]} for run in results] for name in baselines}
    quality_winner = min(variants, key=lambda name: variants[name]["test_ce_mean"])
    gates = {"complete": len(results) == 3, "finite": all(run.finite for run in results), "streaming": max(run.streaming_error for run in results) <= 1e-4, "rs_beats_r": mean(item["rs_minus_base_test_ce"] for item in paired["vr_r_full"]) <= -.01, "rs_beats_s": mean(item["rs_minus_base_test_ce"] for item in paired["vr_s_full"]) <= -.01}
    return {"experiment": "3A.3", "name": "ASM-VR-RS — Variable-Rank Relational Selective State Emitter", "variants": variants, "paired": paired, "quality_winner": quality_winner, "gates": gates, "technical_passed": all(gates[name] for name in ("complete", "finite", "streaming"))}


__all__ = ["summarize_phase3a3"]
