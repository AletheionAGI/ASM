"""Aggregate multiseed long-context ASM-CM-VR gates."""

from statistics import mean, pstdev
import math
from .cmvr_long_curriculum import ARMS, TEST_LENGTHS, arm_passed


def _classify_finite(row):
    classified = dict(row)
    tests = classified.get("test", ())
    streams = classified.get("streaming", ())
    classified["finite"] = (
        math.isfinite(classified.get("streaming_error", float("nan")))
        and all(item.get("ce_finite", False) for item in tests)
        and all(item.get("status") != "failed" for item in streams)
    )
    return classified


def summarize(results, requested_seeds):
    grouped = {
        arm: sorted(
            (_classify_finite(row) for row in results if row["arm"] == arm),
            key=lambda row: row["seed"],
        )
        for arm in ARMS
    }
    arms = {}
    for arm, rows in grouped.items():
        lengths = {}
        for length in TEST_LENGTHS:
            values = [
                next(item for item in row["test"] if item["length"] == length)
                for row in rows
            ]
            ok = [
                item
                for item in values
                if item.get("status") != "failed" and item.get("ce_finite", False)
            ]
            lengths[str(length)] = {
                "runs": values,
                "accuracy_mean": mean(item["accuracy"] for item in values)
                if values
                else None,
                "accuracy_std": pstdev(item["accuracy"] for item in values)
                if len(values) > 1
                else 0.0,
                "accuracy_mean_successful": mean(item["accuracy"] for item in ok)
                if ok
                else None,
                "mean_rank_mean": mean(
                    item["mean_rank"]
                    for item in values
                    if item.get("mean_rank") is not None
                )
                if any(item.get("mean_rank") is not None for item in values)
                else None,
                "successful": len(ok),
                "failed": len(values) - len(ok),
            }
        streams = {}
        for length in TEST_LENGTHS[1:]:
            values = [
                item
                for row in rows
                for item in row["streaming"]
                if item["length"] == length
            ]
            ok = [item for item in values if item.get("status") != "failed"]
            streams[str(length)] = {
                "runs": values,
                "tokens_per_second_mean": mean(item["tokens_per_second"] for item in ok)
                if ok
                else None,
                "retained_state_bytes_mean": mean(
                    item["retained_state_bytes"] for item in ok
                )
                if ok
                else None,
                "successful": len(ok),
            }
        arms[arm] = {
            "runs": rows,
            "by_length": lengths,
            "streaming": streams,
            "passed_seeds": sum(arm_passed(row) for row in rows),
            "streaming_error_mean": mean(row["streaming_error"] for row in rows)
            if rows
            else None,
        }
    fixed = grouped["cm_vr_fixed32"]
    full = grouped["cm_vr_full64"]
    adaptive = grouped["cm_vr_adaptive32"]
    fixed_seed17 = next((row for row in fixed if row["seed"] == 17), None)
    gates = {
        "seed17_fixed_long_gate": fixed_seed17 is not None and arm_passed(fixed_seed17),
        "all_requested_seeds_executed": all(
            len(rows) == len(requested_seeds) for rows in grouped.values()
        ),
        "fixed_long_gate_all_seeds": len(fixed) == len(requested_seeds)
        and all(arm_passed(row) for row in fixed),
        "full_long_gate_all_seeds": len(full) == len(requested_seeds)
        and all(arm_passed(row) for row in full),
        "fixed_memory_causal_all_seeds": len(fixed) == len(requested_seeds)
        and all(
            row["no_read"]["accuracy"] < 0.20 and row["no_write"]["accuracy"] < 0.20
            for row in fixed
        ),
        "fixed_full_parameter_match_trainable": bool(fixed and full)
        and fixed[0]["parameters_trainable"] == full[0]["parameters_trainable"],
        "all_arms_parameter_match_total": all(
            row["parameters_total"] == results[0]["parameters_total"] for row in results
        ),
        "adaptive_executed_all_seeds": len(adaptive) == len(requested_seeds),
        "adaptive_controller_received_gradients": bool(adaptive)
        and all(row["controller_gradient_hits"] > 0 for row in adaptive),
        "adaptive_rank_varied": bool(adaptive)
        and all(
            any(item.get("rank_std", 0) > 0 for item in row["history"])
            for row in adaptive
        ),
    }
    return {
        "experiment": "ASM-CM-VR fixed-32 long multiseed",
        "requested_seeds": list(requested_seeds),
        "arms": arms,
        "gates": gates,
        "passed": gates["fixed_long_gate_all_seeds"],
        "conclusion": "fixed-32 passed all long gates"
        if gates["fixed_long_gate_all_seeds"]
        else "fixed-32 not promoted by long gates",
    }


__all__ = ["summarize"]
