"""Aggregate the parameter-matched ASM-CM versus ASM-VR-S capability suite."""
from __future__ import annotations
from dataclasses import asdict
from statistics import mean, pstdev
import math
from .purpose_variants import PURPOSE_VARIANTS


def _language(runs):
    output = {"runs": [asdict(run) for run in runs]}
    for field in ("validation_ce", "test_ce", "tokens_per_second", "peak_memory_mb", "parameter_count", "streaming_error"):
        values = [float(getattr(run, field)) for run in runs]; output[f"{field}_mean"] = mean(values); output[f"{field}_std"] = pstdev(values)
    return output


def _mqar(rows):
    lengths = sorted({item["length"] for row in rows for item in row["mqar"]})
    by_length = {}
    for length in lengths:
        values = [next(item for item in row["mqar"] if item["length"] == length) for row in rows]
        successful = [item for item in values if item.get("status") != "failed"]
        finite_ce = [item["ce"] for item in successful if item.get("ce") is not None and math.isfinite(item["ce"])]
        by_length[str(length)] = {"runs": values, "accuracy_mean": mean(item["accuracy"] for item in successful) if successful else None, "ce_mean": mean(finite_ce) if finite_ce else None, "ce_finite_runs": len(finite_ce), "successful": len(successful)}
    return {"runs": rows, "by_length": by_length, "language_test_ce_after_mean": mean(row["language_test_ce_after"] for row in rows), "elapsed_sec_mean": mean(row["elapsed_sec"] for row in rows)}


def _stream(rows):
    lengths = sorted({item["length"] for row in rows for item in row["streaming"]})
    by_length = {}
    for length in lengths:
        values = [item for row in rows for item in row["streaming"] if item["length"] == length]
        successful = [item for item in values if item.get("status") != "failed"]
        by_length[str(length)] = {"runs": values, "tokens_per_second_mean": mean(item["tokens_per_second"] for item in successful) if successful else None, "retained_state_bytes_mean": mean(item["retained_state_bytes"] for item in successful) if successful else None, "cuda_peak_mb_mean": mean(item["cuda_peak_mb"] for item in successful) if successful else None, "latency_ms_p95_mean": mean(item["latency_ms_p95"] for item in successful) if successful else None, "seeds": [item["seed"] for item in values], "successful": len(successful)}
    return {"runs": rows, "by_length": by_length}


def summarize_purpose_suite(language_results, mqar_results, streaming_results, parameters):
    grouped_language = {name: sorted((run for run in language_results if run.variant == name), key=lambda run: run.seed) for name in PURPOSE_VARIANTS}
    language = {name: _language(runs) for name, runs in grouped_language.items()}
    mqar = {name: _mqar(sorted((row for row in mqar_results if row["variant"] == name), key=lambda row: row["seed"])) for name in PURPOSE_VARIANTS}
    streaming = {name: _stream(sorted((row for row in streaming_results if row["variant"] == name), key=lambda row: row["seed"])) for name in PURPOSE_VARIANTS}
    full = language["asm_vr_s_full"]; fixed = language["asm_vr_s_fixed_32"]; cm = language["asm_cm"]
    total_delta = max(abs(parameters[name]["total"] / parameters["asm_cm"]["total"] - 1) for name in PURPOSE_VARIANTS)
    mqar_32k = {name: mqar[name]["by_length"].get("32768", {}).get("accuracy_mean") for name in PURPOSE_VARIANTS}
    def bounded(name):
        values = list(streaming[name]["by_length"].values())
        if any(value["successful"] != len(value["runs"]) for value in values): return False
        sizes = [value["retained_state_bytes_mean"] for value in values]; return max(sizes) / min(sizes) <= 1.01
    def all_mqar(name, length, threshold):
        return all(item.get("accuracy", -1) >= threshold for item in mqar[name]["by_length"][str(length)]["runs"])
    def stream_32k_complete(name):
        value = streaming[name]["by_length"].get("32768"); return value is not None and value["successful"] == len(value["runs"])
    gates = {
        "complete_language": all(len(runs) == 3 for runs in grouped_language.values()),
        "parameter_match_total": total_delta <= .001,
        "finite_language": all(run.finite for run in language_results),
        "streaming_parity": max(run.streaming_error for run in language_results) <= 1e-4,
        "cm_bounded_state": bounded("asm_cm"),
        "vr_s_full_bounded_state": bounded("asm_vr_s_full"),
        "vr_s_fixed32_bounded_state": bounded("asm_vr_s_fixed_32"),
        "fixed32_language_noninferior": fixed["test_ce_mean"] - full["test_ce_mean"] <= .03,
        "fixed32_physical_gain": fixed["tokens_per_second_mean"] >= full["tokens_per_second_mean"] * 1.05 and fixed["peak_memory_mb_mean"] <= full["peak_memory_mb_mean"] * .95,
        "cm_mqar_short_80pct_all_seeds": all_mqar("asm_cm", 40, .80),
        "cm_mqar_32k_80pct_mean": mqar_32k["asm_cm"] is not None and mqar_32k["asm_cm"] >= .80,
        "cm_mqar_32k_80pct_all_seeds": all_mqar("asm_cm", 32768, .80),
        "vr_s_full_mqar_short_80pct_all_seeds": all_mqar("asm_vr_s_full", 40, .80),
        "cm_streaming_32k_complete": stream_32k_complete("asm_cm"),
        "vr_s_full_streaming_32k_complete": stream_32k_complete("asm_vr_s_full"),
        "vr_s_fixed32_streaming_32k_complete": stream_32k_complete("asm_vr_s_fixed_32"),
    }
    language_winner = min(language, key=lambda name: language[name]["test_ce_mean"])
    mqar_winner = max(PURPOSE_VARIANTS, key=lambda name: mqar_32k[name] if mqar_32k[name] is not None else -1)
    return {"experiment": "PMCS-64", "name": "ASM-CM vs ASM-VR-S Parameter-Matched Capability Suite", "parameters": parameters, "language": language, "mqar": mqar, "streaming": streaming, "gates": gates, "language_winner": language_winner, "mqar_32k_winner": mqar_winner, "technical_passed": all(gates[name] for name in ("complete_language", "parameter_match_total", "finite_language")), "purpose_conclusion": {"language": language_winner, "durable_recall_32k": mqar_winner, "cm_long_recall_successful_seeds": sum(item.get("accuracy", -1) >= .80 for item in mqar["asm_cm"]["by_length"]["32768"]["runs"]), "cm_streaming_status": "failed before 32K" if not gates["cm_streaming_32k_complete"] else "completed", "variable_rank_practicality": "fixed rank is logical capacity control; no physical gain was established"}}


__all__ = ["summarize_purpose_suite"]
