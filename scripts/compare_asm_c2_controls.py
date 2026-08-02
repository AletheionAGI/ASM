from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def final_rows(payload: dict) -> dict[str, dict]:
    return {result["variant"]: result["rows"][-1] for result in payload["results"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ASM-C2 short and long promotion gates.")
    parser.add_argument("--short-results", type=Path, required=True)
    parser.add_argument("--ablations", type=Path)
    parser.add_argument("--long-results", type=Path)
    parser.add_argument("--confirmations", type=Path, nargs="*")
    parser.add_argument("--language-baseline", type=Path)
    parser.add_argument("--language-candidate", type=Path)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--short-threshold", type=float, default=0.8)
    parser.add_argument("--ablation-margin", type=float, default=0.05)
    parser.add_argument("--long-threshold", type=float, default=0.8)
    args = parser.parse_args()
    short = json.loads(args.short_results.read_text())
    rows = final_rows(short)
    candidates = [
        (name, row)
        for name, row in rows.items()
        if name in {
            "ASM_C2_16", "ASM_C2_32", "ASM_C2_64", "ASM_C2_FW",
            "ASM_C2_FW_DURABLE",
        }
        and row["validation_accuracy"] >= args.short_threshold
    ]
    candidates.sort(key=lambda item: (-item[1]["validation_accuracy"], item[1]["validation_ce"]))
    winner = candidates[0][0] if candidates else None
    criteria = {"short_control_at_least_80pct": winner is not None}
    details: dict = {"short_final_rows": rows}

    if args.ablations and args.ablations.exists() and winner:
        ablation_rows = final_rows(json.loads(args.ablations.read_text()))
        winner_accuracy = ablation_rows[winner]["validation_accuracy"]
        if winner == "ASM_C2_FW_DURABLE":
            prefix = "ASM_C2_FW_DURABLE_"
        elif winner == "ASM_C2_FW":
            prefix = "ASM_C2_FW_"
        else:
            prefix = "ASM_C2_"
        no_read = ablation_rows[f"{prefix}NOREAD"]["validation_accuracy"]
        no_write = ablation_rows[f"{prefix}NOWRITE"]["validation_accuracy"]
        shuffled = ablation_rows[f"{prefix}SHUFFLED"]["validation_accuracy"]
        criteria["read_ablation_margin"] = winner_accuracy - no_read >= args.ablation_margin
        criteria["write_ablation_margin"] = winner_accuracy - no_write >= args.ablation_margin
        criteria["shuffled_memory_margin"] = winner_accuracy - shuffled >= args.ablation_margin
        details["ablation_final_rows"] = ablation_rows

    if args.long_results and args.long_results.exists() and winner:
        long = json.loads(args.long_results.read_text())
        streaming = {row["sequence_length"]: row for row in long["streaming"] if row.get("status") != "failed"}
        four = streaming.get(4096)
        thirty_two = streaming.get(32768)
        criteria["short_control_authorizes_long_interpretation"] = bool(
            long.get("mqar_control_passed")
        )
        retention = {
            row["sequence_length"]: row
            for row in long.get("mqar", [])
            if row.get("status") != "failed"
        }
        for length in (512, 4096, 32768):
            row = retention.get(length)
            criteria[f"mqar_{length}_at_least_80pct"] = bool(
                row and row["validation_accuracy"] >= args.long_threshold
            )
        details["long_mqar"] = retention
        criteria["cache_bounded_4k_to_32k"] = bool(four and thirty_two and thirty_two["cache_tensor_bytes"] <= four["cache_tensor_bytes"] * 1.1)
        criteria["vram_growth_at_most_10pct"] = bool(four and thirty_two and thirty_two["cuda_peak_mb"] <= four["cuda_peak_mb"] * 1.1)
        criteria["throughput_retention_at_least_80pct"] = bool(four and thirty_two and thirty_two["segment_tokens_per_sec"] >= four["segment_tokens_per_sec"] * 0.8)
        details["long_protocol"] = long.get("protocol")

    if args.confirmations and winner:
        confirmation_accuracies = []
        for path in args.confirmations:
            confirmation_accuracies.append(final_rows(json.loads(path.read_text()))[winner]["validation_accuracy"])
        wins = sum(value >= args.short_threshold for value in confirmation_accuracies)
        criteria["multiseed_short_control_2_of_3"] = len(confirmation_accuracies) >= 3 and wins >= 2
        details["confirmation_accuracies"] = confirmation_accuracies
        details["confirmation_mean"] = statistics.fmean(confirmation_accuracies)
        details["confirmation_population_std"] = statistics.pstdev(confirmation_accuracies)

    if args.language_baseline and args.language_candidate and winner:
        baseline_ce = json.loads(args.language_baseline.read_text())["test_ce"]
        candidate_ce = json.loads(args.language_candidate.read_text())["test_ce"]
        regression = candidate_ce - baseline_ce
        criteria["language_ce_regression_at_most_0_05"] = regression <= 0.05
        details["language_ce"] = {"baseline": baseline_ce, "candidate": candidate_ce, "regression": regression}

    if args.parity and args.parity.exists() and winner:
        parity = json.loads(args.parity.read_text())
        criteria["bf16_argmax_mismatch_at_most_1pct"] = parity["argmax_mismatch_rate"] <= 0.01
        criteria["bf16_mean_abs_error_at_most_0_02"] = parity["mean_abs_error"] <= 0.02
        details["bf16_parity"] = parity

    promote = bool(winner) and all(criteria.values())
    payload = {"winner": winner, "promote": promote, "criteria": criteria, **details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# ASM-C2 promotion decision", "", f"Winner: **{winner or 'none'}**", f"", f"Promote: **{promote}**", "", "## Criteria", ""]
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in criteria.items())
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner, "promote": promote, "criteria": criteria}, indent=2))


if __name__ == "__main__":
    main()
