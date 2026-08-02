from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize durable ASM-C2-FW gates.")
    parser.add_argument("--seed-results", type=Path, nargs="+", required=True)
    parser.add_argument("--long-results", type=Path)
    parser.add_argument("--language-baseline", type=Path)
    parser.add_argument("--language-candidate", type=Path)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variant", default="ASM-C2-FW durable")
    args = parser.parse_args()
    seeds = [json.loads(path.read_text()) for path in args.seed_results]
    seed_passes = [bool(result["passed"]) for result in seeds]
    criteria = {
        "durable_curriculum_2_of_3_seeds": len(seeds) >= 3 and sum(seed_passes) >= 2,
    }
    details = {
        "seed_results": [
            {"path": str(path), "passed": result["passed"], "final": result["final"]}
            for path, result in zip(args.seed_results, seeds)
        ]
    }
    if args.long_results and args.long_results.exists():
        long = json.loads(args.long_results.read_text())
        retention = {row["sequence_length"]: row for row in long["mqar"] if row.get("status") != "failed"}
        criteria["short_control_authorizes_long_interpretation"] = bool(long.get("mqar_control_passed"))
        for length in (512, 4096, 32768):
            criteria[f"mqar_{length}_at_least_80pct"] = bool(
                retention.get(length) and retention[length]["validation_accuracy"] >= 0.8
            )
        streaming = {row["sequence_length"]: row for row in long["streaming"] if row.get("status") != "failed"}
        four, thirty_two = streaming.get(4096), streaming.get(32768)
        criteria["cache_bounded_4k_to_32k"] = bool(four and thirty_two and thirty_two["cache_tensor_bytes"] <= 1.1 * four["cache_tensor_bytes"])
        criteria["vram_growth_at_most_10pct"] = bool(four and thirty_two and thirty_two["cuda_peak_mb"] <= 1.1 * four["cuda_peak_mb"])
        criteria["throughput_retention_at_least_80pct"] = bool(four and thirty_two and thirty_two["segment_tokens_per_sec"] >= 0.8 * four["segment_tokens_per_sec"])
        details["long_mqar"] = retention
        details["streaming"] = streaming
    if args.language_baseline and args.language_candidate:
        baseline = json.loads(args.language_baseline.read_text())["test_ce"]
        candidate = json.loads(args.language_candidate.read_text())["test_ce"]
        criteria["language_ce_regression_at_most_0_05"] = candidate - baseline <= 0.05
        details["language_ce"] = {"baseline": baseline, "candidate": candidate, "regression": candidate - baseline}
    if args.parity and args.parity.exists():
        parity = json.loads(args.parity.read_text())
        criteria["bf16_argmax_mismatch_at_most_1pct"] = parity["argmax_mismatch_rate"] <= 0.01
        criteria["bf16_mean_abs_error_at_most_0_02"] = parity["mean_abs_error"] <= 0.02
        details["bf16_parity"] = parity
    payload = {"promote": all(criteria.values()), "criteria": criteria, **details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload["variant"] = args.variant
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [f"# {args.variant} decision", "", f"Promote: **{payload['promote']}**", "", "## Criteria", ""]
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in criteria.items())
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"promote": payload["promote"], "criteria": criteria}, indent=2))


if __name__ == "__main__":
    main()
