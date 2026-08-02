from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the independent ASM-C2-FW-LM confirmation gates."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    seeds = (1, 2, 3)
    rows: list[dict] = []
    for seed in seeds:
        seed_root = args.root / f"seed_{seed}"
        training = load(seed_root / "candidate" / "results.json")
        candidate = load(seed_root / "language" / "asm_c2_fw_lm.json")
        baseline = load(seed_root / "language" / "asm_r.json")
        transformer = load(seed_root / "language" / "transformer.json")
        long_result = load(seed_root / "long_32k.json")
        parity = load(seed_root / "bf16_parity.json")
        mqar = {
            int(row["sequence_length"]): row
            for row in long_result["mqar"]
            if row.get("status") != "failed"
        }
        streaming = {
            int(row["sequence_length"]): row
            for row in long_result["streaming"]
            if row.get("status") != "failed"
        }
        rows.append({
            "seed": seed,
            "candidate_training_passed": bool(training["passed"]),
            "language_tokens": candidate["test_tokens"],
            "manifest_sha256": candidate["manifest_sha256"],
            "candidate_ce": candidate["test_ce"],
            "asm_r_ce": baseline["test_ce"],
            "transformer_ce": transformer["test_ce"],
            "candidate_minus_asm_r_ce": candidate["test_ce"] - baseline["test_ce"],
            "candidate_minus_transformer_ce": candidate["test_ce"] - transformer["test_ce"],
            "mqar_32768_accuracy": mqar[32768]["validation_accuracy"],
            "mqar_32768_targets": mqar[32768]["targets"],
            "cache_4096_bytes": streaming[4096]["cache_tensor_bytes"],
            "cache_32768_bytes": streaming[32768]["cache_tensor_bytes"],
            "vram_4096_mb": streaming[4096]["cuda_peak_mb"],
            "vram_32768_mb": streaming[32768]["cuda_peak_mb"],
            "throughput_4096": streaming[4096]["segment_tokens_per_sec"],
            "throughput_32768": streaming[32768]["segment_tokens_per_sec"],
            "bf16_argmax_mismatch_rate": parity["argmax_mismatch_rate"],
            "bf16_mean_abs_error": parity["mean_abs_error"],
        })

    candidate_ce = [row["candidate_ce"] for row in rows]
    asm_r_ce = [row["asm_r_ce"] for row in rows]
    transformer_ce = [row["transformer_ce"] for row in rows]
    paired = [row["candidate_minus_asm_r_ce"] for row in rows]
    manifest_hashes = {row["manifest_sha256"] for row in rows}
    language_tokens = {row["language_tokens"] for row in rows}
    criteria = {
        "three_independent_seed_lineages": len(rows) == 3,
        "same_frozen_language_corpus": len(manifest_hashes) == 1,
        "full_validation_corpus": min(language_tokens) >= 4_800_000,
        "candidate_curriculum_passes_3_of_3": all(
            row["candidate_training_passed"] for row in rows
        ),
        "candidate_language_regression_at_most_0_05_mean": mean(paired) <= 0.05,
        "candidate_language_regression_at_most_0_05_each_seed": all(
            value <= 0.05 for value in paired
        ),
        "mqar_32k_at_least_80pct_each_seed": all(
            row["mqar_32768_accuracy"] >= 0.8 for row in rows
        ),
        "mqar_32k_at_least_4096_targets_each_seed": all(
            row["mqar_32768_targets"] >= 4096 for row in rows
        ),
        "cache_bounded_4k_to_32k_each_seed": all(
            row["cache_32768_bytes"] <= 1.1 * row["cache_4096_bytes"] for row in rows
        ),
        "vram_growth_at_most_10pct_each_seed": all(
            row["vram_32768_mb"] <= 1.1 * row["vram_4096_mb"] for row in rows
        ),
        "throughput_retention_at_least_80pct_each_seed": all(
            row["throughput_32768"] >= 0.8 * row["throughput_4096"] for row in rows
        ),
        "bf16_argmax_mismatch_at_most_1pct_each_seed": all(
            row["bf16_argmax_mismatch_rate"] <= 0.01 for row in rows
        ),
        "bf16_mean_abs_error_at_most_0_02_each_seed": all(
            row["bf16_mean_abs_error"] <= 0.02 for row in rows
        ),
    }
    payload = {
        "variant": "ASM-C2-FW-LM",
        "promote": all(criteria.values()),
        "criteria": criteria,
        "aggregate": {
            "candidate_ce_mean": mean(candidate_ce),
            "candidate_ce_std": statistics.pstdev(candidate_ce),
            "asm_r_ce_mean": mean(asm_r_ce),
            "transformer_ce_mean": mean(transformer_ce),
            "candidate_minus_asm_r_ce_mean": mean(paired),
            "candidate_paired_wins_over_asm_r": sum(value < 0 for value in paired),
            "candidate_minus_transformer_ce_mean": mean(candidate_ce) - mean(transformer_ce),
        },
        "seeds": rows,
        "interpretation": {
            "transformer_ce_is_a_comparison_not_a_promotion_gate": True,
            "reason": (
                "Promotion tests durable associative memory without unacceptable loss of "
                "ASM-R language quality; it does not claim Transformer CE superiority."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Independent ASM-C2-FW-LM confirmation",
        "",
        f"Official promotion: **{payload['promote']}**",
        "",
        "## Gates",
        "",
        *(f"- {key}: **{'PASS' if value else 'FAIL'}**" for key, value in criteria.items()),
        "",
        "## Frozen language rescoring",
        "",
        "| Seed | ASM-C2-FW-LM CE | ASM-R CE | Transformer CE | Candidate - ASM-R |",
        "|---:|---:|---:|---:|---:|",
        *(
            f"| {row['seed']} | {row['candidate_ce']:.6f} | {row['asm_r_ce']:.6f} | "
            f"{row['transformer_ce']:.6f} | {row['candidate_minus_asm_r_ce']:+.6f} |"
            for row in rows
        ),
        "",
        "Transformer CE is reported as an external architectural control. Beating it is not "
        "a promotion gate and no superiority claim follows from the memory gates.",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"promote": payload["promote"], "aggregate": payload["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
