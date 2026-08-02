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
    parser = argparse.ArgumentParser(description="Consolidate frozen post-FP32 ASM-C2-FW-LM measurements.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    args = parser.parse_args()
    previous = load(args.confirmation)
    rows = []
    for seed in (1, 2, 3):
        root = args.root / f"seed_{seed}"
        language = load(root / "language.json")
        streaming = load(root / "streaming.json")
        parity = load(root / "bf16_parity.json")
        by_length = {int(row["sequence_length"]): row for row in streaming["streaming"]}
        rows.append({
            "seed": seed,
            "ce": language["test_ce"],
            "ppl": language["test_ppl"],
            "tokens": language["test_tokens"],
            "manifest_sha256": language["manifest_sha256"],
            "throughput_4096": by_length[4096]["segment_tokens_per_sec"],
            "throughput_32768": by_length[32768]["segment_tokens_per_sec"],
            "vram_peak_4096_mib": by_length[4096]["cuda_peak_mb"],
            "vram_peak_32768_mib": by_length[32768]["cuda_peak_mb"],
            "cache_4096_bytes": by_length[4096]["cache_tensor_bytes"],
            "cache_32768_bytes": by_length[32768]["cache_tensor_bytes"],
            "bf16_mean_abs_error": parity["mean_abs_error"],
            "bf16_argmax_mismatch_rate": parity["argmax_mismatch_rate"],
        })
    old_by_seed = {int(row["seed"]): row for row in previous["seeds"]}
    criteria = {
        "three_frozen_seeds": len(rows) == 3,
        "same_frozen_manifest": len({row["manifest_sha256"] for row in rows}) == 1,
        "full_validation_each_seed": all(row["tokens"] >= 4_800_000 for row in rows),
        "language_regression_vs_pre_fix_at_most_0_01": all(row["ce"] - old_by_seed[row["seed"]]["candidate_ce"] <= 0.01 for row in rows),
        "cache_bounded_4k_to_32k": all(row["cache_32768_bytes"] <= 1.1 * row["cache_4096_bytes"] for row in rows),
        "vram_growth_at_most_10pct": all(row["vram_peak_32768_mib"] <= 1.1 * row["vram_peak_4096_mib"] for row in rows),
        "throughput_retention_at_least_80pct": all(row["throughput_32768"] >= 0.8 * row["throughput_4096"] for row in rows),
        "bf16_mean_abs_error_at_most_0_02": all(row["bf16_mean_abs_error"] <= 0.02 for row in rows),
        "bf16_argmax_mismatch_at_most_1pct": all(row["bf16_argmax_mismatch_rate"] <= 0.01 for row in rows),
    }
    payload = {
        "public_name": "ASM-CM",
        "technical_variant": "ASM-C2-FW-LM",
        "public_expansion": "Aletheion Compact Memory Model",
        "promote": previous.get("promote", False) and all(criteria.values()),
        "criteria": criteria,
        "aggregate": {
            "ce_mean": mean([row["ce"] for row in rows]),
            "ce_std": statistics.pstdev([row["ce"] for row in rows]),
            "throughput_4096_mean": mean([row["throughput_4096"] for row in rows]),
            "throughput_32768_mean": mean([row["throughput_32768"] for row in rows]),
            "vram_peak_32768_mib_mean": mean([row["vram_peak_32768_mib"] for row in rows]),
            "cache_32768_bytes_mean": mean([row["cache_32768_bytes"] for row in rows]),
        },
        "seeds": rows,
        "provenance": {"training_repeated": False, "prior_confirmation": str(args.confirmation)},
    }
    (args.root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ASM-CM post-FP32 frozen revalidation", "",
        f"Final promotion gate: **{'PASS' if payload['promote'] else 'FAIL'}**", "",
        "ASM-CM is the public name of the technical variant ASM-C2-FW-LM. No checkpoint was retrained.", "",
        "| Seed | CE | PPL | tok/s 4K | tok/s 32K | VRAM 32K (MiB) | Cache 32K (bytes) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *(f"| {r['seed']} | {r['ce']:.6f} | {r['ppl']:.4f} | {r['throughput_4096']:.2f} | {r['throughput_32768']:.2f} | {r['vram_peak_32768_mib']:.2f} | {r['cache_32768_bytes']} |" for r in rows),
        "", "## Gates", "",
        *(f"- {key}: **{'PASS' if value else 'FAIL'}**" for key, value in criteria.items()), "",
        "The Transformer remains the stronger language-modeling control by CE. Promotion identifies durable, bounded-state associative memory without unacceptable regression relative to ASM-R; it does not claim Transformer superiority.",
    ]
    (args.root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"promote": payload["promote"], "aggregate": payload["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
