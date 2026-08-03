#!/usr/bin/env python3
"""Summarize the paired ASM-CM versus ASM-CM-E ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for seed in (1, 2, 3):
        root = args.root / f"seed_{seed}"
        training = load(root / "training" / "results.json")
        candidate_ce = load(root / "language.json")["test_ce"]
        baseline_ce = load(
            args.baseline_root / f"seed_{seed}" / "language" / "asm_c2_fw_lm.json"
        )["test_ce"]
        long = load(root / "long_32k.json")
        mqar = {row["sequence_length"]: row for row in long["mqar"]}
        streaming = long["streaming"]
        cache_values = {row["cache_tensor_bytes"] for row in streaming}
        rows.append({
            "seed": seed,
            "parameter_count": training["parameter_count"],
            "short_curriculum_passed": training["passed"],
            "candidate_ce": candidate_ce,
            "baseline_ce": baseline_ce,
            "ce_delta": candidate_ce - baseline_ce,
            "mqar_32k_accuracy": mqar[32768]["validation_accuracy"],
            "cache_constant": len(cache_values) == 1,
            "cache_bytes": max(cache_values),
        })
    gates = {
        "all_short_curricula_passed": all(row["short_curriculum_passed"] for row in rows),
        "all_32k_mqar_at_least_95pct": all(row["mqar_32k_accuracy"] >= 0.95 for row in rows),
        "all_cache_curves_constant": all(row["cache_constant"] for row in rows),
        "mean_ce_regression_at_most_0_02": mean(row["ce_delta"] for row in rows) <= 0.02,
    }
    payload = {
        "variant": "ASM-CM-E",
        "baseline": "ASM-CM",
        "rows": rows,
        "aggregate": {
            "candidate_ce_mean": mean(row["candidate_ce"] for row in rows),
            "baseline_ce_mean": mean(row["baseline_ce"] for row in rows),
            "ce_delta_mean": mean(row["ce_delta"] for row in rows),
            "mqar_32k_accuracy_mean": mean(row["mqar_32k_accuracy"] for row in rows),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    (args.root / "decision.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# ASM-CM-E paired ablation", "",
        "ASM-CM-E is experimental and does not replace promoted ASM-CM unless every frozen gate passes.", "",
        "| Seed | ASM-CM-E CE | ASM-CM CE | Delta | MQAR 32K | Cache |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['candidate_ce']:.6f} | {row['baseline_ce']:.6f} | "
            f"{row['ce_delta']:+.6f} | {row['mqar_32k_accuracy']:.2%} | "
            f"{row['cache_bytes']} B |"
        )
    lines.extend(["", f"**Decision:** {'PASS' if payload['passed'] else 'FAIL'}", ""])
    (args.root / "report.md").write_text("\n".join(lines))
    print(json.dumps({"passed": payload["passed"], "report": str(args.root / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
