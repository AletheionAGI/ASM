from __future__ import annotations

import argparse
import json
from pathlib import Path

from drm_language_emitter.utils import save_json


def build_decision(
    payload: dict,
    candidate_name: str,
    baseline_name: str,
    required_ce_improvement: float,
    required_seeds: int,
    max_candidate_std: float,
) -> dict:
    aggregate = {row["variant"].upper(): row for row in payload["aggregate"]}
    candidate_name = candidate_name.upper()
    baseline_name = baseline_name.upper()
    candidate = aggregate[candidate_name]
    baseline = aggregate[baseline_name]
    improvement = float(baseline["validation_ce_mean"]) - float(
        candidate["validation_ce_mean"]
    )
    candidate_runs = {
        int(row["seed"]): float(row["validation_ce"])
        for row in payload["runs"]
        if row["variant"].upper() == candidate_name
    }
    baseline_runs = {
        int(row["seed"]): float(row["validation_ce"])
        for row in payload["runs"]
        if row["variant"].upper() == baseline_name
    }
    paired_seeds = sorted(candidate_runs.keys() & baseline_runs.keys())
    paired_improvements = {
        str(seed): baseline_runs[seed] - candidate_runs[seed] for seed in paired_seeds
    }
    paired_wins = sum(delta > 0.0 for delta in paired_improvements.values())
    required_paired_wins = required_seeds // 2 + 1
    reasons = []
    if len(paired_seeds) < required_seeds:
        reasons.append(
            f"only {len(paired_seeds)} paired seeds; {required_seeds} required"
        )
    if improvement < required_ce_improvement:
        reasons.append(
            f"mean CE improvement {improvement:.6f} is below "
            f"{required_ce_improvement:.6f}"
        )
    if paired_wins < required_paired_wins:
        reasons.append(
            f"candidate wins {paired_wins}/{len(paired_seeds)} paired seeds; "
            f"{required_paired_wins} required"
        )
    if float(candidate["validation_ce_std"]) > max_candidate_std:
        reasons.append(
            f"candidate CE std {candidate['validation_ce_std']:.6f} exceeds "
            f"{max_candidate_std:.6f}"
        )
    promote = not reasons
    return {
        "promote_to_tokens": 30_000_000 if promote else None,
        "promote": promote,
        "candidate": candidate,
        "baseline": baseline,
        "mean_ce_improvement": improvement,
        "paired_improvements": paired_improvements,
        "paired_wins": paired_wins,
        "required_paired_wins": required_paired_wins,
        "required_ce_improvement": required_ce_improvement,
        "required_seeds": required_seeds,
        "max_candidate_std": max_candidate_std,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate DRM-fix promotion from 5M to 30M tokens.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--required-ce-improvement",
        type=float,
        default=0.005,
        help="Minimum reduction in mean validation CE required for promotion.",
    )
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--max-candidate-std", type=float, default=0.03)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    candidate_name = args.candidate.upper()
    baseline_name = args.baseline.upper()
    decision = build_decision(
        payload,
        candidate_name,
        baseline_name,
        args.required_ce_improvement,
        args.required_seeds,
        args.max_candidate_std,
    )
    output = args.output or args.summary.with_name("promotion_decision.json")
    save_json(output, decision)
    print(json.dumps(decision, indent=2))
    if decision["promote"]:
        print(
            "\nPromotion approved. Suggested command:\n"
            f"./scripts/run_drm_fix_ablation.sh --variants {candidate_name} "
            "--target-tokens 30000000 --output-root runs/drm_fix_ablation_30m"
        )
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
