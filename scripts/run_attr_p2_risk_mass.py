#!/usr/bin/env python3
"""Run the post-hoc ATTR P2 ASM-X native-risk-mass diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from aletheion_state_models.benchmarks.transition_risk.p2_risk_mass_plots import (
    render_risk_mass_extension,
)
from aletheion_state_models.benchmarks.transition_risk.p2_risk_mass_runner import (
    evaluate_risk_mass_extension,
    train_risk_mass_extension,
)
from aletheion_state_models.benchmarks.transition_risk.p2_risk_mass_summary import (
    build_risk_mass_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("train", "evaluate", "summarize", "all"), default="all"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/asm_transformer_transition_risk/p2"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs/attr_p2"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output, run_root = args.root / args.output, args.root / args.run_root
    if args.phase in {"train", "all"}:
        results = train_risk_mass_extension(
            args.root, output, run_root, device=args.device
        )
        print({"risk_mass_training_complete": len(results)})
    if args.phase in {"evaluate", "all"}:
        evaluate_risk_mass_extension(args.root, output, run_root, device=args.device)
        print({"risk_mass_evaluation_complete": True})
    if args.phase in {"summarize", "all"}:
        summary = build_risk_mass_summary(args.root, output, run_root)
        render_risk_mass_extension(summary, output)
        print({"risk_mass_summary_complete": True})


if __name__ == "__main__":
    main()
