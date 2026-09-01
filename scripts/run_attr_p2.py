#!/usr/bin/env python3
"""Run sealed ATTR P2 training and test evaluation in ordered phases."""

from __future__ import annotations
import argparse
from pathlib import Path
from aletheion_state_models.benchmarks.transition_risk.p2_plots import render_p2
from aletheion_state_models.benchmarks.transition_risk.p2_runner import (
    evaluate_opened,
    seal_and_open,
    train_and_freeze,
)
from aletheion_state_models.benchmarks.transition_risk.p2_summary import (
    build_p2_summary,
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
    output = args.root / args.output
    run_root = args.root / args.run_root
    if args.phase in {"train", "all"}:
        results = train_and_freeze(args.root, output, run_root, device=args.device)
        print({"p2_training_complete": len(results)})
    if args.phase in {"evaluate", "all"}:
        opened = seal_and_open(output, run_root)
        evaluate_opened(args.root, run_root, opened, device=args.device)
        print({"p2_evaluation_complete": list(opened)})
    if args.phase in {"summarize", "all"}:
        summary = build_p2_summary(args.root, output, run_root)
        render_p2(summary, output)
        print({"p2_summary_complete": summary["gates"]})


if __name__ == "__main__":
    main()
