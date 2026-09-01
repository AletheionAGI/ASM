#!/usr/bin/env python3
"""Run an ATTR P0 smoke or train-only P1 pilot without unsealing test."""

from __future__ import annotations
import argparse
from pathlib import Path
from aletheion_state_models.benchmarks.transition_risk.runner import (
    run_attr_p0_smoke,
    run_attr_train_only,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("p0-smoke", "p1-pilot"), default="p0-smoke")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.phase == "p0-smoke":
        output = args.output_dir or Path("docs/benchmarks/attr_p0_smoke")
        summary = run_attr_p0_smoke(
            args.root,
            args.root / output,
            updates=args.updates or 2,
            seed=args.seed,
            device=args.device,
        )
    else:
        output = args.output_dir or Path(
            f"docs/benchmarks/asm_transformer_transition_risk/pilot_seed_{args.seed}"
        )
        summary = run_attr_train_only(
            args.root,
            args.root / output,
            updates=args.updates or 1000,
            seed=args.seed,
            device=args.device,
            train_world_count=64,
            validation_world_count=16,
            episodes_per_world=4,
        )
    print(summary)


if __name__ == "__main__":
    main()
