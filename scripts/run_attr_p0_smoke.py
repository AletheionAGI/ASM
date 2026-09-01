#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from aletheion_state_models.benchmarks.transition_risk.runner import run_attr_p0_smoke


def main():
    parser = argparse.ArgumentParser(description="Run the unsealed ATTR P0 CPU smoke")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/benchmarks/attr_p0_smoke")
    )
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = run_attr_p0_smoke(
        args.root,
        args.root / args.output_dir,
        updates=args.updates,
        seed=args.seed,
        device=args.device,
    )
    print(summary)


if __name__ == "__main__":
    main()
