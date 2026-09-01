#!/usr/bin/env python3
"""Train supplementary ASM-CM, ASM-VR-S, and ASM-R arms on frozen ATTR P1 data."""

from __future__ import annotations
import argparse
from pathlib import Path
from aletheion_state_models.benchmarks.transition_risk.supplementary import (
    run_supplementary_p1,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run_supplementary_p1(
        args.root, args.root / args.output, device=args.device
    )
    print(
        {"status": summary["status"], "arms": [item["arm"] for item in summary["arms"]]}
    )


if __name__ == "__main__":
    main()
