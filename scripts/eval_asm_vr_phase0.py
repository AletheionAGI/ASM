"""Run the isolated ASM-VR Phase 0 invariant experiments."""

from __future__ import annotations

import argparse
import json

from aletheion_state_models.geometry.variable_rank import run_phase0_experiments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = run_phase0_experiments(samples=args.samples, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
