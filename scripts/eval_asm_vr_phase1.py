"""Run the isolated ASM-VR Phase 1 no-bypass experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aletheion_state_models.geometry.variable_rank import run_phase1_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_phase1_experiments(seed=args.seed)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
