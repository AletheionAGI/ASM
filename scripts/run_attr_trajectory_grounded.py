#!/usr/bin/env python3
"""Phase CLI for the sealed ATTR-TG1 run. It performs no work on import."""

from __future__ import annotations

import argparse
from pathlib import Path

from aletheion_state_models.benchmarks.transition_risk.trajectory_runner import (
    TrajectoryRunner,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_summary import (
    summarize_trajectory_grounded,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run one explicit ATTR-TG1 phase")
    value.add_argument(
        "phase",
        choices=(
            "preseal",
            "train",
            "validate",
            "validation-seal",
            "open-test",
            "evaluate",
            "summary",
        ),
    )
    value.add_argument("--root", type=Path, default=Path.cwd())
    value.add_argument("--arm", choices=("asm_x_base", "transformer_base"))
    value.add_argument("--seed", type=int, choices=(29, 43, 71, 89, 107))
    value.add_argument("--device", default="cpu")
    return value


def main() -> None:
    args = parser().parse_args()
    runner = TrajectoryRunner(args.root, device=args.device)
    if args.phase in {"train", "validate"} and (args.arm is None or args.seed is None):
        raise SystemExit("train/validate require --arm and --seed")
    if args.phase == "preseal":
        result = runner.preseal()
    elif args.phase == "train":
        result = runner.train(args.arm, args.seed)
    elif args.phase == "validate":
        result = runner.validate(args.arm, args.seed)
    elif args.phase == "validation-seal":
        result = runner.validation_seal()
    elif args.phase == "open-test":
        result = runner.open_test()
    elif args.phase == "evaluate":
        result = runner.evaluate_test()
    else:
        result = summarize_trajectory_grounded(runner.paths.predictions)
    print(result)


if __name__ == "__main__":
    main()
