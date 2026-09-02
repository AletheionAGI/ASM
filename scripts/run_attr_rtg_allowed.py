"""Thin CLI for allowed pre-test ATTR-RTG stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_artifacts import (
    finalize_allowed_artifacts,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_calibration import (
    run_calibration_extraction,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_train import (
    run_prepare,
    run_training,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_validation import (
    run_validation,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_training_data import (
    BehavioralEpisode,
    collate_ce_episodes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run only allowed ATTR-RTG pre-test stages")
    parser.add_argument("stage", choices=("prepare", "train", "validate", "calibrate", "finalize", "smoke"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/attr_rtg_allowed"))
    parser.add_argument("--device", default="cpu")
    return parser


def _smoke() -> None:
    """Exercise toy-only contracts without writing any sealable artifact."""
    episode = BehavioralEpisode("NON-SEALABLE-TOY", (1, 2, 3, 4))
    batch = collate_ce_episodes((episode,))
    if batch.targets[0, 3].item() != -100:
        raise RuntimeError("toy boundary smoke failed")
    print("NON-SEALABLE smoke passed; no artifacts were created")


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.stage == "prepare":
        print(run_prepare(arguments.root, arguments.output))
    elif arguments.stage == "train":
        results = run_training(arguments.root, arguments.output, updates=1_000, device=arguments.device)
        print(f"trained {len(results)} terminal backbone/head sets")
    elif arguments.stage == "validate":
        paths = run_validation(arguments.root, arguments.output, device=arguments.device)
        print(f"created {len(paths)} terminal validation artifacts")
    elif arguments.stage == "calibrate":
        paths = run_calibration_extraction(
            arguments.root, arguments.output, device=arguments.device
        )
        print(f"created {len(paths)} calibration score artifacts")
    elif arguments.stage == "finalize":
        print(finalize_allowed_artifacts(arguments.root, arguments.output))
    else:
        _smoke()


if __name__ == "__main__":
    main()
