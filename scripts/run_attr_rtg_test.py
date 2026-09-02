"""Run the canonical sealed ATTR-RTG tests exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_seal import (
    PipelineSealPaths,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_test import (
    run_test_pipeline,
)

_MAPPING_FIELDS = (
    "sources",
    "backbone_checkpoints",
    "head_checkpoints",
    "maps",
    "statistics",
    "calibration",
    "manifests",
    "state_records",
)
_SCALAR_FIELDS = ("root", "generator", "evaluator")


def _load_pipeline_paths(path: Path) -> PipelineSealPaths:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = set(_MAPPING_FIELDS + _SCALAR_FIELDS) | {"schema"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("artifact manifest must contain the complete pipeline matrix")
    if raw.pop("schema") != "PipelineSealPaths":
        raise ValueError("artifact manifest schema differs")
    for field in _MAPPING_FIELDS:
        value = raw[field]
        if not isinstance(value, dict) or not value:
            raise ValueError(f"pipeline artifact field must be nonempty: {field}")
        if any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in value.items()
        ):
            raise ValueError(
                f"pipeline artifact names and paths must be strings: {field}"
            )
    for field in _SCALAR_FIELDS:
        if not isinstance(raw[field], str) or not raw[field]:
            raise ValueError(f"pipeline artifact path must be a string: {field}")
    return PipelineSealPaths(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument("--preregistration-manifest", required=True, type=Path)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    outcome = run_test_pipeline(
        args.seal,
        args.preregistration_manifest,
        _load_pipeline_paths(args.artifact_manifest),
        args.result,
    )
    print(outcome.completion_path)


if __name__ == "__main__":
    main()
