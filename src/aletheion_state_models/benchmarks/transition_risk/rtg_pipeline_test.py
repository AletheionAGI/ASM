"""One-shot, fail-closed orchestration for registered ATTR-RTG tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .rtg_artifacts import atomic_write_json, file_sha256
from .rtg_pipeline_seal import PipelineSealPaths, open_pipeline_tests
from .rtg_registered_evaluation import evaluate_registered_test
from .rtg_seal import (
    complete_test_open,
    read_implementation_seal,
)
from .rtg_splits import make_test_worlds, split_spec

TEST_SPLITS = (
    ("test_id", 360101),
    ("test_shift", 360102),
    ("test_ood", 360103),
)


@dataclass(frozen=True)
class TestPipelineResult:
    """Paths and seal identity produced by a completed one-shot evaluation."""

    result_path: Path
    completion_path: Path
    seal_sha256: str


def run_test_pipeline(
    seal_path: str | Path,
    preregistration_manifest: str | Path,
    paths: PipelineSealPaths,
    result_path: str | Path,
) -> TestPipelineResult:
    """Open the seal, create each registered test split once, and evaluate once.

    Every output is create-only. An interrupted run leaves its opening receipt in
    place, so callers cannot retry after any test data might have been observed.
    """
    _validate_test_split_registry()
    seal_file = Path(seal_path)
    open_receipt = Path(f"{seal_file}.opened")
    completion_file = Path(f"{open_receipt}.completed")
    result_file = Path(result_path)
    _reject_existing_outputs(open_receipt, result_file, completion_file)

    seal = read_implementation_seal(seal_file)
    capability = open_pipeline_tests(
        seal_file,
        preregistration_manifest,
        paths,
    )

    split_worlds = {}
    for split_name, split_id in TEST_SPLITS:
        worlds = tuple(make_test_worlds(split_name, split_id, capability, seal.sha256))
        if not worlds:
            raise ValueError(f"registered test split is empty: {split_name}")
        split_worlds[split_name] = worlds

    evidence = capability.begin_evaluation(seal.sha256)
    evaluation = evaluate_registered_test(
        Path(paths.root).resolve(),
        paths,
        MappingProxyType(split_worlds),
        capability,
        evidence,
        seal.sha256,
    )
    atomic_write_json(
        result_file,
        {
            "kind": "attr_rtg_test_result",
            "schema_version": 1,
            "seal_sha256": seal.sha256,
            "splits": [
                {"name": name, "split_id": split_id} for name, split_id in TEST_SPLITS
            ],
            "evaluation": evaluation,
        },
    )
    completed = complete_test_open(
        capability,
        file_sha256(result_file),
    )
    return TestPipelineResult(result_file, completed, seal.sha256)


def _validate_test_split_registry() -> None:
    for split_name, split_id in TEST_SPLITS:
        spec = split_spec(split_name)
        if split_id != spec.split_seed:
            raise ValueError(
                f"registered split ID differs from frozen split seed: {split_name}"
            )


def _reject_existing_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(
                f"ATTR-RTG test pipeline output already exists: {path}"
            )
