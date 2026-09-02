"""Toy end-to-end checks for the strict ATTR-RTG pipeline seal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aletheion_state_models.benchmarks.transition_risk import (
    rtg_pipeline_seal,
    rtg_seal,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_artifacts import file_sha256
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_seal import (
    BACKBONE_NAMES,
    CALIBRATION_NAMES,
    HEAD_NAMES,
    MANIFEST_NAMES,
    MAP_NAMES,
    SOURCE_RELATIVE_PATHS,
    STATE_RECORD_NAMES,
    STATISTICS_NAMES,
    PipelineSealPaths,
    verify_pipeline_seal,
    write_pipeline_seal,
)


def _mapping(names: tuple[str, ...], path: Path) -> dict[str, Path]:
    return {name: path.name for name in names}


def _paths(artifact: Path) -> PipelineSealPaths:
    root = artifact.parent
    for relative in (
        *SOURCE_RELATIVE_PATHS,
        "src/aletheion_state_models/benchmarks/transition_risk/rtg_splits.py",
        "src/aletheion_state_models/benchmarks/transition_risk/rtg_registered_evaluation.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"canonical source")
    return PipelineSealPaths(
        root=root,
        sources={relative: relative for relative in SOURCE_RELATIVE_PATHS},
        generator="src/aletheion_state_models/benchmarks/transition_risk/rtg_splits.py",
        evaluator="src/aletheion_state_models/benchmarks/transition_risk/rtg_registered_evaluation.py",
        backbone_checkpoints=_mapping(BACKBONE_NAMES, artifact),
        head_checkpoints=_mapping(HEAD_NAMES, artifact),
        maps=_mapping(MAP_NAMES, artifact),
        statistics=_mapping(STATISTICS_NAMES, artifact),
        calibration=_mapping(CALIBRATION_NAMES, artifact),
        manifests=_mapping(MANIFEST_NAMES, artifact),
        state_records=_mapping(STATE_RECORD_NAMES, artifact),
    )


def test_toy_pipeline_matrix_writes_and_verifies_frozen_seal(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    preregistration = tmp_path / "prereg.json"
    preregistration.write_text(
        json.dumps(
            {"status": "FROZEN PREREGISTRATION", "scope": {"tests_opened": False}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rtg_seal, "PREREGISTRATION_MANIFEST_SHA256", file_sha256(preregistration)
    )
    monkeypatch.setattr(
        rtg_pipeline_seal,
        "verify_preregistration",
        lambda _root: file_sha256(preregistration),
    )
    paths = _paths(artifact)
    groups = paths.artifact_groups()
    assert tuple(groups) == rtg_seal.REQUIRED_GROUPS
    assert len(groups["checkpoints"]) == 40
    assert len(groups["maps"]) == 2
    assert len(groups["calibration"]) == 30
    assert len(groups["manifests"]) == 63

    seal_path = tmp_path / "implementation-seal.json"
    created = write_pipeline_seal(preregistration, paths, seal_path)
    assert created.preregistration_sha256 == file_sha256(preregistration)
    assert verify_pipeline_seal(seal_path, preregistration, paths) == created


def test_toy_pipeline_matrix_fails_closed_on_missing_head(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"fixed")
    paths = _paths(artifact)
    incomplete = dict(paths.head_checkpoints)
    incomplete.pop(HEAD_NAMES[-1])
    broken = PipelineSealPaths(**{**paths.__dict__, "head_checkpoints": incomplete})
    with pytest.raises(ValueError, match="head checkpoints matrix differs"):
        broken.artifact_groups()


def test_toy_pipeline_matrix_accepts_json_object_key_reordering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"fixed")
    paths = _paths(artifact)
    reordered = dict(reversed(tuple(paths.backbone_checkpoints.items())))
    roundtrip = PipelineSealPaths(
        **{**paths.__dict__, "backbone_checkpoints": reordered}
    )
    assert len(roundtrip.artifact_groups()["checkpoints"]) == 40
