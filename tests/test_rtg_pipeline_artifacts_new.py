import json
from pathlib import Path

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_artifacts import (
    atomic_write_json,
    canonical_sha256,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_artifacts import (
    _collect_paths,
    finalize_allowed_artifacts,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_seal import (
    BACKBONE_NAMES,
    CALIBRATION_NAMES,
    HEAD_NAMES,
)

ROOT = Path(__file__).resolve().parents[1]


def test_collect_paths_has_exact_seal_matrix_without_touching_tests(tmp_path):
    paths, validations, scores = _collect_paths(ROOT, ROOT / ".attr-rtg-test-artifacts")
    assert set(paths.backbone_checkpoints) == set(BACKBONE_NAMES)
    assert set(paths.head_checkpoints) == set(HEAD_NAMES)
    assert set(paths.calibration) == set(CALIBRATION_NAMES)
    assert len(paths.maps) == 2
    assert len(paths.statistics) == len(validations) == len(scores) == 10
    assert set(paths.manifests) == {"train", "validation", "calibration"}
    assert all("test_" not in Path(path).name for path in paths.manifests.values())


def test_finalize_fails_closed_before_writing_indexes_on_missing_matrix(tmp_path):
    body = {"schema": "ATTR-RTG-ALLOWED-MANIFEST-V1"}
    body["content_sha256"] = canonical_sha256(body)
    atomic_write_json(tmp_path / "allowed_manifest.json", body)
    with pytest.raises((ValueError, FileNotFoundError)):
        finalize_allowed_artifacts(ROOT, tmp_path)
    assert not (tmp_path / "validation_manifest.json").exists()
    assert not (tmp_path / "calibration_manifest.json").exists()
    assert not (tmp_path / "pipeline_paths.json").exists()
    assert json.loads((tmp_path / "allowed_manifest.json").read_text())["schema"].endswith("V1")
