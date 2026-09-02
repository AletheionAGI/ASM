import json
from types import SimpleNamespace

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_truth import (
    audit_materialized_truths,
    write_training_manifest,
)


def origin(split, index):
    safe = (0, 1, 2, 3, 4, 1, 1, 10, 0, 0, 0)
    unsafe = (0, 1, 2, 0, 4, 1, 1, 10, 0, 0, 0)
    candidates = tuple(
        SimpleNamespace(target=unsafe if action % 2 else safe, unsafe=bool(action % 2))
        for action in range(6)
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(
            split_id=split, world_id=f"{split}-w{index // 10}",
            episode_id=f"{split}-e{index}", t=1,
        ),
        truth=SimpleNamespace(
            persistence_target=safe, candidates=candidates, failure_delay=3,
        ),
    )


def test_truth_manifest_audits_counts_classes_hashes_and_disjunction():
    groups = {
        split: tuple(origin(split, index) for index in range(500 if split == "train" else 100))
        for split in ("train", "validation", "calibration")
    }
    manifest = audit_materialized_truths(groups)
    assert [row["positive_labels"] for row in manifest["splits"]] == [1500, 300, 300]
    assert all(len(row["truth_sha256"]) == 64 for row in manifest["splits"])
    assert "target" not in json.dumps(manifest)


def test_training_manifest_references_all_artifact_classes_create_only(tmp_path):
    names = ("allowed", "truth", "backbone", "head", "states")
    files = {}
    for name in names:
        files[name] = tmp_path / f"{name}.bin"
        files[name].write_bytes(name.encode())
    destination = tmp_path / "training_manifest.json"
    write_training_manifest(
        destination,
        allowed_manifest=files["allowed"], truth_manifest=files["truth"],
        backbone_checkpoints={"arm": files["backbone"]},
        head_checkpoints={"arm.G": files["head"]},
        state_records={"arm.train": files["states"]},
    )
    payload = json.loads(destination.read_text())
    assert payload["schema"] == "ATTR-RTG-TRAINING-MANIFEST-V1"
    assert len(payload["artifacts"]) == 5
    with pytest.raises(FileExistsError):
        write_training_manifest(
            destination,
            allowed_manifest=files["allowed"], truth_manifest=files["truth"],
            backbone_checkpoints={"arm": files["backbone"]},
            head_checkpoints={"arm.G": files["head"]},
            state_records={"arm.train": files["states"]},
        )
