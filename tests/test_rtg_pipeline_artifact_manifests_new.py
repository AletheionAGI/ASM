import json

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_artifacts import (
    canonical_sha256,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_artifacts import (
    _manifest,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_truth import (
    write_training_manifest,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_training_manifest_validation import (
    validate_training_manifest,
)


def test_top_level_manifest_has_canonical_content_and_sorted_digests(tmp_path):
    first = tmp_path / "b.json"
    second = tmp_path / "a.json"
    first.write_text("b")
    second.write_text("a")
    manifest = _manifest("TOY-V1", {"z": first, "a": second})
    claimed = manifest.pop("content_sha256")
    assert claimed == canonical_sha256(manifest)
    assert [row["logical_name"] for row in manifest["artifacts"]] == ["a", "z"]
    assert all(len(row["sha256"]) == 64 for row in manifest["artifacts"])
    json.dumps(manifest, allow_nan=False)


def test_training_manifest_relationally_hashes_input_and_attached_ledgers(tmp_path):
    files = {}
    for name in ("allowed", "truth", "backbone", "head", "input", "attached"):
        files[name] = tmp_path / name
        files[name].write_text(name)
    manifest = tmp_path / "training_manifest.json"
    write_training_manifest(
        manifest, allowed_manifest=files["allowed"], truth_manifest=files["truth"],
        backbone_checkpoints={"arm": files["backbone"]},
        head_checkpoints={"arm.G": files["head"]},
        state_records={"arm.input.train": files["input"],
                       "arm.train": files["attached"]},
    )
    arguments = {
        "allowed_manifest": files["allowed"], "truth_manifest": files["truth"],
        "backbone_checkpoints": {"arm": files["backbone"]},
        "head_checkpoints": {"arm.G": files["head"]},
        "state_records": {"arm.state_inputs_train": files["input"],
                          "arm.states_train": files["attached"]},
    }
    validate_training_manifest(manifest, **arguments)
    files["input"].write_text("tampered")
    with pytest.raises(ValueError, match="relations or hashes"):
        validate_training_manifest(manifest, **arguments)
