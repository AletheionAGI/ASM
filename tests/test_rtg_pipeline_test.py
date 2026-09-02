from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from aletheion_state_models.benchmarks.transition_risk import rtg_pipeline_test
from aletheion_state_models.benchmarks.transition_risk.rtg_integrity import (
    require_integrity_evidence,
)
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
    create_pipeline_seal,
    open_pipeline_tests,
    verify_pipeline_seal,
    write_pipeline_seal,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_test import (
    TEST_SPLITS,
    run_test_pipeline,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_seal import (
    claim_evaluation_authority,
)


def _sealed_fixture(tmp_path: Path):
    frozen_manifest = Path("docs/ATTR_RTG_PREREGISTRATION_MANIFEST.json")
    frozen = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    for record in frozen["artifacts"]:
        source = Path(record["path"])
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    preregistration = tmp_path / frozen_manifest
    preregistration.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(frozen_manifest, preregistration)
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"toy sealed artifact")

    def matrix(names):
        return {name: artifact.name for name in names}

    for relative in (
        *SOURCE_RELATIVE_PATHS,
        "src/aletheion_state_models/benchmarks/transition_risk/rtg_splits.py",
        "src/aletheion_state_models/benchmarks/transition_risk/rtg_registered_evaluation.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"canonical source")
    paths = PipelineSealPaths(
        root=tmp_path,
        sources={relative: relative for relative in SOURCE_RELATIVE_PATHS},
        generator="src/aletheion_state_models/benchmarks/transition_risk/rtg_splits.py",
        evaluator="src/aletheion_state_models/benchmarks/transition_risk/rtg_registered_evaluation.py",
        backbone_checkpoints=matrix(BACKBONE_NAMES),
        head_checkpoints=matrix(HEAD_NAMES),
        maps=matrix(MAP_NAMES),
        statistics=matrix(STATISTICS_NAMES),
        calibration=matrix(CALIBRATION_NAMES),
        manifests=matrix(MANIFEST_NAMES),
        state_records=matrix(STATE_RECORD_NAMES),
    )
    seal_path = tmp_path / "seal.json"
    write_pipeline_seal(preregistration, paths, seal_path)
    return preregistration, paths, seal_path


def test_pipeline_uses_only_canonical_generator_and_evaluator(tmp_path, monkeypatch):
    preregistration, artifacts, seal_path = _sealed_fixture(tmp_path)
    receipt = Path(f"{seal_path}.opened")
    calls = []

    def make_worlds(name, split_id, capability, seal_sha256):
        assert receipt.exists()
        assert capability.valid
        assert capability.seal_sha256 == seal_sha256
        capability._register_split_generation(name)
        calls.append((name, split_id))
        return (f"{name}-toy-world",)

    def evaluate(root, paths, splits, capability, evidence, seal_sha256):
        calls.append("evaluate")
        assert root == Path(paths.root).resolve()
        assert capability.state == "EVALUATING"
        assert evidence.seal_sha256 == seal_sha256
        assert require_integrity_evidence(evidence, seal_sha256) is True
        claim_evaluation_authority(capability, evidence, seal_sha256)
        with pytest.raises(PermissionError, match="already claimed"):
            claim_evaluation_authority(capability, evidence, seal_sha256)
        with pytest.raises(PermissionError, match="not open"):
            capability.begin_evaluation(seal_sha256)
        assert capability.seal_sha256 == seal_sha256
        assert paths is artifacts
        assert tuple(splits) == tuple(name for name, _ in TEST_SPLITS)
        return {"worlds": sum(len(worlds) for worlds in splits.values())}

    monkeypatch.setattr(rtg_pipeline_test, "make_test_worlds", make_worlds)
    monkeypatch.setattr(rtg_pipeline_test, "evaluate_registered_test", evaluate)
    outcome = run_test_pipeline(
        seal_path,
        preregistration,
        artifacts,
        tmp_path / "result.json",
    )

    assert calls == [*TEST_SPLITS, "evaluate"]
    assert outcome.result_path.exists()
    assert outcome.completion_path.exists()


@pytest.mark.parametrize("existing", ["receipt", "result", "completion"])
def test_pipeline_blocks_any_existing_output_before_opening(
    tmp_path, existing, monkeypatch
):
    preregistration, artifacts, seal_path = _sealed_fixture(tmp_path)
    receipt = Path(f"{seal_path}.opened")
    paths = {
        "receipt": receipt,
        "result": tmp_path / "result.json",
        "completion": Path(f"{receipt}.completed"),
    }
    paths[existing].write_text("already exists", encoding="utf-8")

    def unexpected(*_args):
        raise AssertionError("canonical test generator/evaluator must not run")

    monkeypatch.setattr(rtg_pipeline_test, "make_test_worlds", unexpected)
    monkeypatch.setattr(rtg_pipeline_test, "evaluate_registered_test", unexpected)
    with pytest.raises(FileExistsError):
        run_test_pipeline(
            seal_path,
            preregistration,
            artifacts,
            paths["result"],
        )

    if existing != "receipt":
        assert not paths["receipt"].exists()


def test_failed_evaluation_leaves_receipt_and_cannot_rerun(tmp_path, monkeypatch):
    preregistration, artifacts, seal_path = _sealed_fixture(tmp_path)
    receipt = Path(f"{seal_path}.opened")

    monkeypatch.setattr(
        rtg_pipeline_test,
        "make_test_worlds",
        lambda name, split_id, capability, digest: (
            capability._register_split_generation(name) or name,
        ),
    )

    def fail(_root, _paths, _splits, capability, evidence, seal_sha256):
        assert capability.state == "EVALUATING"
        claim_evaluation_authority(capability, evidence, seal_sha256)
        raise RuntimeError("toy evaluator failed")

    monkeypatch.setattr(rtg_pipeline_test, "evaluate_registered_test", fail)
    arguments = (
        seal_path,
        preregistration,
        artifacts,
        tmp_path / "result.json",
    )
    with pytest.raises(RuntimeError, match="toy evaluator"):
        run_test_pipeline(*arguments)
    assert receipt.exists()
    with pytest.raises(FileExistsError):
        run_test_pipeline(*arguments)


def test_pipeline_has_no_receipt_or_completion_path_override():
    import inspect

    parameters = inspect.signature(run_test_pipeline).parameters
    assert "receipt_path" not in parameters
    assert "completion_path" not in parameters


def test_pipeline_rejects_split_id_that_differs_from_frozen_seed(monkeypatch):
    monkeypatch.setattr(
        rtg_pipeline_test,
        "TEST_SPLITS",
        (("test_id", 999999),),
    )
    with pytest.raises(ValueError, match="differs from frozen split seed"):
        run_test_pipeline("unused", "unused", object(), "unused")


def test_verify_and_open_recheck_all_frozen_preregistration_artifacts(tmp_path):
    preregistration, paths, seal_path = _sealed_fixture(tmp_path)
    registered_config = tmp_path / "configs/rtg_asm_30k_seed29.yaml"
    registered_config.write_text("tampered after seal", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact (size|hash) mismatch"):
        create_pipeline_seal(preregistration, paths)
    with pytest.raises(ValueError, match="artifact (size|hash) mismatch"):
        verify_pipeline_seal(seal_path, preregistration, paths)
    with pytest.raises(ValueError, match="artifact (size|hash) mismatch"):
        open_pipeline_tests(seal_path, preregistration, paths)
    assert not Path(f"{seal_path}.opened").exists()
