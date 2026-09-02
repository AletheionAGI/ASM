from __future__ import annotations

import inspect
from collections import OrderedDict
from pathlib import Path

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_artifacts import file_sha256
from aletheion_state_models.benchmarks.transition_risk.rtg_integrity import (
    IntegrityEvidence,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_seal import (
    REQUIRED_GROUPS,
    TestOpenCapability,
    complete_test_open,
    create_implementation_seal,
    open_implementation_seal,
    read_implementation_seal,
    require_test_capability,
    write_implementation_seal,
)


def _fixture(tmp_path):
    prereg = Path("docs/ATTR_RTG_PREREGISTRATION_MANIFEST.json")
    paths = OrderedDict()
    for group in REQUIRED_GROUPS:
        artifact = tmp_path / f"{group}.bin"
        artifact.write_bytes(group.encode())
        paths[group] = {group: artifact}
    return prereg, paths


def test_seal_roundtrip_open_once_and_completion(tmp_path):
    prereg, paths = _fixture(tmp_path)
    seal = create_implementation_seal(prereg, paths)
    seal_path = write_implementation_seal(seal, tmp_path / "seal.json")
    assert read_implementation_seal(seal_path) == seal

    capability = open_implementation_seal(seal_path, prereg, paths)
    assert not hasattr(capability, "register_split_generation")
    require_test_capability(capability, seal.sha256)
    assert capability.receipt_path.exists()
    with pytest.raises(FileExistsError):
        open_implementation_seal(seal_path, prereg, paths)

    capability._register_split_generation("test_id")
    assert capability.valid
    with pytest.raises(PermissionError, match="already generated"):
        capability._register_split_generation("test_id")
    capability._register_split_generation("test_shift")
    capability._register_split_generation("test_ood")
    capability.require_all_splits_generated()
    assert capability.valid
    with pytest.raises(PermissionError, match="exact pipeline verification"):
        capability.begin_evaluation(seal.sha256)
    result = tmp_path / "result.json"
    result.write_text("result")
    with pytest.raises(PermissionError, match="evaluating"):
        complete_test_open(capability, file_sha256(result))


def test_open_fails_closed_on_tamper_without_receipt(tmp_path):
    prereg, paths = _fixture(tmp_path)
    seal_path = write_implementation_seal(
        create_implementation_seal(prereg, paths), tmp_path / "seal.json"
    )
    paths["maps"]["maps"].write_text("tampered")
    receipt = Path(f"{seal_path}.opened")
    with pytest.raises(ValueError, match="mismatch"):
        open_implementation_seal(seal_path, prereg, paths)
    assert not receipt.exists()


def test_preregistration_hash_is_pinned(tmp_path):
    _, paths = _fixture(tmp_path)
    altered = tmp_path / "altered-prereg.json"
    altered.write_text(
        '{"scope":{"tests_opened":true},"status":"FROZEN PREREGISTRATION"}'
    )
    with pytest.raises(ValueError, match="frozen hash"):
        create_implementation_seal(altered, paths)


def test_capability_cannot_be_constructed_by_callers(tmp_path):
    with pytest.raises(TypeError, match="issued only"):
        TestOpenCapability("0" * 64, tmp_path / "receipt", object())


def test_integrity_evidence_cannot_be_constructed_by_callers():
    with pytest.raises(TypeError, match="issued only"):
        IntegrityEvidence("0" * 64, object())


def test_public_open_and_completion_apis_have_no_path_override():
    assert "receipt_path" not in inspect.signature(open_implementation_seal).parameters
    assert "path" not in inspect.signature(complete_test_open).parameters
