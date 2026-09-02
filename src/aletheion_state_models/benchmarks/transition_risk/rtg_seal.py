"""Fail-closed ATTR-RTG implementation seal and one-shot test opening."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .rtg_artifacts import (
    ArtifactDigest,
    atomic_write_json,
    canonical_sha256,
    digest_files,
    digest_payload,
    file_sha256,
    verify_files,
)
from .rtg_config import PREREGISTRATION_MANIFEST_SHA256
from .rtg_integrity import (
    IntegrityEvidence,
    _issue_integrity_evidence,
)

SCHEMA_VERSION = 1
REQUIRED_GROUPS = (
    "sources",
    "generator",
    "evaluator",
    "calibration",
    "manifests",
    "maps",
    "checkpoints",
)
_CAPABILITY_TOKEN = object()
_REGISTERED_TEST_SPLITS = frozenset({"test_id", "test_shift", "test_ood"})


@dataclass(frozen=True)
class ArtifactGroup:
    name: str
    records: tuple[ArtifactDigest, ...]

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_GROUPS or not self.records:
            raise ValueError("invalid or empty implementation artifact group")
        if tuple(sorted(self.records)) != self.records:
            raise ValueError("artifact records must be canonically ordered")


@dataclass(frozen=True)
class ImplementationSeal:
    schema_version: int
    preregistration_sha256: str
    groups: tuple[ArtifactGroup, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported implementation seal schema")
        if len(self.preregistration_sha256) != 64:
            raise ValueError("invalid preregistration SHA-256")
        try:
            int(self.preregistration_sha256, 16)
        except ValueError as exc:
            raise ValueError("invalid preregistration SHA-256") from exc
        if tuple(group.name for group in self.groups) != REQUIRED_GROUPS:
            raise ValueError("implementation seal requires every canonical group")

    @property
    def sha256(self) -> str:
        return canonical_sha256(_seal_payload(self))


class TestOpenCapability:
    __test__ = False
    __slots__ = (
        "_evaluation_claimed",
        "_evidence",
        "_generated_splits",
        "_pipeline_verified",
        "_state",
        "receipt_path",
        "seal_sha256",
    )

    def __init__(self, seal_sha256: str, receipt_path: Path, token: object) -> None:
        if token is not _CAPABILITY_TOKEN:
            raise TypeError("test-open capabilities are issued only by seal opening")
        self.seal_sha256 = seal_sha256
        self.receipt_path = receipt_path
        self._generated_splits: set[str] = set()
        self._evaluation_claimed = False
        self._pipeline_verified = False
        self._evidence: IntegrityEvidence | None = None
        self._state = "OPEN"

    def _register_split_generation(self, split_name: str) -> None:
        if self._state != "OPEN":
            raise PermissionError("test-open capability is not open")
        if split_name not in _REGISTERED_TEST_SPLITS:
            raise PermissionError("split is not registered for test generation")
        if split_name in self._generated_splits:
            raise PermissionError(
                f"registered test split was already generated: {split_name}"
            )
        self._generated_splits.add(split_name)

    def _mark_pipeline_verified(self, seal_sha256: str) -> None:
        if self._state != "OPEN" or seal_sha256 != self.seal_sha256:
            raise PermissionError(
                "exact pipeline verification does not match capability"
            )
        self._pipeline_verified = True

    def require_all_splits_generated(self) -> None:
        if self._state != "OPEN":
            raise PermissionError("test-open capability is not open")
        if self._generated_splits != _REGISTERED_TEST_SPLITS:
            missing = sorted(_REGISTERED_TEST_SPLITS - self._generated_splits)
            raise PermissionError(
                f"exactly three registered split generations required; missing={missing}"
            )

    def begin_evaluation(self, seal_sha256: str) -> IntegrityEvidence:
        self.require_all_splits_generated()
        if seal_sha256 != self.seal_sha256:
            raise PermissionError("evaluation seal SHA-256 differs from capability")
        if not self._pipeline_verified:
            raise PermissionError("exact pipeline verification is required")
        self._state = "EVALUATING"
        evidence = _issue_integrity_evidence(seal_sha256)
        self._evidence = evidence
        return evidence

    def _claim_evaluation(self) -> None:
        if self._state != "EVALUATING" or self._evaluation_claimed:
            raise PermissionError("registered evaluation was already claimed")
        self._evaluation_claimed = True

    def consume(self) -> None:
        if self._state != "EVALUATING":
            raise PermissionError("completion requires an evaluating capability")
        self._state = "CONSUMED"

    @property
    def state(self) -> str:
        return self._state

    @property
    def valid(self) -> bool:
        return self._state != "CONSUMED"


def create_implementation_seal(
    preregistration_manifest: str | Path,
    artifact_paths: Mapping[str, Mapping[str, str | Path]],
) -> ImplementationSeal:
    manifest_path = Path(preregistration_manifest)
    manifest_digest = file_sha256(manifest_path)
    if manifest_digest != PREREGISTRATION_MANIFEST_SHA256:
        raise ValueError("preregistration manifest differs from the frozen hash")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "FROZEN PREREGISTRATION"
        or manifest.get("scope", {}).get("tests_opened") is not False
    ):
        raise ValueError("preregistration is not frozen and unopened")
    if tuple(artifact_paths) != REQUIRED_GROUPS:
        raise ValueError("artifact groups must use canonical order")
    groups = tuple(
        ArtifactGroup(name, digest_files(artifact_paths[name]))
        for name in REQUIRED_GROUPS
    )
    return ImplementationSeal(SCHEMA_VERSION, manifest_digest, groups)


def write_implementation_seal(seal: ImplementationSeal, path: str | Path) -> Path:
    return atomic_write_json(
        path,
        {
            "kind": "attr_rtg_implementation_seal",
            "payload": _seal_payload(seal),
            "sha256": seal.sha256,
        },
    )


def read_implementation_seal(path: str | Path) -> ImplementationSeal:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        set(document) != {"kind", "payload", "sha256"}
        or document["kind"] != "attr_rtg_implementation_seal"
    ):
        raise ValueError("invalid implementation seal document")
    payload = document["payload"]
    if canonical_sha256(payload) != document["sha256"]:
        raise ValueError("implementation seal payload SHA-256 mismatch")
    groups = tuple(
        ArtifactGroup(
            group["name"],
            tuple(ArtifactDigest(**record) for record in group["records"]),
        )
        for group in payload["groups"]
    )
    return ImplementationSeal(
        payload["schema_version"], payload["preregistration_sha256"], groups
    )


def open_implementation_seal(
    seal_path: str | Path,
    preregistration_manifest: str | Path,
    artifact_paths: Mapping[str, Mapping[str, str | Path]],
) -> TestOpenCapability:
    seal = read_implementation_seal(seal_path)
    manifest_digest = file_sha256(preregistration_manifest)
    if manifest_digest != PREREGISTRATION_MANIFEST_SHA256:
        raise ValueError("preregistration manifest differs from the frozen hash")
    if manifest_digest != seal.preregistration_sha256:
        raise ValueError("preregistration manifest SHA-256 mismatch")
    if tuple(artifact_paths) != REQUIRED_GROUPS:
        raise ValueError("artifact groups must use canonical order")
    for group in seal.groups:
        verify_files(group.records, artifact_paths[group.name])
    destination = Path(f"{seal_path}.opened")
    atomic_write_json(
        destination,
        {
            "kind": "attr_rtg_test_open_receipt",
            "schema_version": SCHEMA_VERSION,
            "seal_sha256": seal.sha256,
            "status": "opened",
        },
    )
    return TestOpenCapability(seal.sha256, destination, _CAPABILITY_TOKEN)


def complete_test_open(
    capability: TestOpenCapability,
    result_sha256: str,
) -> Path:
    if len(result_sha256) != 64:
        raise ValueError("invalid result SHA-256")
    try:
        int(result_sha256, 16)
    except ValueError as exc:
        raise ValueError("invalid result SHA-256") from exc
    capability.consume()
    return atomic_write_json(
        f"{capability.receipt_path}.completed",
        {
            "kind": "attr_rtg_test_completion_receipt",
            "schema_version": SCHEMA_VERSION,
            "seal_sha256": capability.seal_sha256,
            "result_sha256": result_sha256,
            "status": "completed",
        },
    )


def require_test_capability(capability: TestOpenCapability, seal_sha256: str) -> None:
    if not isinstance(capability, TestOpenCapability) or not capability.valid:
        raise PermissionError("valid test-open capability required")
    if capability.seal_sha256 != seal_sha256:
        raise PermissionError("capability belongs to another implementation seal")


def require_evaluation_authority(
    capability: TestOpenCapability,
    evidence: IntegrityEvidence,
    seal_sha256: str,
) -> None:
    require_test_capability(capability, seal_sha256)
    if capability.state != "EVALUATING":
        raise PermissionError("evaluation requires an evaluating capability")
    if not isinstance(evidence, IntegrityEvidence):
        raise PermissionError("opaque pipeline integrity evidence required")
    if evidence is not capability._evidence or evidence.seal_sha256 != seal_sha256:
        raise PermissionError("integrity evidence does not match evaluation authority")


def claim_evaluation_authority(capability: TestOpenCapability, evidence: IntegrityEvidence,
                               seal_sha256: str) -> None:
    require_evaluation_authority(capability, evidence, seal_sha256)
    capability._claim_evaluation()


def _seal_payload(seal: ImplementationSeal) -> dict[str, object]:
    return {
        "schema_version": seal.schema_version,
        "preregistration_sha256": seal.preregistration_sha256,
        "groups": [
            {"name": group.name, "records": digest_payload(group.records)}
            for group in seal.groups
        ],
    }
