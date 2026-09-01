"""Fail-closed canonical preseal and one-time ATTR-TG1 test opening."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .trajectory_checkpoint import (
    CheckpointPaths,
    FileDigest,
    TrajectoryCheckpoint,
    atomic_write_json,
    canonical_sha256,
    checkpoint_records,
    digest_files,
    verify_checkpoint_records,
)
from .trajectory_manifests import (
    SCHEMA_VERSION,
    ProceduralLeakageControl,
    SplitManifest,
    TG2Gates,
    TrajectoryProtocol,
    default_trajectory_protocol,
)


@dataclass(frozen=True)
class TrajectoryPreseal:
    schema_version: int
    protocol: TrajectoryProtocol
    code_manifest: tuple[FileDigest, ...]
    data_manifest: tuple[FileDigest, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise ValueError("preseal schema version must be an integer")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported trajectory preseal schema")
        if self.protocol != default_trajectory_protocol():
            raise ValueError("preseal protocol is not canonical TG1")
        if not self.code_manifest or not self.data_manifest:
            raise ValueError("code and common-data manifests are required")
        _validate_manifest(self.code_manifest)
        _validate_manifest(self.data_manifest)

    @property
    def sha256(self) -> str:
        return canonical_sha256(_preseal_payload(self))


@dataclass(frozen=True)
class TrajectorySeal:
    schema_version: int
    preseal: TrajectoryPreseal
    checkpoints: tuple[TrajectoryCheckpoint, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise ValueError("seal schema version must be an integer")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported trajectory seal schema")
        expected = tuple(
            (arm, seed)
            for arm in self.preseal.protocol.arms
            for seed in self.preseal.protocol.optimizer_seeds
        )
        actual = tuple((item.arm, item.optimizer_seed) for item in self.checkpoints)
        if actual != expected or len(self.checkpoints) != 10:
            raise ValueError("trajectory seal requires exactly 10 checkpoints")

    @property
    def sha256(self) -> str:
        return canonical_sha256(_seal_payload(self))


def create_trajectory_preseal(
    code_paths: Mapping[str, str | Path],
    data_paths: Mapping[str, str | Path],
    *,
    protocol: TrajectoryProtocol | None = None,
) -> TrajectoryPreseal:
    """Create specs and manifests only; no procedural test data is generated."""
    return TrajectoryPreseal(
        SCHEMA_VERSION,
        protocol or default_trajectory_protocol(),
        digest_files(code_paths),
        digest_files(data_paths),
    )


def create_trajectory_seal(
    preseal: TrajectoryPreseal, checkpoints: CheckpointPaths
) -> TrajectorySeal:
    """Bind the canonical pre-training artifact to the exact 10 checkpoints."""
    return TrajectorySeal(SCHEMA_VERSION, preseal, checkpoint_records(checkpoints))


def write_trajectory_preseal(preseal: TrajectoryPreseal, path: str | Path) -> Path:
    return atomic_write_json(
        path, _document("trajectory_preseal", _preseal_payload(preseal))
    )


def write_trajectory_seal(seal: TrajectorySeal, path: str | Path) -> Path:
    return atomic_write_json(path, _document("trajectory_seal", _seal_payload(seal)))


def read_trajectory_preseal(path: str | Path) -> TrajectoryPreseal:
    payload = _read_document(path, "trajectory_preseal")
    return _preseal_from_payload(payload)


def read_trajectory_seal(path: str | Path) -> TrajectorySeal:
    payload = _read_document(path, "trajectory_seal")
    return _seal_from_payload(payload)


def open_trajectory_seal(
    seal_path: str | Path,
    checkpoint_paths: CheckpointPaths,
    code_paths: Mapping[str, str | Path],
    data_paths: Mapping[str, str | Path],
    *,
    receipt_path: str | Path | None = None,
) -> tuple[SplitManifest, ...]:
    """Validate every artifact, consume the seal once, and return specs only."""
    path = Path(seal_path)
    seal = read_trajectory_seal(path)
    verify_checkpoint_records(seal.checkpoints, checkpoint_paths)
    if seal.preseal.code_manifest != digest_files(code_paths):
        raise ValueError("code manifest size or SHA256 mismatch")
    if seal.preseal.data_manifest != digest_files(data_paths):
        raise ValueError("data manifest size or SHA256 mismatch")
    receipt = (
        Path(receipt_path)
        if receipt_path
        else path.with_suffix(path.suffix + ".opened")
    )
    atomic_write_json(
        receipt,
        {
            "kind": "trajectory_test_open_receipt",
            "schema_version": SCHEMA_VERSION,
            "seal_sha256": seal.sha256,
        },
    )
    return tuple(
        split
        for split in seal.preseal.protocol.splits
        if split.name.startswith("test_")
    )


def _document(kind: str, payload: dict) -> dict:
    return {"kind": kind, "payload": payload, "sha256": canonical_sha256(payload)}


def _read_document(path: str | Path, expected_kind: str) -> Mapping[str, object]:
    try:
        document = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid trajectory seal JSON") from error
    if not isinstance(document, dict) or set(document) != {"kind", "payload", "sha256"}:
        raise ValueError("invalid trajectory seal document schema")
    payload = document["payload"]
    if document["kind"] != expected_kind or not isinstance(payload, dict):
        raise ValueError("wrong trajectory seal kind or payload schema")
    if canonical_sha256(payload) != document["sha256"]:
        raise ValueError("trajectory seal SHA256 mismatch")
    return payload


def _preseal_payload(preseal: TrajectoryPreseal) -> dict:
    return {
        "schema_version": preseal.schema_version,
        "protocol": asdict(preseal.protocol),
        "code_manifest": [asdict(item) for item in preseal.code_manifest],
        "data_manifest": [asdict(item) for item in preseal.data_manifest],
    }


def _seal_payload(seal: TrajectorySeal) -> dict:
    return {
        "schema_version": seal.schema_version,
        "preseal": _preseal_payload(seal.preseal),
        "preseal_sha256": seal.preseal.sha256,
        "checkpoints": [asdict(item) for item in seal.checkpoints],
    }


def _preseal_from_payload(payload: Mapping[str, object]) -> TrajectoryPreseal:
    if set(payload) != {"schema_version", "protocol", "code_manifest", "data_manifest"}:
        raise ValueError("invalid trajectory preseal payload schema")
    try:
        raw = payload["protocol"]
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "arms",
            "optimizer_seeds",
            "horizon",
            "negative_samples",
            "splits",
            "leakage_control",
            "gates",
            "bootstrap_seed",
            "bootstrap_replicates",
        }:
            raise ValueError("invalid trajectory protocol schema")
        protocol = TrajectoryProtocol(
            raw["schema_version"],
            tuple(raw["arms"]),
            tuple(raw["optimizer_seeds"]),
            raw["horizon"],
            raw["negative_samples"],
            tuple(SplitManifest(**item) for item in raw["splits"]),
            ProceduralLeakageControl(**raw["leakage_control"]),
            TG2Gates(**raw["gates"]),
            raw["bootstrap_seed"],
            raw["bootstrap_replicates"],
        )
        return TrajectoryPreseal(
            payload["schema_version"],
            protocol,
            tuple(FileDigest(**item) for item in payload["code_manifest"]),
            tuple(FileDigest(**item) for item in payload["data_manifest"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("invalid trajectory preseal value schema") from error


def _seal_from_payload(payload: Mapping[str, object]) -> TrajectorySeal:
    if set(payload) != {"schema_version", "preseal", "preseal_sha256", "checkpoints"}:
        raise ValueError("invalid trajectory seal payload schema")
    try:
        preseal = _preseal_from_payload(payload["preseal"])
        if preseal.sha256 != payload["preseal_sha256"]:
            raise ValueError("trajectory preseal SHA256 mismatch")
        checkpoints = tuple(
            TrajectoryCheckpoint(**item) for item in payload["checkpoints"]
        )
        return TrajectorySeal(payload["schema_version"], preseal, checkpoints)
    except (KeyError, TypeError) as error:
        raise ValueError("invalid trajectory seal value schema") from error


def _validate_manifest(items: tuple[FileDigest, ...]) -> None:
    names = tuple(item.name for item in items)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError("manifest entries must be unique and sorted")


create_preseal = create_trajectory_preseal
create_seal = create_trajectory_seal
open_seal = open_trajectory_seal
read_preseal = read_trajectory_preseal
read_seal = read_trajectory_seal
write_preseal = write_trajectory_preseal
write_seal = write_trajectory_seal

__all__ = [
    "TrajectoryPreseal",
    "TrajectorySeal",
    "create_preseal",
    "create_seal",
    "create_trajectory_preseal",
    "create_trajectory_seal",
    "open_seal",
    "open_trajectory_seal",
    "read_preseal",
    "read_seal",
    "read_trajectory_preseal",
    "read_trajectory_seal",
    "write_preseal",
    "write_seal",
    "write_trajectory_preseal",
    "write_trajectory_seal",
]
