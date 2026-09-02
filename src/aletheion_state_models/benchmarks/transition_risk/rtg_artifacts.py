"""Canonical, immutable artifact primitives for ATTR-RTG."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class ArtifactDigest:
    logical_name: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.logical_name or self.bytes < 0 or len(self.sha256) != 64:
            raise ValueError("invalid artifact digest")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("invalid SHA-256") from exc


def canonical_json(value: object) -> bytes:
    """Encode JSON deterministically and reject NaN/Infinity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_files(paths: Mapping[str, str | Path]) -> tuple[ArtifactDigest, ...]:
    if not paths:
        raise ValueError("artifact mapping must not be empty")
    records = []
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not name or not path.is_file():
            raise ValueError(f"missing artifact: {name}")
        records.append(ArtifactDigest(name, path.stat().st_size, file_sha256(path)))
    return tuple(records)


def verify_files(
    records: Sequence[ArtifactDigest], paths: Mapping[str, str | Path]
) -> None:
    if tuple(records) != digest_files(paths):
        raise ValueError("artifact manifest size or SHA-256 mismatch")


def atomic_write_json(path: str | Path, value: object) -> Path:
    """Create canonical JSON atomically; never overwrite an artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value) + b"\n"
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def digest_payload(records: Sequence[ArtifactDigest]) -> list[dict[str, object]]:
    return [asdict(record) for record in records]
