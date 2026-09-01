"""Checkpoint hashing and atomic JSON primitives for ATTR-TG1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .trajectory_manifests import (
    ARMS,
    HORIZON,
    NEGATIVE_SAMPLES,
    OPTIMIZER_SEEDS,
    SCHEMA_VERSION,
)

REQUIRED_CHECKPOINTS = len(ARMS) * len(OPTIMIZER_SEEDS)
CheckpointPaths = Mapping[tuple[str, int], str | Path]


@dataclass(frozen=True)
class FileDigest:
    name: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if type(self.size) is not int:
            raise ValueError("file digest size must be an integer")
        if not self.name or self.size < 0 or not _valid_sha256(self.sha256):
            raise ValueError("invalid file digest")


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    arm: str
    optimizer_seed: int
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if type(self.optimizer_seed) is not int or type(self.size) is not int:
            raise ValueError("checkpoint seed and size must be integers")
        if self.arm not in ARMS or self.optimizer_seed not in OPTIMIZER_SEEDS:
            raise ValueError("unregistered trajectory checkpoint")
        if self.size < 0 or not _valid_sha256(self.sha256):
            raise ValueError("invalid trajectory checkpoint digest")


def canonical_json(value: object) -> bytes:
    """Return strict canonical JSON. NaN and infinities are always rejected."""
    _require_finite(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value is not strict JSON") from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_files(paths: Mapping[str, str | Path]) -> tuple[FileDigest, ...]:
    if not paths or any(not isinstance(name, str) or not name for name in paths):
        raise ValueError("file manifest cannot be empty")
    result = []
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError("manifest file is missing")
        result.append(FileDigest(name, file_sha256(path), path.stat().st_size))
    return tuple(result)


def checkpoint_records(paths: CheckpointPaths) -> tuple[TrajectoryCheckpoint, ...]:
    expected = {(arm, seed) for arm in ARMS for seed in OPTIMIZER_SEEDS}
    if set(paths) != expected:
        raise ValueError("TG1 requires the exact two-arm/five-seed checkpoint matrix")
    resolved = [Path(path).resolve() for path in paths.values()]
    if len(set(resolved)) != REQUIRED_CHECKPOINTS:
        raise ValueError("TG1 checkpoints must be 10 distinct files")
    records = []
    for arm in ARMS:
        for seed in OPTIMIZER_SEEDS:
            path = Path(paths[(arm, seed)])
            if not path.is_file():
                raise ValueError("all 10 TG1 checkpoints must exist")
            validate_checkpoint_file(path, arm=arm, optimizer_seed=seed)
            records.append(
                TrajectoryCheckpoint(arm, seed, file_sha256(path), path.stat().st_size)
            )
    return tuple(records)


def verify_checkpoint_records(
    records: Sequence[TrajectoryCheckpoint], paths: CheckpointPaths
) -> None:
    expected = tuple((arm, seed) for arm in ARMS for seed in OPTIMIZER_SEEDS)
    actual = tuple((item.arm, item.optimizer_seed) for item in records)
    if len(records) != REQUIRED_CHECKPOINTS or actual != expected:
        raise ValueError("checkpoint records are missing, extra, or out of order")
    if tuple(records) != checkpoint_records(paths):
        raise ValueError("checkpoint size or SHA256 mismatch")


def atomic_write_json(path: str | Path, value: object) -> Path:
    """Create immutable canonical JSON atomically; never replace an artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
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


def checkpoint_metadata(arm: str, optimizer_seed: int) -> dict[str, object]:
    if not isinstance(arm, str) or type(optimizer_seed) is not int:
        raise ValueError("checkpoint identity has invalid types")
    if arm not in ARMS or optimizer_seed not in OPTIMIZER_SEEDS:
        raise ValueError("unregistered trajectory checkpoint identity")
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "optimizer_seed": optimizer_seed,
        "horizon": HORIZON,
        "k": NEGATIVE_SAMPLES,
        "test_opened": False,
    }


def save_terminal_checkpoint(
    path: str | Path,
    adapter: torch.nn.Module,
    heads: torch.nn.Module,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Atomically save finite CPU state without optimizer or test artifacts."""
    _validate_metadata(metadata)
    payload = {
        "model_state": _cpu_state(adapter.model),
        "heads_state": _cpu_state(heads),
        "metadata": dict(metadata),
    }
    _validate_state(payload["model_state"])
    _validate_state(payload["heads_state"])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": destination.as_posix(),
        "sha256": file_sha256(destination),
        "metadata": dict(metadata),
    }


def load_terminal_checkpoint(
    path: str | Path,
    adapter: torch.nn.Module,
    heads: torch.nn.Module,
    *,
    device: str | torch.device,
) -> dict[str, object]:
    """Validate the complete payload before mutating either destination module."""
    payload = _load_payload(path)
    adapter.model.load_state_dict(payload["model_state"], strict=True)
    heads.load_state_dict(payload["heads_state"], strict=True)
    adapter.to(device)
    heads.to(device)
    return dict(payload["metadata"])


def validate_checkpoint_file(
    path: str | Path, *, arm: str | None = None, optimizer_seed: int | None = None
) -> Mapping[str, object]:
    payload = _load_payload(path)
    metadata = payload["metadata"]
    if arm is not None and metadata["arm"] != arm:
        raise ValueError("checkpoint arm does not match matrix key")
    if optimizer_seed is not None and metadata["optimizer_seed"] != optimizer_seed:
        raise ValueError("checkpoint optimizer seed does not match matrix key")
    return metadata


def _load_payload(path: str | Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("invalid TG1 terminal checkpoint") from error
    if not isinstance(payload, dict) or set(payload) != {
        "model_state",
        "heads_state",
        "metadata",
    }:
        raise ValueError("invalid TG1 terminal checkpoint payload")
    _validate_state(payload["model_state"])
    _validate_state(payload["heads_state"])
    _validate_metadata(payload["metadata"])
    return payload


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _validate_state(state: object) -> None:
    if not isinstance(state, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("checkpoint state must map names to tensors")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("non-finite checkpoint tensor")


def _validate_metadata(metadata: object) -> None:
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    required = {
        "schema_version",
        "arm",
        "optimizer_seed",
        "horizon",
        "k",
        "test_opened",
    }
    if set(metadata) != required:
        raise ValueError("invalid TG1 checkpoint metadata schema")
    expected = checkpoint_metadata(metadata["arm"], metadata["optimizer_seed"])
    if dict(metadata) != expected:
        raise ValueError("checkpoint metadata does not match TG1 protocol")
    _require_finite(metadata)


def digest_payload(items: Sequence[FileDigest | TrajectoryCheckpoint]) -> list[dict]:
    return [asdict(item) for item in items]


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _require_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite values are forbidden")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite(item)
