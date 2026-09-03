"""Fail-closed validation and audit records for checkpoint recovery."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import ARMS, BACKBONE_UPDATES, TRAINING_SEEDS


@dataclass(frozen=True)
class RecoveredCheckpoint:
    arm: str
    seed: int
    path: Path
    sha256: str


def training_arms(
    seed: int, recovered: dict[int, dict[str, RecoveredCheckpoint]]
) -> tuple[str, ...]:
    """Return no arms for a completely recovered seed; never permit partial skips."""
    if seed not in recovered:
        return ARMS
    if tuple(arm for arm in ARMS if arm in recovered[seed]) != ARMS:
        raise ValueError(f"incomplete recovered seed group: {seed}")
    return ()


def validate_recovery_manifest(path: Path) -> dict[int, dict[str, RecoveredCheckpoint]]:
    """Validate every declared checkpoint before any official work starts."""
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "checkpoints",
    }:
        raise ValueError("recovery manifest fields differ")
    if document["schema_version"] != 1 or not isinstance(document["checkpoints"], list):
        raise ValueError("unsupported recovery manifest schema")
    recovered: dict[int, dict[str, RecoveredCheckpoint]] = {}
    for entry in document["checkpoints"]:
        item = _validate_entry(path.parent, entry)
        arms = recovered.setdefault(item.seed, {})
        if item.arm in arms:
            raise ValueError(
                f"duplicate recovered checkpoint: seed {item.seed} {item.arm}"
            )
        arms[item.arm] = item
    if not recovered:
        raise ValueError("recovery manifest contains no checkpoints")
    for seed, arms in recovered.items():
        if tuple(arm for arm in ARMS if arm in arms) != ARMS or len(arms) != len(ARMS):
            raise ValueError(
                f"recovery requires the complete R,CM,Z,T group for seed {seed}"
            )
    return recovered


def _validate_entry(base: Path, entry: Any) -> RecoveredCheckpoint:
    expected = {"arm", "seed", "update", "path", "sha256"}
    if not isinstance(entry, dict) or set(entry) != expected:
        raise ValueError("recovery checkpoint entry fields differ")
    arm, seed, update = entry["arm"], entry["seed"], entry["update"]
    digest = entry["sha256"]
    if arm not in ARMS or seed not in TRAINING_SEEDS or update != BACKBONE_UPDATES:
        raise ValueError(
            "recovery checkpoint identity is not registered terminal state"
        )
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("recovery checkpoint SHA-256 is malformed")
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "recovery checkpoint path must remain below manifest directory"
        )
    checkpoint = (base / relative).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"recovery checkpoint absent: {checkpoint}")
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"recovery checkpoint digest differs: seed {seed} {arm}")
    _validate_payload(checkpoint, arm, seed, update)
    return RecoveredCheckpoint(arm, seed, checkpoint, actual)


def _validate_payload(path: Path, arm: str, seed: int, update: int) -> None:
    import torch

    from .models import build_adapter
    from .official_training import _authorized_config

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {
        "update",
        "config",
        "model",
        "optimizer",
    }:
        raise ValueError(f"recovery checkpoint payload fields differ: {path}")
    config = _authorized_config(arm, seed)
    if payload["update"] != update or payload["config"] != config.__dict__:
        raise ValueError(
            f"recovery checkpoint update/config differs: seed {seed} {arm}"
        )
    expected = build_adapter(config).state_dict()
    state = payload["model"]
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError(f"recovery checkpoint model keys differ: seed {seed} {arm}")
    for name, tensor in expected.items():
        restored = state[name]
        if (
            not isinstance(restored, torch.Tensor)
            or restored.shape != tensor.shape
            or restored.dtype != tensor.dtype
        ):
            raise ValueError(
                f"recovery checkpoint tensor shape/dtype differs: {arm}.{name}"
            )
    if not isinstance(payload["optimizer"], dict):
        raise TypeError(
            f"recovery checkpoint optimizer state differs: seed {seed} {arm}"
        )


def write_recovery_record(
    output_dir: Path, source: Path, recovered: dict[int, dict[str, RecoveredCheckpoint]]
) -> Path:
    """Write an exclusive audit record after all inputs have validated."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "recovery_manifest.json"
    if target.exists():
        raise FileExistsError(f"recovery record already exists: {target}")
    rows = [
        {
            "seed": seed,
            "arm": arm,
            "update": BACKBONE_UPDATES,
            "path": str(items[arm].path),
            "sha256": items[arm].sha256,
            "action": "reuse",
        }
        for seed, items in recovered.items()
        for arm in ARMS
    ]
    pending = [
        {"seed": seed, "arms": list(ARMS), "action": "train"}
        for seed in TRAINING_SEEDS
        if seed not in recovered
    ]
    document = {
        "schema_version": 1,
        "status": "VALIDATED",
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reused": rows,
        "pending": pending,
    }
    temporary = target.with_suffix(".json.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def archive_previous_run(
    output_dir: Path, extra_paths: tuple[Path, ...] = ()
) -> Path | None:
    """Move only mutable prior-run receipts aside; preserve all checkpoints."""
    candidates = [
        output_dir / name
        for name in (
            "TOMBSTONE.json",
            "status.json",
            "run.log",
            "recovery_manifest.json",
        )
    ]
    candidates.extend(extra_paths)
    unique = {str(path.resolve()): path for path in candidates}
    present = [path for path in unique.values() if path.exists()]
    if not present:
        return None
    archive_root = output_dir / "recovery_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (archive_root / f"attempt-{index:04d}").exists():
        index += 1
    destination = archive_root / f"attempt-{index:04d}"
    destination.mkdir()
    for source in present:
        target = destination / source.name
        if target.exists():
            raise FileExistsError(f"recovery archive collision: {target}")
        os.replace(source, target)
    return destination
