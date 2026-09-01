"""Fail-closed dataset seal for the five-seed ATTR P2 benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

SEAL_SCHEMA_VERSION = 1
TRAINING_SEEDS = (29, 43, 71, 89, 107)
MODEL_ARMS = (
    "asm_x_directional",
    "tiny_transformer_220k",
    "asm_cm_durable",
    "asm_vr_s_full64",
    "asm_vr_s_fixed32",
    "asm_r_240k_control",
)
REQUIRED_CHECKPOINTS = len(TRAINING_SEEDS) * len(MODEL_ARMS)
_SHA256_LENGTH = 64
CheckpointPaths = Mapping[tuple[str, int], str | Path]


@dataclass(frozen=True)
class P2SplitSpec:
    """Generator inputs only; constructing a spec cannot expose test labels."""

    test_id: str
    seed: int
    world_count: int
    episodes_per_world: int
    dynamic_family: str
    max_steps: int = 16
    sensor_noise: float = 0.04
    forcing: float = 0.10
    failure_delay: int = 3
    recovery_window: int = 3

    def __post_init__(self) -> None:
        if not self.test_id.startswith("test_"):
            raise ValueError("test_id must name a sealed test split")
        if self.world_count < 1 or self.episodes_per_world < 1 or self.max_steps < 2:
            raise ValueError("invalid P2 split size")
        if self.failure_delay < 1 or self.recovery_window < 1:
            raise ValueError("invalid P2 failure timing")


@dataclass(frozen=True)
class CheckpointDigest:
    arm: str
    training_seed: int
    sha256: str

    def __post_init__(self) -> None:
        if self.arm not in MODEL_ARMS or self.training_seed not in TRAINING_SEEDS:
            raise ValueError("unregistered P2 checkpoint")
        if len(self.sha256) != _SHA256_LENGTH or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ValueError("checkpoint hash must be lowercase SHA256")


@dataclass(frozen=True)
class P2DatasetSeal:
    schema_version: int
    training_seeds: tuple[int, ...]
    model_arms: tuple[str, ...]
    splits: tuple[P2SplitSpec, ...]
    checkpoints: tuple[CheckpointDigest, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SEAL_SCHEMA_VERSION:
            raise ValueError("unsupported P2 seal schema")
        if self.training_seeds != TRAINING_SEEDS or self.model_arms != MODEL_ARMS:
            raise ValueError("P2 training configuration is not registered")
        if tuple(spec.test_id for spec in self.splits) != (
            "test_id", "test_shift", "test_ood"
        ):
            raise ValueError("P2 seal requires ordered ID, shift, and OOD specs")
        _validate_checkpoint_records(self.checkpoints)

    @property
    def sha256(self) -> str:
        return canonical_sha256(_seal_payload(self))


def default_p2_specs() -> tuple[P2SplitSpec, ...]:
    """Return deterministic specs; do not generate worlds, episodes, or labels."""
    return (
        P2SplitSpec("test_id", 170_001, 32, 4, "baseline"),
        P2SplitSpec(
            "test_shift", 170_002, 32, 4, "shift",
            sensor_noise=0.12, forcing=0.22,
        ),
        P2SplitSpec(
            "test_ood", 170_003, 32, 4, "ood",
            sensor_noise=0.18, forcing=0.16, failure_delay=1,
            recovery_window=2,
        ),
    )


def canonical_json(value: object) -> bytes:
    """Encode JSON with the one canonical representation used by the seal."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_p2_seal(
    checkpoint_paths: CheckpointPaths,
    *,
    splits: Sequence[P2SplitSpec] | None = None,
) -> P2DatasetSeal:
    """Hash the exact six-arm/five-seed validation-selected checkpoint matrix."""
    return P2DatasetSeal(
        SEAL_SCHEMA_VERSION,
        TRAINING_SEEDS,
        MODEL_ARMS,
        tuple(splits or default_p2_specs()),
        _checkpoint_records(checkpoint_paths),
    )


def write_p2_seal(seal: P2DatasetSeal, path: str | Path) -> Path:
    """Persist a canonical, self-authenticating seal without opening test data."""
    path = Path(path)
    payload = _seal_payload(seal)
    document = {"payload": payload, "sha256": canonical_sha256(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(document) + b"\n")
    return path


def read_p2_seal(path: str | Path) -> P2DatasetSeal:
    document = json.loads(Path(path).read_bytes())
    if set(document) != {"payload", "sha256"}:
        raise ValueError("invalid P2 seal document")
    if canonical_sha256(document["payload"]) != document["sha256"]:
        raise ValueError("P2 seal SHA256 mismatch")
    return _seal_from_payload(document["payload"])


def open_p2_seal(
    seal: P2DatasetSeal,
    checkpoint_paths: CheckpointPaths,
) -> tuple[P2SplitSpec, ...]:
    """Return specs only after all 30 files exist and their SHA256 hashes match."""
    actual = _checkpoint_records(checkpoint_paths)
    if actual != seal.checkpoints:
        raise ValueError("checkpoint list or SHA256 does not match the P2 seal")
    return seal.splits


def _checkpoint_records(paths: CheckpointPaths) -> tuple[CheckpointDigest, ...]:
    expected = {(arm, seed) for arm in MODEL_ARMS for seed in TRAINING_SEEDS}
    if set(paths) != expected:
        raise ValueError("P2 requires the exact six-arm/five-seed checkpoint matrix")
    if any(not Path(path).is_file() for path in paths.values()):
        raise ValueError("all 30 P2 checkpoints must exist")
    return tuple(
        CheckpointDigest(arm, seed, file_sha256(paths[(arm, seed)]))
        for arm in MODEL_ARMS
        for seed in TRAINING_SEEDS
    )


def _validate_checkpoint_records(records: Sequence[CheckpointDigest]) -> None:
    expected = [(arm, seed) for arm in MODEL_ARMS for seed in TRAINING_SEEDS]
    actual = [(record.arm, record.training_seed) for record in records]
    if actual != expected or len(records) != REQUIRED_CHECKPOINTS:
        raise ValueError("P2 checkpoint hashes are incomplete or out of order")


def _seal_payload(seal: P2DatasetSeal) -> dict:
    return {
        "schema_version": seal.schema_version,
        "training_seeds": list(seal.training_seeds),
        "model_arms": list(seal.model_arms),
        "splits": [asdict(spec) for spec in seal.splits],
        "checkpoints": [asdict(record) for record in seal.checkpoints],
    }


def _seal_from_payload(payload: Mapping[str, object]) -> P2DatasetSeal:
    required = {
        "schema_version", "training_seeds", "model_arms", "splits", "checkpoints"
    }
    if set(payload) != required:
        raise ValueError("invalid P2 seal payload")
    seal = P2DatasetSeal(
        int(payload["schema_version"]),
        tuple(payload["training_seeds"]),
        tuple(payload["model_arms"]),
        tuple(P2SplitSpec(**item) for item in payload["splits"]),
        tuple(CheckpointDigest(**item) for item in payload["checkpoints"]),
    )
    return seal


write_seal = write_p2_seal
read_seal = read_p2_seal
open_seal = open_p2_seal

__all__ = [
    "CheckpointDigest", "MODEL_ARMS", "P2DatasetSeal", "P2SplitSpec",
    "REQUIRED_CHECKPOINTS", "TRAINING_SEEDS", "canonical_json",
    "canonical_sha256", "create_p2_seal", "default_p2_specs", "file_sha256",
    "open_p2_seal", "open_seal", "read_p2_seal", "read_seal",
    "write_p2_seal", "write_seal",
]
