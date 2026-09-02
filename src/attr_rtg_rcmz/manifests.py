"""Deterministic split, batch, and six-candidate manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .constants import (
    BACKBONE_UPDATES,
    CANDIDATES,
    PROTOCOL_ID,
    PROTOCOL_STATUS,
    SPLIT_ROWS,
    SYNTHETIC_NOTICE,
    TRAINING_SEEDS,
)
from .data_contracts import OriginKey

# The generated trust anchor is never candidate-manifest content.
CANDIDATE_MANIFEST_EXCLUDED_PATHS = frozenset({"src/attr_rtg_rcmz/lock_anchor.py"})


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class Manifest:
    kind: str
    payload: dict[str, object]
    sha256: str

    def __post_init__(self) -> None:
        if self.sha256 != payload_sha256(self.payload):
            raise ValueError("manifest digest differs from canonical payload")

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "payload": self.payload, "sha256": self.sha256}


def _manifest(kind: str, body: dict[str, object]) -> Manifest:
    payload = {
        "protocol": PROTOCOL_ID,
        "status": PROTOCOL_STATUS,
        "notice": SYNTHETIC_NOTICE,
        **body,
    }
    return Manifest(kind, payload, payload_sha256(payload))


def split_manifest() -> Manifest:
    """Describe the frozen allocation without generating any world."""
    rows = [
        {
            "name": name,
            "worlds": worlds,
            "episodes_per_world": 4,
            "max_episode_length": 64,
            "regime": regime,
        }
        for name, worlds, regime in SPLIT_ROWS
    ]
    return _manifest("split", {"splits": rows})


def _origin_dict(key: OriginKey) -> dict[str, object]:
    return asdict(key)


def _permutation(keys: tuple[OriginKey, ...], training_seed: int) -> tuple[int, ...]:
    def rank(index: int) -> bytes:
        material = canonical_json(
            {
                "purpose": "batch",
                "seed": training_seed,
                "origin": _origin_dict(keys[index]),
            }
        )
        return hashlib.sha256(material).digest()

    return tuple(sorted(range(len(keys)), key=lambda index: (rank(index), index)))


def batch_manifest(
    origins: Sequence[OriginKey],
    training_seed: int,
    *,
    batch_size: int,
    updates: int = BACKBONE_UPDATES,
) -> Manifest:
    """Hash-permute once, then cycle in fixed batches without reshuffling."""
    keys = tuple(sorted(origins))
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("unregistered training seed")
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("origins must be non-empty and unique")
    if (
        type(batch_size) is not int
        or batch_size < 1
        or type(updates) is not int
        or updates < 1
    ):
        raise ValueError("batch_size and updates must be positive integers")
    order = _permutation(keys, training_seed)
    batches = [
        [order[(offset + slot) % len(order)] for slot in range(batch_size)]
        for offset in range(0, updates * batch_size, batch_size)
    ]
    return _manifest(
        "batch",
        {
            "training_seed": training_seed,
            "population": [_origin_dict(key) for key in keys],
            "permutation": list(order),
            "batch_size": batch_size,
            "updates": updates,
            "batches": batches,
        },
    )


def candidate_manifest(origins: Sequence[OriginKey]) -> Manifest:
    """Enumerate every origin with the fixed external six-candidate axis."""
    keys = tuple(origins)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("origins must be non-empty and unique")
    rows = [
        {"origin": _origin_dict(key), "candidates": list(CANDIDATES)}
        for key in sorted(keys)
    ]
    return _manifest("candidate", {"rows": rows})
