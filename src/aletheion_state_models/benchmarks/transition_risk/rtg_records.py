"""Strict generic candidate records and canonical JSONL for ATTR-RTG."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class CandidateIdentity:
    training_seed: int
    split_id: int
    world_id: str
    episode_id: str
    t: int
    action_index: int

    def __post_init__(self) -> None:
        if type(self.training_seed) is not int or type(self.split_id) is not int:
            raise ValueError("seed and split must be integers")
        if not self.world_id or not self.episode_id or self.t < 1:
            raise ValueError("invalid origin identity")
        if self.action_index not in range(6):
            raise ValueError("action_index must be in 0..5")

    @property
    def origin(self) -> tuple[int, int, str, str, int]:
        return (
            self.training_seed,
            self.split_id,
            self.world_id,
            self.episode_id,
            self.t,
        )


@dataclass(frozen=True)
class CandidateRecord:
    identity: CandidateIdentity
    values: Mapping[str, float | int | str | bool]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("record values must not be empty")
        for value in self.values.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("record values must be finite")
            if not isinstance(value, (float, int, str, bool)):
                raise TypeError("record values must be JSON scalars")


def validate_candidate_records(
    records: Iterable[CandidateRecord], *, require_six: bool = True
) -> tuple[CandidateRecord, ...]:
    ordered = tuple(sorted(records, key=lambda item: item.identity))
    if not ordered or len({item.identity for item in ordered}) != len(ordered):
        raise ValueError("records must be nonempty with unique identities")
    if require_six:
        origins: dict[tuple, set[int]] = {}
        for record in ordered:
            origins.setdefault(record.identity.origin, set()).add(
                record.identity.action_index
            )
        if any(actions != set(range(6)) for actions in origins.values()):
            raise ValueError("every origin must contain exactly six candidates")
    return ordered


def write_records_jsonl(
    path: str | Path, records: Iterable[CandidateRecord]
) -> Path:
    destination = Path(path)
    rows = validate_candidate_records(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(asdict(row), sort_keys=True, allow_nan=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_records_jsonl(path: str | Path) -> tuple[CandidateRecord, ...]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if set(raw) != {"identity", "values"}:
            raise ValueError("unexpected record fields")
        rows.append(CandidateRecord(CandidateIdentity(**raw["identity"]), raw["values"]))
    return validate_candidate_records(rows)
