"""Trajectory-grounded ATTR-TG1 evaluation and strict JSONL persistence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HORIZONS = (1, 4, 8)


@dataclass(frozen=True)
class TrajectoryIdentity:
    seed: int
    arm: str
    split: str
    world_id: str
    episode_id: str
    anchor: int


@dataclass(frozen=True)
class PhysicalTrajectorySample:
    """One sampled physical rollout; no classifier probabilities are accepted."""

    unsafe_by_step: tuple[bool, ...]
    nll_by_step: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class TrajectoryRecord:
    identity: TrajectoryIdentity
    risk_by_horizon: Mapping[int, float]
    unsafe_truth: tuple[bool, ...]
    trajectory_nll: Mapping[int, Mapping[str, float]]
    physical_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["risk_by_horizon"] = {str(k): v for k, v in self.risk_by_horizon.items()}
        value["trajectory_nll"] = {
            str(k): dict(v) for k, v in self.trajectory_nll.items()
        }
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrajectoryRecord:
        expected = {
            "identity",
            "risk_by_horizon",
            "unsafe_truth",
            "trajectory_nll",
            "physical_sample_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("trajectory record has missing or unexpected fields")
        identity = value["identity"]
        identity_fields = {"seed", "arm", "split", "world_id", "episode_id", "anchor"}
        if not isinstance(identity, Mapping) or set(identity) != identity_fields:
            raise ValueError("invalid trajectory identity fields")
        risks, nll, truth = (
            value["risk_by_horizon"],
            value["trajectory_nll"],
            value["unsafe_truth"],
        )
        if not isinstance(risks, Mapping) or set(risks) != {"1", "4", "8"}:
            raise ValueError("invalid serialized risk horizons")
        if not isinstance(nll, Mapping) or set(nll) != {"1", "4", "8"}:
            raise ValueError("invalid serialized NLL horizons")
        if not isinstance(truth, Sequence) or isinstance(truth, (str, bytes)):
            raise TypeError("unsafe_truth must be a sequence")
        numbers = list(risks.values()) + [
            item
            for row in nll.values()
            if isinstance(row, Mapping)
            for item in row.values()
        ]
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in numbers
        ):
            raise ValueError("risk and NLL JSON values must be numbers")
        if any(
            not isinstance(row, Mapping)
            or any(not isinstance(field, str) for field in row)
            for row in nll.values()
        ):
            raise ValueError("invalid serialized NLL fields")
        try:
            record = cls(
                TrajectoryIdentity(**identity),
                {int(k): float(v) for k, v in risks.items()},
                tuple(truth),
                {
                    int(k): {field: float(number) for field, number in row.items()}
                    for k, row in nll.items()
                },
                value["physical_sample_count"],
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid trajectory record") from exc
        validate_record(record)
        return record


def _valid_identity(identity: TrajectoryIdentity) -> bool:
    return (
        type(identity.seed) is int
        and type(identity.anchor) is int
        and identity.anchor >= 0
        and all(
            isinstance(v, str) and bool(v)
            for v in (
                identity.arm,
                identity.split,
                identity.world_id,
                identity.episode_id,
            )
        )
    )


def validate_record(record: TrajectoryRecord) -> None:
    """Apply the sealed TG1 schema, including exact registered horizons."""
    if not isinstance(record.identity, TrajectoryIdentity) or not _valid_identity(
        record.identity
    ):
        raise ValueError("identity requires integer seed/anchor and non-empty names")
    if tuple(sorted(record.risk_by_horizon)) != HORIZONS:
        raise ValueError("risk_by_horizon must contain exactly H1/H4/H8")
    if any(
        type(v) is not float or not math.isfinite(v) or not 0 <= v <= 1
        for v in record.risk_by_horizon.values()
    ):
        raise ValueError("risks must be finite float probabilities")
    risks = [record.risk_by_horizon[horizon] for horizon in HORIZONS]
    if risks != sorted(risks):
        raise ValueError("prefix-event risk must be non-decreasing with horizon")
    if len(record.unsafe_truth) < max(HORIZONS) or any(
        type(v) is not bool for v in record.unsafe_truth
    ):
        raise ValueError(
            "unsafe_truth must contain at least eight boolean physical steps"
        )
    if tuple(sorted(record.trajectory_nll)) != HORIZONS:
        raise ValueError("trajectory_nll must contain exactly H1/H4/H8")
    fields = None
    for row in record.trajectory_nll.values():
        if not isinstance(row, Mapping) or not row:
            raise ValueError("each NLL horizon needs at least one field")
        if any(
            not isinstance(k, str)
            or not k
            or type(v) is not float
            or not math.isfinite(v)
            for k, v in row.items()
        ):
            raise ValueError("NLL fields require finite floats")
        fields = set(row) if fields is None else fields
        if set(row) != fields:
            raise ValueError("NLL fields must agree across horizons")
    if (
        type(record.physical_sample_count) is not int
        or record.physical_sample_count < 1
    ):
        raise ValueError("physical_sample_count must be a positive integer")


def _validate_sample(sample: PhysicalTrajectorySample) -> None:
    if not isinstance(sample, PhysicalTrajectorySample):
        raise TypeError("sampler must return PhysicalTrajectorySample values")
    if len(sample.unsafe_by_step) < 8 or len(sample.nll_by_step) < 8:
        raise ValueError("physical samples must cover eight steps")
    if any(type(v) is not bool for v in sample.unsafe_by_step):
        raise ValueError("physical unsafe flags must be boolean")
    fields = set(sample.nll_by_step[0]) if sample.nll_by_step else set()
    if not fields or any(set(row) != fields for row in sample.nll_by_step):
        raise ValueError("physical NLL fields must be non-empty and step-aligned")
    if any(
        not isinstance(k, str)
        or not k
        or isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(float(v))
        for row in sample.nll_by_step
        for k, v in row.items()
    ):
        raise ValueError("physical NLL values must be finite numbers")


def evaluate_physical_trajectories(
    identity: TrajectoryIdentity,
    unsafe_truth: Sequence[bool],
    sampler: Callable[[TrajectoryIdentity], Iterable[PhysicalTrajectorySample]],
) -> TrajectoryRecord:
    """Evaluate an origin through an explicit physical-rollout sampler contract.

    ``sampler`` must run the transition model/environment and return physical
    state predicates and field NLLs. Logits, HazardHead, and supplied risk scores
    are intentionally absent from this interface.
    """
    samples = tuple(sampler(identity))
    if not samples:
        raise ValueError("sampler returned no physical trajectories")
    for sample in samples:
        _validate_sample(sample)
    risks = {
        h: float(sum(any(s.unsafe_by_step[:h]) for s in samples) / len(samples))
        for h in HORIZONS
    }
    fields = tuple(samples[0].nll_by_step[0])
    nll = {
        h: {
            field: float(
                sum(
                    sum(float(s.nll_by_step[i][field]) for i in range(h))
                    for s in samples
                )
                / (len(samples) * h)
            )
            for field in fields
        }
        for h in HORIZONS
    }
    record = TrajectoryRecord(identity, risks, tuple(unsafe_truth), nll, len(samples))
    validate_record(record)
    return record


def write_records_jsonl(path: str | Path, records: Iterable[TrajectoryRecord]) -> Path:
    """Atomically replace a JSONL file after validating the complete payload."""
    target = Path(path)
    values = tuple(records)
    if not values:
        raise ValueError("records must not be empty")
    seen = set()
    for record in values:
        validate_record(record)
        key = asdict(record.identity)
        marker = tuple(key.values())
        if marker in seen:
            raise ValueError("duplicate trajectory identity")
        seen.add(marker)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent, text=True
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for record in values:
                stream.write(
                    json.dumps(
                        record.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def read_records_jsonl(path: str | Path) -> tuple[TrajectoryRecord, ...]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
                records.append(TrajectoryRecord.from_dict(value))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid trajectory JSONL line {number}") from exc
    if not records:
        raise ValueError("trajectory JSONL must not be empty")
    identities = [tuple(asdict(record.identity).values()) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate trajectory identity in JSONL")
    return tuple(records)
