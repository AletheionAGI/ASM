"""Per-episode prediction persistence for the sealed ATTR P2 evaluation."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .dataset import HazardEpisode, collate_episodes, gather_step_representations
from .metrics import basic_risk_metrics


@dataclass(frozen=True)
class EpisodePrediction:
    """Aligned predictions and targets for one complete episode."""

    episode_id: str
    world_id: str
    horizons: tuple[int, ...]
    actions: tuple[str, ...]
    hazard_labels: tuple[tuple[int, ...], ...]
    hazard_probabilities: tuple[tuple[float, ...], ...]
    next_state_nll: tuple[float, ...]
    severity: tuple[float, ...]
    severity_predictions: tuple[float, ...]
    time_to_hazard: tuple[float, ...]
    time_to_hazard_predictions: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EpisodePrediction:
        try:
            record = cls(
                episode_id=str(value["episode_id"]),
                world_id=str(value["world_id"]),
                horizons=tuple(int(item) for item in value["horizons"]),
                actions=tuple(str(item) for item in value["actions"]),
                hazard_labels=tuple(
                    tuple(int(x) for x in row) for row in value["hazard_labels"]
                ),
                hazard_probabilities=tuple(
                    tuple(float(x) for x in row)
                    for row in value["hazard_probabilities"]
                ),
                next_state_nll=tuple(float(x) for x in value["next_state_nll"]),
                severity=tuple(float(x) for x in value["severity"]),
                severity_predictions=tuple(
                    float(x) for x in value["severity_predictions"]
                ),
                time_to_hazard=tuple(float(x) for x in value["time_to_hazard"]),
                time_to_hazard_predictions=tuple(
                    float(x) for x in value["time_to_hazard_predictions"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid episode prediction record") from exc
        validate_episode_prediction(record)
        return record


@dataclass(frozen=True)
class P2Evaluation:
    records: tuple[EpisodePrediction, ...]
    metrics: dict[str, Any]


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def validate_episode_prediction(record: EpisodePrediction) -> None:
    """Reject records that cannot be compared step-for-step."""
    if not record.episode_id or not record.world_id:
        raise ValueError("episode_id and world_id must be non-empty")
    if not record.horizons or any(h <= 0 for h in record.horizons):
        raise ValueError("horizons must be positive and non-empty")
    if len(set(record.horizons)) != len(record.horizons):
        raise ValueError("horizons must be unique")
    steps = len(record.actions)
    vectors = (
        record.hazard_labels,
        record.hazard_probabilities,
        record.next_state_nll,
        record.severity,
        record.severity_predictions,
        record.time_to_hazard,
        record.time_to_hazard_predictions,
    )
    if steps == 0 or any(len(values) != steps for values in vectors):
        raise ValueError("all prediction fields must align with actions")
    width = len(record.horizons)
    if any(
        len(row) != width for row in record.hazard_labels + record.hazard_probabilities
    ):
        raise ValueError("hazard rows must align with horizons")
    labels = [value for row in record.hazard_labels for value in row]
    probabilities = [value for row in record.hazard_probabilities for value in row]
    numeric = probabilities + list(record.next_state_nll) + list(record.severity)
    numeric += list(record.severity_predictions) + list(record.time_to_hazard)
    numeric += list(record.time_to_hazard_predictions)
    if any(value not in (0, 1) for value in labels):
        raise ValueError("hazard labels must be binary")
    if not _finite(numeric):
        raise ValueError("prediction records must contain only finite numbers")
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("hazard probabilities must be in [0, 1]")


def _record(
    episode: HazardEpisode, horizons: tuple[int, ...], predictions
) -> EpisodePrediction:
    count = episode.step_positions.numel()
    state = predictions["next_state"]
    mean, log_scale = state["mean"][0, :count], state["log_scale"][0, :count]
    target = episode.next_states.to(mean.device)
    nll = (
        0.5 * math.log(2.0 * math.pi)
        + log_scale
        + 0.5 * ((target - mean) / log_scale.exp()).square()
    ).sum(-1)
    hazard = predictions["hazard_logits"][0, :count].sigmoid()
    severity = predictions["severity"]
    record = EpisodePrediction(
        episode.episode_id,
        episode.world_id,
        horizons,
        episode.actions,
        tuple(tuple(int(x) for x in row) for row in episode.hazard_labels.tolist()),
        tuple(tuple(float(x) for x in row) for row in hazard.cpu().tolist()),
        tuple(float(x) for x in nll.cpu().tolist()),
        tuple(float(x) for x in episode.severity.tolist()),
        tuple(float(x) for x in severity["severity"][0, :count].cpu().tolist()),
        tuple(float(x) for x in episode.time_to_hazard.tolist()),
        tuple(float(x) for x in severity["time_to_hazard"][0, :count].cpu().tolist()),
    )
    validate_episode_prediction(record)
    return record


def evaluate_episodes(
    adapter,
    heads,
    episodes: Sequence[HazardEpisode],
    *,
    device: str | torch.device = "cpu",
) -> P2Evaluation:
    """Evaluate common heads without gradients and retain episode boundaries."""
    if not episodes:
        raise ValueError("episodes must be non-empty")
    horizons = tuple(int(value) for value in heads.hazard.horizons)
    target = torch.device(device)
    adapter.to(target).eval()
    heads.to(target).eval()
    records = []
    with torch.no_grad():
        for episode in episodes:
            if episode.hazard_labels.ndim != 2 or episode.hazard_labels.shape[1] != len(
                horizons
            ):
                raise ValueError(
                    "episode hazard labels do not align with head horizons"
                )
            batch = {
                key: value.to(target)
                for key, value in collate_episodes([episode]).items()
            }
            representations = adapter(batch["input_ids"])
            steps = gather_step_representations(
                representations, batch["step_positions"]
            )
            records.append(_record(episode, horizons, heads(steps)))
    return P2Evaluation(tuple(records), compute_aggregate_metrics(records))


def compute_aggregate_metrics(records: Sequence[EpisodePrediction]) -> dict[str, Any]:
    """Compute step-weighted P2 metrics while accepting only aligned records."""
    if not records:
        raise ValueError("records must be non-empty")
    for record in records:
        validate_episode_prediction(record)
    horizons = records[0].horizons
    if any(record.horizons != horizons for record in records):
        raise ValueError("all records must use the same horizons")
    hazard = {}
    for column, horizon in enumerate(horizons):
        labels = [row[column] for record in records for row in record.hazard_labels]
        scores = [
            row[column] for record in records for row in record.hazard_probabilities
        ]
        metric = basic_risk_metrics(labels, scores)
        hazard[str(horizon)] = {"auprc": metric.auprc, "brier": metric.brier}

    def mean_error(prediction: str, truth: str) -> float:
        values = [
            abs(p - y)
            for record in records
            for p, y in zip(getattr(record, prediction), getattr(record, truth))
        ]
        return math.fsum(values) / len(values)

    nll = [value for record in records for value in record.next_state_nll]
    return {
        "episodes": len(records),
        "steps": len(nll),
        "hazard": hazard,
        "mean_next_state_nll": math.fsum(nll) / len(nll),
        "severity_mae": mean_error("severity_predictions", "severity"),
        "time_to_hazard_mae": mean_error(
            "time_to_hazard_predictions", "time_to_hazard"
        ),
    }


def write_episode_records_jsonl(
    path: str | Path, records: Sequence[EpisodePrediction]
) -> Path:
    """Atomically replace ``path`` with strict one-record-per-line JSONL."""
    if not records:
        raise ValueError("records must be non-empty")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for record in records:
                validate_episode_prediction(record)
                stream.write(
                    json.dumps(record.to_dict(), allow_nan=False, separators=(",", ":"))
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def read_episode_records_jsonl(path: str | Path) -> tuple[EpisodePrediction, ...]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(  # noqa: TRY004
                    f"JSONL line {line_number} is not an object"
                )
            records.append(EpisodePrediction.from_dict(value))
    if not records:
        raise ValueError("JSONL file contains no records")
    return tuple(records)


write_jsonl = write_episode_records_jsonl
read_jsonl = read_episode_records_jsonl

__all__ = [
    "EpisodePrediction",
    "P2Evaluation",
    "compute_aggregate_metrics",
    "evaluate_episodes",
    "read_episode_records_jsonl",
    "read_jsonl",
    "validate_episode_prediction",
    "write_episode_records_jsonl",
    "write_jsonl",
]
