"""Exact state-transition and physical-fidelity metrics for frozen ATTR-RTG."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import fsum, isfinite, log
from typing import Any

Record = Mapping[str, Any]
_GROUPS = 11
_BINS = 15


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _identity(row: Record) -> tuple[int, str, str, str, int]:
    try:
        seed = row["seed"]
        world = str(row["world_id"])
        episode = str(row["episode_id"])
        origin = str(row["origin_id"] if "origin_id" in row else row["t"])
        action = row["action_index"]
    except (KeyError, TypeError) as error:
        raise ValueError("record is missing its hierarchical identity") from error
    if type(seed) is not int or type(action) is not int or action not in range(6):
        raise ValueError("seed must be int and action_index must be in range(6)")
    return seed, world, episode, origin, action


def validate_six_candidate_clusters(records: Iterable[Record]) -> tuple[Record, ...]:
    """Fail closed unless every origin has exactly the six registered candidates."""
    rows = tuple(records)
    if not rows:
        raise ValueError("records must not be empty")
    clusters: dict[tuple[int, str, str, str], set[int]] = defaultdict(set)
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each record must be a mapping")
        identity = _identity(row)
        if identity in seen:
            raise ValueError("candidate identities must be unique")
        seen.add(identity)
        clusters[identity[:-1]].add(identity[-1])
    if any(actions != set(range(6)) for actions in clusters.values()):
        raise ValueError("every origin must preserve exactly six candidates")
    return rows


def hierarchical_mean(records: Iterable[Record], field: str) -> float:
    """Average candidate→origin→episode→world→seed with equal weights."""
    rows = validate_six_candidate_clusters(records)
    level: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        identity = _identity(row)
        level[identity[:-1]].append(_number(row[field], field))
    values: dict[tuple[Any, ...], float] = {
        key: fsum(items) / len(items) for key, items in level.items()
    }
    for length in (3, 2, 1):
        grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for key, value in values.items():
            grouped[key[:length]].append(value)
        values = {key: fsum(items) / len(items) for key, items in grouped.items()}
    return fsum(values.values()) / len(values)


def _candidate_mse(row: Record, predicted: str, target: str, dimension: int) -> float:
    try:
        left, right = tuple(row[predicted]), tuple(row[target])
    except (KeyError, TypeError) as error:
        raise ValueError(f"{predicted} and {target} must be sequences") from error
    if len(left) != dimension or len(right) != dimension:
        raise ValueError(f"state vectors must have dimension {dimension}")
    squares = [(_number(a, predicted) - _number(b, target)) ** 2 for a, b in zip(left, right)]
    return fsum(squares) / dimension


def transition_state_metrics(
    records: Iterable[Record], *, dimension: int = 28,
    prediction_field: str = "predicted_state", target_field: str = "true_state",
    persistence_field: str = "persistence_state",
) -> dict[str, float]:
    """Compute registered RTG1-Z MSEs and their ratio from candidate records."""
    if type(dimension) is not int or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    rows = validate_six_candidate_clusters(records)
    enriched = []
    for row in rows:
        item = dict(row)
        item["_model_mse"] = _candidate_mse(row, prediction_field, target_field, dimension)
        item["_persistence_mse"] = _candidate_mse(row, persistence_field, target_field, dimension)
        enriched.append(item)
    model = hierarchical_mean(enriched, "_model_mse")
    persistence = hierarchical_mean(enriched, "_persistence_mse")
    if persistence <= 0:
        raise ValueError("state persistence MSE must be finite and positive")
    return {"mse_g": model, "mse_state_persistence": persistence,
            "transition_nmse": model / persistence}


def consequence_macro_accuracy(
    records: Iterable[Record], *, prediction_field: str = "group_predictions",
    target_field: str = "group_targets",
) -> float:
    """Average the 11 group top-1 accuracies, then use the registered hierarchy."""
    rows = validate_six_candidate_clusters(records)
    enriched = []
    for row in rows:
        try:
            predictions = tuple(row[prediction_field])
            targets = tuple(row[target_field])
        except (KeyError, TypeError) as error:
            raise ValueError("group predictions and targets must be sequences") from error
        if len(predictions) != _GROUPS or len(targets) != _GROUPS:
            raise ValueError("physical consequence requires exactly 11 groups")
        if any(type(value) is not int for value in predictions + targets):
            raise ValueError("group predictions and targets must be integer categories")
        item = dict(row)
        item["_macro_accuracy"] = fsum(
            prediction == target for prediction, target in zip(predictions, targets)
        ) / _GROUPS
        enriched.append(item)
    return hierarchical_mean(enriched, "_macro_accuracy")


def consequence_nll(records: Iterable[Record], field: str = "group_nll") -> float:
    """Average 11 group NLLs per candidate, then use the registered hierarchy."""
    rows = validate_six_candidate_clusters(records)
    enriched = []
    for row in rows:
        try:
            groups = tuple(row[field])
        except (KeyError, TypeError) as error:
            raise ValueError(f"{field} must be a sequence") from error
        if len(groups) != _GROUPS:
            raise ValueError("physical consequence requires exactly 11 groups")
        item = dict(row)
        item["_nll"] = fsum(_number(value, field) for value in groups) / _GROUPS
        enriched.append(item)
    return hierarchical_mean(enriched, "_nll")


def expected_calibration_error(
    records: Iterable[Record], *, score_field: str = "risk", label_field: str = "unsafe",
) -> float:
    """Compute registered 15-bin ECE; the final bin includes probability one."""
    rows = validate_six_candidate_clusters(records)
    bins: list[list[tuple[float, int]]] = [[] for _ in range(_BINS)]
    for row in rows:
        score = _number(row[score_field], score_field)
        label = row[label_field]
        if not 0 <= score <= 1 or type(label) not in (bool, int) or label not in (0, 1):
            raise ValueError("risk must be in [0,1] and unsafe must be binary")
        bins[min(_BINS - 1, int(score * _BINS))].append((score, int(label)))
    total = len(rows)
    return fsum(
        len(items) / total * abs(fsum(p for p, _ in items) / len(items)
                                 - fsum(y for _, y in items) / len(items))
        for items in bins if items
    )


def categorical_nll(probabilities: Sequence[float], target: int) -> float:
    """Return fail-closed categorical NLL for one registered physical group."""
    if type(target) is not int or target not in range(len(probabilities)):
        raise ValueError("categorical target is outside its group")
    probability = _number(probabilities[target], "target probability")
    if probability <= 0 or any(_number(p, "probability") < 0 for p in probabilities):
        raise ValueError("probabilities must be finite and target probability positive")
    if abs(fsum(float(p) for p in probabilities) - 1.0) > 1e-9:
        raise ValueError("categorical probabilities must sum to one")
    return -log(probability)
