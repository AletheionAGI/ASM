"""P2 horizon metrics and paired seed/world/episode bootstrap statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import floor, isfinite
from random import Random
from typing import Any

from .metrics import brier_score

Record = Mapping[str, Any]
Key = tuple[int, str, str, str]
Sample = tuple[list[int], list[float]]


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision with tied scores treated as one threshold."""
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ordered = sorted(zip(scores, labels), reverse=True)
    true_positives = seen = index = 0
    previous_recall = area = 0.0
    while index < len(ordered):
        score = ordered[index][0]
        group_positives = group_size = 0
        while index < len(ordered) and ordered[index][0] == score:
            group_positives += ordered[index][1]
            group_size += 1
            index += 1
        true_positives += group_positives
        seen += group_size
        recall = true_positives / positives
        area += (recall - previous_recall) * (true_positives / seen)
        previous_recall = recall
    return area


def _as_sequence(value: Any, field: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return list(value)


def _normalise(records: Iterable[Record]) -> dict[Key, dict[int, Sample]]:
    result: dict[Key, dict[int, Sample]] = {}
    arms: set[str] = set()
    splits: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each record must be a mapping")
        required = ("seed", "arm", "split", "world_id", "episode_id")
        if any(name not in record for name in required):
            raise ValueError("record is missing an identity field")
        if isinstance(record["seed"], bool) or not isinstance(record["seed"], int):
            raise ValueError("seed must be an integer")
        if any(not isinstance(record[name], str) or not record[name] for name in required[1:]):
            raise ValueError("arm, split, world_id, and episode_id must be non-empty strings")
        arms.add(record["arm"])
        splits.add(record["split"])
        key = (record["seed"], record["split"], record["world_id"], record["episode_id"])
        if key in result:
            raise ValueError(f"duplicate seed/split/world/episode key: {key!r}")
        horizons = _as_sequence(record.get("horizons"), "horizons")
        labels = _as_sequence(record.get("hazard_labels"), "hazard_labels")
        scores = _as_sequence(record.get("hazard_probabilities"), "hazard_probabilities")
        if not horizons or len(horizons) != len(labels) or len(labels) != len(scores):
            raise ValueError("horizons, hazard_labels, and hazard_probabilities must align")
        by_horizon: dict[int, Sample] = {}
        for raw_horizon, raw_labels, raw_scores in zip(horizons, labels, scores):
            if isinstance(raw_horizon, bool) or not isinstance(raw_horizon, int) or raw_horizon < 1:
                raise ValueError("horizons must be positive integers")
            if raw_horizon in by_horizon:
                raise ValueError("horizons must be unique")
            label_values = _as_sequence(raw_labels, "hazard_labels item")
            if any(type(value) not in (int, bool) or value not in (0, 1) for value in label_values):
                raise ValueError("hazard labels must be binary integers")
            y = [int(value) for value in label_values]
            p = [float(value) for value in _as_sequence(raw_scores, "hazard_probabilities item")]
            if not y or len(y) != len(p):
                raise ValueError("labels and probabilities must be non-empty and equal length")
            if any(not isfinite(value) or not 0 <= value <= 1 for value in p):
                raise ValueError("hazard probabilities must be finite and in [0, 1]")
            by_horizon[raw_horizon] = (y, p)
        result[key] = by_horizon
    if not result:
        raise ValueError("records must not be empty")
    if len(arms) != 1 or len(splits) != 1:
        raise ValueError("records must describe exactly one arm and one split")
    horizon_sets = {tuple(sorted(row)) for row in result.values()}
    if len(horizon_sets) != 1:
        raise ValueError("all records must contain the same horizons")
    return result


def _metrics(samples: Iterable[Sample]) -> dict[str, float | int]:
    labels: list[int] = []
    scores: list[float] = []
    for sample_labels, sample_scores in samples:
        labels.extend(sample_labels)
        scores.extend(sample_scores)
    return {
        "auprc": _average_precision(labels, scores),
        "brier": brier_score(labels, scores),
        "n": len(labels),
        "positives": sum(labels),
    }


def aggregate_by_horizon(records: Iterable[Record]) -> dict[int, dict[str, float | int]]:
    """Pool serializable episode records and compute AUPRC and Brier by horizon."""
    data = _normalise(records)
    horizons = next(iter(data.values()))
    return {horizon: _metrics(row[horizon] for row in data.values()) for horizon in horizons}


def _paired(candidate: Iterable[Record], reference: Iterable[Record]):
    left, right = _normalise(candidate), _normalise(reference)
    if left.keys() != right.keys():
        raise ValueError("left and right records must have identical paired keys")
    for key in left:
        if left[key].keys() != right[key].keys():
            raise ValueError("paired records must have identical horizons")
        if any(left[key][h][0] != right[key][h][0] for h in left[key]):
            raise ValueError("paired records must have identical hazard labels")
    tree: dict[int, dict[tuple[str, str], list[Key]]] = defaultdict(lambda: defaultdict(list))
    for key in left:
        tree[key[0]][(key[1], key[2])].append(key)
    return left, right, tree


def _draw_keys(tree: Mapping[int, Mapping[Any, Sequence[Key]]], rng: Random) -> list[Key]:
    selected: list[Key] = []
    seeds = sorted(tree)
    for seed in rng.choices(seeds, k=len(seeds)):
        worlds = sorted(tree[seed])
        for world in rng.choices(worlds, k=len(worlds)):
            episodes = sorted(tree[seed][world])
            selected.extend(rng.choices(episodes, k=len(episodes)))
    return selected


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def _delta(left: Mapping[Key, Mapping[int, Sample]], right: Mapping[Key, Mapping[int, Sample]], keys: Sequence[Key], horizon: int) -> tuple[float, float]:
    left_metrics = _metrics(left[key][horizon] for key in keys)
    right_metrics = _metrics(right[key][horizon] for key in keys)
    return (float(left_metrics["auprc"]) - float(right_metrics["auprc"]),
            float(left_metrics["brier"]) - float(right_metrics["brier"]))


def paired_hierarchical_bootstrap(
    left_records: Iterable[Record],
    right_records: Iterable[Record],
    *,
    horizon: int = 8,
    replicates: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap paired deltas by resampling seed, then world, then episode."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    left, right, tree = _paired(left_records, right_records)
    if isinstance(horizon, bool) or horizon not in next(iter(left.values())):
        raise ValueError(f"horizon {horizon!r} is absent")
    keys = list(left)
    observed_auprc, observed_brier = _delta(left, right, keys, horizon)
    auprc_draws: list[float] = []
    brier_draws: list[float] = []
    rng = Random(seed)
    for _ in range(replicates):
        delta_auprc, delta_brier = _delta(left, right, _draw_keys(tree, rng), horizon)
        auprc_draws.append(delta_auprc)
        brier_draws.append(delta_brier)
    per_seed = {}
    for seed_id, worlds in sorted(tree.items()):
        seed_keys = [key for episodes in worlds.values() for key in episodes]
        delta_auprc, delta_brier = _delta(left, right, seed_keys, horizon)
        per_seed[seed_id] = {"delta_auprc": delta_auprc, "delta_brier": delta_brier,
                             "positive": delta_auprc > 0,
                             "direction": "left" if delta_auprc > 0 else "right" if delta_auprc < 0 else "tie"}
    return {
        "horizon": horizon,
        "mean_delta_auprc": observed_auprc,
        "delta_auprc_ci95": [_percentile(auprc_draws, 0.025), _percentile(auprc_draws, 0.975)],
        "mean_delta_brier": observed_brier,
        "delta_brier_ci95": [_percentile(brier_draws, 0.025), _percentile(brier_draws, 0.975)],
        "per_seed": per_seed,
        "replicates": replicates,
        "bootstrap_seed": seed,
    }


def evaluate_g2_g5(
    statistics: Mapping[str, Any],
    critical_ood_floors: Mapping[str, bool] | None = None,
    *,
    min_gain: float = 0.03,
    max_brier_degradation: float = 0.01,
    required_positive_seeds: int = 4,
    expected_seeds: int = 5,
) -> dict[str, bool]:
    """Evaluate registered G2/G5, returning False for missing or invalid evidence."""
    g2 = g5 = False
    try:
        gain = float(statistics["mean_delta_auprc"])
        lower = float(statistics["delta_auprc_ci95"][0])
        degradation = float(statistics["mean_delta_brier"])
        g2 = (all(isfinite(value) for value in (gain, lower, degradation))
              and lower > 0 and gain >= min_gain and degradation <= max_brier_degradation)
        per_seed = statistics["per_seed"]
        positives = [item["positive"] for item in per_seed.values()]
        valid_directions = len(per_seed) == expected_seeds and all(type(item) is bool for item in positives)
        valid_ood = bool(critical_ood_floors) and all(type(item) is bool and item for item in critical_ood_floors.values())
        g5 = valid_directions and sum(positives) >= required_positive_seeds and valid_ood
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        pass
    return {"g2": g2, "g5": g5}


metrics_by_horizon = aggregate_by_horizon
p2_metrics_by_horizon = aggregate_by_horizon
