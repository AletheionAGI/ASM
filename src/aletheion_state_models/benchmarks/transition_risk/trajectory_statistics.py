from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .metrics import average_precision, brier_score
from .trajectory_evaluation import HORIZONS, TrajectoryRecord, validate_record


def _event(record: TrajectoryRecord, horizon: int) -> int:
    return int(any(record.unsafe_truth[:horizon]))


def _logloss(labels: Sequence[int], scores: Sequence[float]) -> float:
    epsilon = 1e-15
    return math.fsum(
        -(
            y * math.log(min(1 - epsilon, max(epsilon, p)))
            + (1 - y) * math.log(min(1 - epsilon, max(epsilon, 1 - p)))
        )
        for y, p in zip(labels, scores)
    ) / len(labels)


def quality_by_horizon(
    records: Iterable[TrajectoryRecord],
) -> dict[str, dict[str, Any]]:
    """Pool origins and report risk quality and trajectory NLL by field."""
    rows = tuple(records)
    if not rows:
        raise ValueError("records must not be empty")
    for row in rows:
        validate_record(row)
    if len({(row.identity.arm, row.identity.split) for row in rows}) != 1:
        raise ValueError("records must describe exactly one arm and split")
    output = {}
    for horizon in HORIZONS:
        labels = [_event(row, horizon) for row in rows]
        scores = [row.risk_by_horizon[horizon] for row in rows]
        fields = tuple(rows[0].trajectory_nll[horizon])
        if any(tuple(row.trajectory_nll[horizon]) != fields for row in rows):
            raise ValueError("all records must have identical ordered NLL fields")
        output[f"H{horizon}"] = {
            "auprc": average_precision(labels, scores),
            "brier": brier_score(labels, scores),
            "event_logloss": _logloss(labels, scores),
            "prevalence": sum(labels) / len(labels),
            "n": len(labels),
            "positives": sum(labels),
            "trajectory_nll": {
                field: math.fsum(row.trajectory_nll[horizon][field] for row in rows)
                / len(rows)
                for field in fields
            },
        }
    return output


def fit_fpr_thresholds(
    validation_records: Iterable[TrajectoryRecord], *, max_fpr: float = 0.05
) -> dict[int, float]:
    """Choose per-horizon validation thresholds with maximum recall at FPR <= 5%."""
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be in [0, 1]")
    rows = tuple(validation_records)
    if not rows or any(row.identity.split != "validation" for row in rows):
        raise ValueError("thresholds require non-empty validation records")
    for row in rows:
        validate_record(row)
    if len({row.identity.arm for row in rows}) != 1:
        raise ValueError("thresholds must be fit for exactly one arm")
    result = {}
    for horizon in HORIZONS:
        labels = [_event(row, horizon) for row in rows]
        scores = [row.risk_by_horizon[horizon] for row in rows]
        candidates = [math.nextafter(max(scores), math.inf)] + sorted(
            set(scores), reverse=True
        )
        best = (0.0, candidates[0])
        negatives = len(labels) - sum(labels)
        positives = sum(labels)
        for threshold in candidates:
            tp = sum(y and p >= threshold for y, p in zip(labels, scores))
            fp = sum(not y and p >= threshold for y, p in zip(labels, scores))
            fpr = fp / negatives if negatives else 0.0
            recall = tp / positives if positives else 0.0
            if fpr <= max_fpr and (
                recall > best[0] or (recall == best[0] and threshold > best[1])
            ):
                best = (recall, threshold)
        result[horizon] = float(best[1])
    return result


def threshold_metrics(
    records: Iterable[TrajectoryRecord], thresholds: Mapping[int, float]
) -> dict[str, dict[str, float | int]]:
    rows = tuple(records)
    if not rows or set(thresholds) != set(HORIZONS):
        raise ValueError("records and exact H1/H4/H8 thresholds are required")
    quality_by_horizon(rows)
    output = {}
    for horizon in HORIZONS:
        threshold = float(thresholds[horizon])
        if not math.isfinite(threshold):
            raise ValueError("thresholds must be finite")
        labels = [_event(row, horizon) for row in rows]
        scores = [row.risk_by_horizon[horizon] for row in rows]
        tp = sum(y and p >= threshold for y, p in zip(labels, scores))
        fp = sum(not y and p >= threshold for y, p in zip(labels, scores))
        positives, negatives = sum(labels), len(labels) - sum(labels)
        output[f"H{horizon}"] = {
            "threshold": threshold,
            "recall": tp / positives if positives else 0.0,
            "fpr": fp / negatives if negatives else 0.0,
            "positives": positives,
            "negatives": negatives,
        }
    return output


def simple_lead(
    records: Iterable[TrajectoryRecord], threshold: float, *, horizon: int = 8
) -> dict[str, float | int]:
    """Report steps from first alarm anchor to the first observed unsafe step."""
    if horizon not in HORIZONS or not math.isfinite(threshold):
        raise ValueError("invalid horizon or threshold")
    episodes: dict[tuple[int, str, str, str], list[TrajectoryRecord]] = defaultdict(
        list
    )
    for row in records:
        validate_record(row)
        key = (
            row.identity.seed,
            row.identity.arm,
            row.identity.world_id,
            row.identity.episode_id,
        )
        episodes[key].append(row)
    leads = []
    events = 0
    for rows in episodes.values():
        event_steps = [
            row.identity.anchor + offset
            for row in rows
            for offset, unsafe in enumerate(row.unsafe_truth)
            if unsafe
        ]
        if not event_steps:
            continue
        events += 1
        event = min(event_steps)
        alarms = [
            row.identity.anchor
            for row in rows
            if row.identity.anchor < event and row.risk_by_horizon[horizon] >= threshold
        ]
        if alarms:
            leads.append(event - min(alarms))
    return {
        "event_episodes": events,
        "detected_episodes": len(leads),
        "detection_rate": len(leads) / events if events else 0.0,
        "mean_lead_steps": math.fsum(leads) / len(leads) if leads else 0.0,
    }


def _pair_key(row: TrajectoryRecord) -> tuple[int, str, str, str, int]:
    item = row.identity
    return item.seed, item.split, item.world_id, item.episode_id, item.anchor


def _paired(left: Sequence[TrajectoryRecord], right: Sequence[TrajectoryRecord]):
    a, b = {_pair_key(row): row for row in left}, {_pair_key(row): row for row in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError("arms require unique, identical paired origins")
    if (
        len({row.identity.arm for row in left}) != 1
        or len({row.identity.arm for row in right}) != 1
    ):
        raise ValueError("each paired side must contain exactly one arm")
    for key, left_row in a.items():
        if left_row.unsafe_truth != b[key].unsafe_truth:
            raise ValueError("paired origins must share physical unsafe truth")
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key in a:
        seed, _split, world, episode, _anchor = key
        tree[seed][world][episode].append(key)
    return a, b, tree


def _draw_episodes(tree, rng: random.Random):
    selected = []
    seeds = sorted(tree)
    for seed in rng.choices(seeds, k=len(seeds)):
        worlds = sorted(tree[seed])
        for world in rng.choices(worlds, k=len(worlds)):
            episodes = sorted(tree[seed][world])
            for episode in rng.choices(episodes, k=len(episodes)):
                selected.extend(tree[seed][world][episode])
    return selected


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[min(lower + 1, len(ordered) - 1)] * fraction
    )


def _metric_delta(a, b, keys, horizon, metric):
    left = [a[key] for key in keys]
    right = [b[key] for key in keys]
    return float(quality_by_horizon(left)[f"H{horizon}"][metric]) - float(
        quality_by_horizon(right)[f"H{horizon}"][metric]
    )


def paired_hierarchical_bootstrap(
    left_records: Iterable[TrajectoryRecord],
    right_records: Iterable[TrajectoryRecord],
    *,
    replicates: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired seed→world→episode bootstrap; every selected episode keeps all anchors."""
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    left, right = tuple(left_records), tuple(right_records)
    if not left or not right:
        raise ValueError("paired records must not be empty")
    a, b, tree = _paired(left, right)
    rng = random.Random(seed)
    metrics = ("auprc", "brier", "event_logloss")
    output = {}
    observed_keys = list(a)
    for horizon in HORIZONS:
        output[f"H{horizon}"] = {}
        draws = {metric: [] for metric in metrics}
        for _ in range(replicates):
            keys = _draw_episodes(tree, rng)
            for metric in metrics:
                draws[metric].append(_metric_delta(a, b, keys, horizon, metric))
        for metric in metrics:
            values = draws[metric]
            output[f"H{horizon}"][metric] = {
                "delta": _metric_delta(a, b, observed_keys, horizon, metric),
                "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
            }
    nll_output = {}
    fields = tuple(left[0].trajectory_nll[1])
    for horizon in HORIZONS:
        nll_output[f"H{horizon}"] = {}
        for field in fields:

            def nll_delta(keys, selected_horizon=horizon, selected_field=field):
                return (
                    math.fsum(
                        a[key].trajectory_nll[selected_horizon][selected_field]
                        for key in keys
                    )
                    - math.fsum(
                        b[key].trajectory_nll[selected_horizon][selected_field]
                        for key in keys
                    )
                ) / len(keys)

            observed = nll_delta(observed_keys)
            draws = [nll_delta(_draw_episodes(tree, rng)) for _ in range(replicates)]
            nll_output[f"H{horizon}"][field] = {
                "delta": observed,
                "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
            }
    return {
        "left_arm": left[0].identity.arm,
        "right_arm": right[0].identity.arm,
        "replicates": replicates,
        "by_horizon": output,
        "trajectory_nll": nll_output,
    }


def build_summary(
    records: Iterable[TrajectoryRecord],
    *,
    thresholds: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    rows = tuple(records)
    result: dict[str, Any] = {"quality_by_horizon": quality_by_horizon(rows)}
    if thresholds is not None:
        result["threshold_metrics"] = threshold_metrics(rows, thresholds)
        result["lead_h8"] = simple_lead(rows, thresholds[8], horizon=8)
    return result
