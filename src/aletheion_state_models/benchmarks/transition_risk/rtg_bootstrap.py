"""Frozen seed→world→episode PCG64 bootstrap for ATTR-RTG metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import floor, isfinite
from typing import Any, TypeAlias

import numpy as np

from .rtg_config import TRAINING_SEEDS
from .rtg_metrics_state import validate_six_candidate_clusters
from .rtg_pairing import canonical_records, require_byte_equivalent

Record: TypeAlias = Mapping[str, Any]
Estimator: TypeAlias = Callable[[Sequence[Record]], Mapping[str, float]]
BOOTSTRAP_SEED = 20260903
BOOTSTRAP_REPLICATES = 1000


def percentile_type7(values: Sequence[float], probability: float) -> float:
    """R/NumPy linear (Hyndman-Fan type 7) sample percentile."""
    if not values or not 0 <= probability <= 1:
        raise ValueError("non-empty values and probability in [0,1] are required")
    ordered = sorted(float(value) for value in values)
    if any(not isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def _tree(records: Sequence[Record]):
    tree: dict[int, dict[str, dict[str, list[Record]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in records:
        tree[row["seed"]][str(row["world_id"])][str(row["episode_id"])].append(row)
    if tuple(sorted(tree)) != tuple(sorted(TRAINING_SEEDS)):
        raise ValueError("bootstrap requires exactly registered seeds 29,43,71,89,107")
    return tree


def _draw(tree: Mapping[int, Mapping[str, Mapping[str, Sequence[Record]]]],
          rng: np.random.Generator) -> list[Record]:
    selected: list[Record] = []
    seeds = sorted(tree)
    for seed_draw, seed_index in enumerate(rng.integers(0, len(seeds), size=5)):
        seed = seeds[int(seed_index)]
        worlds = sorted(tree[seed])
        for world_draw, world_index in enumerate(rng.integers(0, len(worlds), size=len(worlds))):
            world = worlds[int(world_index)]
            episodes = sorted(tree[seed][world])
            indices = rng.integers(0, len(episodes), size=len(episodes))
            for episode_draw, episode_index in enumerate(indices):
                episode = episodes[int(episode_index)]
                for row in tree[seed][world][episode]:
                    copy = dict(row)
                    copy.update({
                        "_source_seed": seed,
                        "_source_world_id": world,
                        "_source_episode_id": episode,
                        "seed": seed_draw,
                        "world_id": f"{world_draw}:{world}",
                        "episode_id": f"{episode_draw}:{episode}",
                    })
                    selected.append(copy)
    return selected


def _evaluate(estimator: Estimator, records: Sequence[Record]) -> dict[str, float]:
    try:
        raw = estimator(records)
        result = {str(name): float(value) for name, value in raw.items()}
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("bootstrap estimator failed") from error
    if not result or any(not isfinite(value) for value in result.values()):
        raise ValueError("bootstrap estimator returned missing or nonfinite metrics")
    return result


def hierarchical_bootstrap(records: Iterable[Record], estimator: Estimator) -> dict[str, Any]:
    """Run exactly 1,000 PCG64 replicates and fail on any invalid replicate."""
    rows = validate_six_candidate_clusters(records)
    tree = _tree(rows)
    observed = _evaluate(estimator, rows)
    draws = {name: [] for name in observed}
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    for _ in range(BOOTSTRAP_REPLICATES):
        replicate = _evaluate(estimator, _draw(tree, rng))
        if replicate.keys() != observed.keys():
            raise ValueError("estimator metric schema changed in a replicate")
        for name, value in replicate.items():
            draws[name].append(value)
    metrics = {
        name: {
            "estimate": observed[name],
            "ci95": [percentile_type7(values, 0.025), percentile_type7(values, 0.975)],
        }
        for name, values in draws.items()
    }
    return {"bootstrap_seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
            "bit_generator": "PCG64", "cluster_order": "seed-world-episode",
            "metrics": metrics}


def paired_hierarchical_bootstrap(
    left_records: Iterable[Record], right_records: Iterable[Record], estimator: Estimator,
) -> dict[str, Any]:
    """Bootstrap paired rows after verifying identical candidate identities."""
    left = validate_six_candidate_clusters(left_records)
    right = validate_six_candidate_clusters(right_records)
    canonical_records(left)
    canonical_records(right)
    require_byte_equivalent(left, right)
    paired = []
    for left_row, right_row in zip(left, right, strict=True):
        item = dict(left_row)
        item["left"] = left_row
        item["right"] = right_row
        paired.append(item)
    return hierarchical_bootstrap(paired, estimator)
