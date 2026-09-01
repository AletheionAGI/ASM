"""JSON persistence and leak-free world splitting for HazardWorld."""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import math
from typing import Any, Iterable, Mapping
from .hazard_world_types import HazardWorldConfig, HazardWorldState


def serialize_config(config: HazardWorldConfig) -> str:
    return json.dumps(_jsonable(asdict(config)), sort_keys=True, separators=(",", ":"))


def serialize_state(state: HazardWorldState) -> str:
    return json.dumps(_jsonable(asdict(state)), sort_keys=True, separators=(",", ":"))


def config_from_json(value: str) -> HazardWorldConfig:
    data = json.loads(value)
    for key in ("goal", "initial_agent"):
        data[key] = tuple(data[key])
    for key in ("walls", "traps", "moving_hazards", "hazard_velocities"):
        data[key] = tuple(map(tuple, data[key]))
    return HazardWorldConfig(**data)


def state_from_json(value: str) -> HazardWorldState:
    data = json.loads(value)
    for key in ("agent", "velocity", "hazards", "hazard_velocities"):
        data[key] = (
            tuple(map(tuple, data[key]))
            if key in {"hazards", "hazard_velocities"}
            else tuple(data[key])
        )
    return HazardWorldState(**data)


def split_worlds(
    worlds: Iterable[HazardWorldConfig], ratios: Mapping[str, float] | None = None
) -> dict[str, list[HazardWorldConfig]]:
    ratios = ratios or {
        "train": 0.7,
        "validation": 0.1,
        "test_id": 0.1,
        "test_shift": 0.05,
        "test_ood": 0.05,
    }
    if (
        not ratios
        or any(value < 0 for value in ratios.values())
        or not math.isclose(sum(ratios.values()), 1.0)
    ):
        raise ValueError("split ratios must be non-negative and sum to one")
    result = {name: [] for name in ratios}
    seen: set[str] = set()
    boundaries = []
    total = 0.0
    for name, fraction in ratios.items():
        total += fraction
        boundaries.append((total, name))
    for world in worlds:
        if world.world_id in seen:
            raise ValueError(f"world leakage: duplicate world_id {world.world_id}")
        seen.add(world.world_id)
        score = (
            int(hashlib.sha256(world.world_id.encode()).hexdigest()[:16], 16) / 2**64
        )
        name = next(
            name
            for bound, name in boundaries
            if score < bound or math.isclose(bound, 1.0)
        )
        result[name].append(world)
    return result


def assert_no_world_leakage(splits: Mapping[str, Iterable[HazardWorldConfig]]) -> None:
    owner: dict[str, str] = {}
    for split, worlds in splits.items():
        for world in worlds:
            if world.world_id in owner:
                raise ValueError(
                    f"world {world.world_id} occurs in {owner[world.world_id]} and {split}"
                )
            owner[world.world_id] = split


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
