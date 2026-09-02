"""Frozen split registry with a capability barrier around ATTR-RTG tests."""

from __future__ import annotations

from dataclasses import dataclass

from world_model.hazard_world_types import HazardWorldConfig

from .dataset import make_worlds
from .rtg_seal import TestOpenCapability, require_test_capability


@dataclass(frozen=True)
class SplitSpec:
    name: str
    split_seed: int
    world_count: int
    episodes_per_world: int
    dynamic_family: str
    test_only: bool


SPLITS = (
    SplitSpec("train", 360001, 64, 4, "baseline", False),
    SplitSpec("validation", 360002, 16, 4, "baseline", False),
    SplitSpec("calibration", 360003, 16, 4, "baseline", False),
    SplitSpec("test_id", 360101, 32, 4, "baseline", True),
    SplitSpec("test_shift", 360102, 32, 4, "shift", True),
    SplitSpec("test_ood", 360103, 32, 4, "ood", True),
)
_BY_NAME = {item.name: item for item in SPLITS}


def split_spec(name: str) -> SplitSpec:
    try:
        return _BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"unknown ATTR-RTG split: {name}") from error


def _make(spec: SplitSpec) -> tuple[HazardWorldConfig, ...]:
    return tuple(
        make_worlds(
            spec.world_count,
            spec.split_seed,
            dynamic_family=spec.dynamic_family,
            max_steps=16,
        )
    )


def make_allowed_worlds(name: str) -> tuple[HazardWorldConfig, ...]:
    """Generate only train, validation, or calibration before implementation seal."""
    spec = split_spec(name)
    if spec.test_only:
        raise PermissionError("registered test worlds require a test-open capability")
    return _make(spec)


def make_test_worlds(
    name: str,
    split_id: int,
    capability: TestOpenCapability,
    seal_sha256: str,
) -> tuple[HazardWorldConfig, ...]:
    """Generate the named registered split only after one-shot seal opening."""
    spec = split_spec(name)
    if not spec.test_only:
        raise ValueError("test capability cannot be used for an allowed split")
    if type(split_id) is not int or split_id != spec.split_seed:
        raise ValueError("test split ID differs from the frozen registry")
    require_test_capability(capability, seal_sha256)
    capability._register_split_generation(name)
    return _make(spec)


def assert_disjoint_world_ids(
    groups: tuple[tuple[HazardWorldConfig, ...], ...],
) -> None:
    owners: set[str] = set()
    for worlds in groups:
        identifiers = {world.world_id for world in worlds}
        if len(identifiers) != len(worlds) or owners.intersection(identifiers):
            raise ValueError("ATTR-RTG world IDs are duplicated across splits")
        owners.update(identifiers)
