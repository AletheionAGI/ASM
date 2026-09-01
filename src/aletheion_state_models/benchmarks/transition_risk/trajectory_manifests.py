"""Registered ATTR-TG1 protocol and reproducibility manifests."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

ARMS = ("asm_x_base", "transformer_base")
OPTIMIZER_SEEDS = (29, 43, 71, 89, 107)
HORIZON = 8
NEGATIVE_SAMPLES = 256
K = NEGATIVE_SAMPLES
BOOTSTRAP_SEED = 20_260_901
BOOTSTRAP_REPLICATES = 1_000
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProceduralLeakageControl:
    common_train_data_seed: int = 310_001
    common_validation_data_seed: int = 320_001
    optimizer_seed_used_for_data: bool = False
    test_generator_invoked_before_open: bool = False


PROCEDURAL_LEAKAGE_CONTROL = ProceduralLeakageControl()


@dataclass(frozen=True)
class SplitManifest:
    """Procedural inputs only. It deliberately cannot hold generated examples."""

    name: str
    seed: int
    world_count: int
    episodes_per_world: int
    family: str

    def __post_init__(self) -> None:
        if not all(
            type(item) is int
            for item in (self.seed, self.world_count, self.episodes_per_world)
        ):
            raise ValueError("split integers must have exact integer types")
        if self.world_count < 1 or self.episodes_per_world < 1:
            raise ValueError("split sizes must be positive")
        if self.name not in {
            "train",
            "validation",
            "test_id",
            "test_shift",
            "test_ood",
        }:
            raise ValueError("unregistered trajectory split")


@dataclass(frozen=True)
class TG2Gates:
    delta_auprc_min: float = 0.03
    delta_auprc_lower_ci_min: float = 0.0
    brier_delta_max: float = 0.01

    def __post_init__(self) -> None:
        if not all(type(item) is float for item in asdict(self).values()):
            raise ValueError("TG2 gates require float values")
        _require_finite(asdict(self))
        if self.delta_auprc_min != 0.03 or self.delta_auprc_lower_ci_min != 0.0:
            raise ValueError("TG2 AUPRC gates are fixed")
        if self.brier_delta_max != 0.01:
            raise ValueError("TG2 Brier gate is fixed")


@dataclass(frozen=True)
class TrajectoryProtocol:
    schema_version: int
    arms: tuple[str, ...]
    optimizer_seeds: tuple[int, ...]
    horizon: int
    negative_samples: int
    splits: tuple[SplitManifest, ...]
    leakage_control: ProceduralLeakageControl
    gates: TG2Gates
    bootstrap_seed: int
    bootstrap_replicates: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.schema_version,
            self.horizon,
            self.negative_samples,
            self.bootstrap_seed,
            self.bootstrap_replicates,
        )
        if not all(type(item) is int for item in integer_fields):
            raise ValueError("protocol integers must have exact integer types")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported trajectory protocol schema")
        if self.arms != ARMS or self.optimizer_seeds != OPTIMIZER_SEEDS:
            raise ValueError("unregistered arm/seed matrix")
        if self.horizon != HORIZON or self.negative_samples != NEGATIVE_SAMPLES:
            raise ValueError("trajectory H/K protocol mismatch")
        if self.splits != default_splits():
            raise ValueError("trajectory split protocol mismatch")
        if self.leakage_control != PROCEDURAL_LEAKAGE_CONTROL:
            raise ValueError("procedural leakage control must be explicit")
        if (self.bootstrap_seed, self.bootstrap_replicates) != (
            BOOTSTRAP_SEED,
            BOOTSTRAP_REPLICATES,
        ):
            raise ValueError("bootstrap protocol mismatch")


def default_splits() -> tuple[SplitManifest, ...]:
    return (
        SplitManifest("train", 310_001, 64, 4, "common_fixed"),
        SplitManifest("validation", 320_001, 16, 4, "common_fixed"),
        SplitManifest("test_id", 330_001, 32, 4, "id"),
        SplitManifest("test_shift", 330_002, 32, 4, "shift"),
        SplitManifest("test_ood", 330_003, 32, 4, "ood"),
    )


def default_trajectory_protocol() -> TrajectoryProtocol:
    return TrajectoryProtocol(
        SCHEMA_VERSION,
        ARMS,
        OPTIMIZER_SEEDS,
        HORIZON,
        NEGATIVE_SAMPLES,
        default_splits(),
        PROCEDURAL_LEAKAGE_CONTROL,
        TG2Gates(),
        BOOTSTRAP_SEED,
        BOOTSTRAP_REPLICATES,
    )


def manifest_paths(paths: Mapping[str, str | Path]) -> dict[str, Path]:
    """Normalize a named manifest while rejecting aliases and absent files."""
    if not paths or any(not isinstance(name, str) or not name for name in paths):
        raise ValueError("a manifest requires non-empty string names")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate manifest name")
    normalized = {name: Path(path) for name, path in paths.items()}
    if any(not path.is_file() for path in normalized.values()):
        raise ValueError("every manifest entry must be an existing file")
    return normalized


def protocol_payload(protocol: TrajectoryProtocol) -> dict:
    return asdict(protocol)


def _require_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite values are forbidden")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite(item)


__all__ = [
    "ARMS",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "HORIZON",
    "NEGATIVE_SAMPLES",
    "OPTIMIZER_SEEDS",
    "PROCEDURAL_LEAKAGE_CONTROL",
    "SCHEMA_VERSION",
    "K",
    "ProceduralLeakageControl",
    "SplitManifest",
    "TG2Gates",
    "TrajectoryProtocol",
    "default_splits",
    "default_trajectory_protocol",
    "manifest_paths",
    "protocol_payload",
]
