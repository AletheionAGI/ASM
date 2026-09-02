"""Strict artifact matrix and implementation-seal orchestration for ATTR-RTG."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .rtg_artifacts import file_sha256
from .rtg_config import verify_preregistration
from .rtg_seal import (
    REQUIRED_GROUPS,
    ImplementationSeal,
    create_implementation_seal,
    open_implementation_seal,
    read_implementation_seal,
    write_implementation_seal,
)
from .rtg_source_inventory import SOURCE_RELATIVE_PATHS

TRAINING_SEEDS = (29, 43, 71, 89, 107)
BACKBONES = ("asm", "transformer")
HEADS = ("G", "D", "C")
CALIBRATED_HEADS = ("G", "C")
MANIFEST_NAMES = ("train", "validation", "calibration")
GENERATOR_RELATIVE_PATH = (
    "src/aletheion_state_models/benchmarks/transition_risk/rtg_splits.py"
)
EVALUATOR_RELATIVE_PATH = (
    "src/aletheion_state_models/benchmarks/transition_risk/rtg_registered_evaluation.py"
)


def _matrix(
    backbones: tuple[str, ...], suffixes: tuple[str, ...] = ()
) -> tuple[str, ...]:
    if suffixes:
        return tuple(
            f"{backbone}.seed{seed}.{suffix}"
            for backbone in backbones
            for seed in TRAINING_SEEDS
            for suffix in suffixes
        )
    return tuple(
        f"{backbone}.seed{seed}" for backbone in backbones for seed in TRAINING_SEEDS
    )


BACKBONE_NAMES = _matrix(BACKBONES)
HEAD_NAMES = _matrix(BACKBONES, HEADS)
MAP_NAMES = tuple(f"projection_{backbone}" for backbone in BACKBONES)
STATISTICS_NAMES = tuple(f"{name}.normalization" for name in BACKBONE_NAMES)
CALIBRATION_NAMES = tuple(
    f"{name}.calibration_{head}" for name in BACKBONE_NAMES for head in CALIBRATED_HEADS
)
STATE_RECORD_NAMES = tuple(
    f"{name}.{prefix}_{split}"
    for name in BACKBONE_NAMES
    for prefix in ("state_inputs", "states")
    for split in MANIFEST_NAMES
)


@dataclass(frozen=True)
class PipelineSealPaths:
    """All preregistered implementation inputs relative to one project root."""

    root: str | Path
    sources: Mapping[str, str | Path]
    generator: str | Path
    evaluator: str | Path
    backbone_checkpoints: Mapping[str, str | Path]
    head_checkpoints: Mapping[str, str | Path]
    maps: Mapping[str, str | Path]
    statistics: Mapping[str, str | Path]
    calibration: Mapping[str, str | Path]
    manifests: Mapping[str, str | Path]
    state_records: Mapping[str, str | Path]

    def resolve(self, raw_path: str | Path) -> Path:
        """Resolve one artifact below the canonical root and reject traversal."""
        root = Path(self.root).resolve()
        raw = Path(raw_path)
        if raw.is_absolute():
            raise ValueError(
                "pipeline artifact paths must be relative to the canonical root"
            )
        candidate = (root / raw).resolve()
        if candidate == root or root not in candidate.parents:
            raise ValueError(
                "pipeline artifact must be a file below the canonical root"
            )
        return candidate

    def artifact_groups(self) -> dict[str, dict[str, Path]]:
        """Validate the complete matrix and return exactly the seven seal groups."""
        sources = self._canonical_sources()
        _require_exact(
            self, "backbone checkpoints", self.backbone_checkpoints, BACKBONE_NAMES
        )
        _require_exact(self, "head checkpoints", self.head_checkpoints, HEAD_NAMES)
        _require_exact(self, "maps", self.maps, MAP_NAMES)
        _require_exact(self, "statistics", self.statistics, STATISTICS_NAMES)
        _require_exact(self, "calibration", self.calibration, CALIBRATION_NAMES)
        _require_exact(self, "manifests", self.manifests, MANIFEST_NAMES)
        _require_exact(self, "state records", self.state_records, STATE_RECORD_NAMES)
        generator = self._canonical_module(
            "generator", self.generator, GENERATOR_RELATIVE_PATH
        )
        evaluator = self._canonical_module(
            "evaluator", self.evaluator, EVALUATOR_RELATIVE_PATH
        )
        groups = {
            "sources": sources,
            "generator": {"generator": generator},
            "evaluator": {"evaluator": evaluator},
            "calibration": _resolved_prefixed(self, "statistics", self.statistics)
            | _resolved_prefixed(self, "parameters", self.calibration),
            "manifests": _resolved(self, self.manifests)
            | _resolved(self, self.state_records),
            "maps": _resolved(self, self.maps),
            "checkpoints": _resolved_prefixed(
                self, "backbone", self.backbone_checkpoints
            )
            | _resolved_prefixed(self, "head", self.head_checkpoints),
        }
        if tuple(groups) != REQUIRED_GROUPS or len(groups) != 7:
            raise RuntimeError(
                "pipeline must produce exactly the seven canonical groups"
            )
        return groups

    def _canonical_sources(self) -> dict[str, Path]:
        _require_exact(self, "sources", self.sources, SOURCE_RELATIVE_PATHS)
        resolved = _resolved(self, self.sources)
        for relative in SOURCE_RELATIVE_PATHS:
            if resolved[relative] != (Path(self.root).resolve() / relative).resolve():
                raise ValueError(
                    "source inventory paths must match their canonical logical names"
                )
        return resolved

    def _canonical_module(self, name: str, raw: str | Path, relative: str) -> Path:
        path = self.resolve(raw)
        expected = (Path(self.root).resolve() / relative).resolve()
        if path != expected or not path.is_file():
            raise ValueError(
                f"{name} must be the canonical imported module: {relative}"
            )
        return path


def _verify_complete_preregistration(
    preregistration_manifest: str | Path, paths: PipelineSealPaths
) -> str:
    """Recheck the frozen document and all ten registered YAML artifacts."""
    digest = verify_preregistration(paths.root)
    if file_sha256(preregistration_manifest) != digest:
        raise ValueError(
            "preregistration manifest differs from the complete frozen registration"
        )
    return digest


def create_pipeline_seal(
    preregistration_manifest: str | Path, paths: PipelineSealPaths
) -> ImplementationSeal:
    _verify_complete_preregistration(preregistration_manifest, paths)
    return create_implementation_seal(preregistration_manifest, paths.artifact_groups())


def write_pipeline_seal(
    preregistration_manifest: str | Path,
    paths: PipelineSealPaths,
    seal_path: str | Path,
) -> ImplementationSeal:
    seal = create_pipeline_seal(preregistration_manifest, paths)
    write_implementation_seal(seal, seal_path)
    return seal


def verify_pipeline_seal(
    seal_path: str | Path,
    preregistration_manifest: str | Path,
    paths: PipelineSealPaths,
) -> ImplementationSeal:
    _verify_complete_preregistration(preregistration_manifest, paths)
    seal = read_implementation_seal(seal_path)
    candidate = create_pipeline_seal(preregistration_manifest, paths)
    if candidate != seal or candidate.sha256 != seal.sha256:
        raise ValueError(
            "implementation seal differs from the complete pipeline matrix"
        )
    return seal


def open_pipeline_tests(
    seal_path: str | Path,
    preregistration_manifest: str | Path,
    paths: PipelineSealPaths,
):
    """Open only after the candidate exactly matches the complete pipeline seal."""
    _verify_complete_preregistration(preregistration_manifest, paths)
    verified = verify_pipeline_seal(seal_path, preregistration_manifest, paths)
    capability = open_implementation_seal(
        seal_path,
        preregistration_manifest,
        paths.artifact_groups(),
    )
    capability._mark_pipeline_verified(verified.sha256)
    return capability


def _require_exact(
    owner: PipelineSealPaths,
    name: str,
    paths: Mapping[str, str | Path],
    expected: tuple[str, ...],
) -> None:
    if set(paths) != set(expected) or len(paths) != len(expected):
        missing = sorted(set(expected) - set(paths))
        extra = sorted(set(paths) - set(expected))
        raise ValueError(f"{name} matrix differs; missing={missing}, extra={extra}")
    for path in _resolved(owner, paths).values():
        if not path.is_file():
            raise ValueError(f"{name} must contain only named existing files")


def _resolved(
    owner: PipelineSealPaths, paths: Mapping[str, str | Path]
) -> dict[str, Path]:
    return {name: owner.resolve(path) for name, path in paths.items()}


def _resolved_prefixed(
    owner: PipelineSealPaths, prefix: str, paths: Mapping[str, str | Path]
) -> dict[str, Path]:
    return {f"{prefix}.{name}": owner.resolve(path) for name, path in paths.items()}
