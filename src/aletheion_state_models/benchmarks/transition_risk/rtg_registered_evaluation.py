"""Canonical artifact-backed evaluation for the registered ATTR-RTG tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world_model.hazard_world_types import HazardWorldConfig

from .rtg_cloning import materialize_origin_truth
from .rtg_config import TRAINING_SEEDS
from .rtg_dataset import prepare_rtg_input_origins
from .rtg_evaluator import EvaluationBatch, evaluate_c_records, evaluate_g_records
from .rtg_integrity import IntegrityEvidence
from .rtg_pipeline_seal import BACKBONES, PipelineSealPaths
from .rtg_registered_artifacts import RegisteredArm as _Arm
from .rtg_registered_artifacts import load_registered_arm as _load_arm
from .rtg_registered_summary import summarize_registered_evaluation
from .rtg_seal import TestOpenCapability, claim_evaluation_authority
from .rtg_splits import split_spec
from .rtg_state_records import (
    CandidateStateInputRecord,
    CandidateStateRecord,
    attach_candidate_truths,
    export_candidate_state_inputs,
)
from .rtg_types import NONSTOP_ACTIONS

_TEST_SPLITS = ("test_id", "test_shift", "test_ood")


@dataclass(frozen=True)
class _Dependencies:
    load_arm: Callable[[Path, PipelineSealPaths, str, int], _Arm]
    prepare_origins: Callable[..., tuple[Any, ...]]
    export_inputs: Callable[..., tuple[CandidateStateInputRecord, ...]]
    materialize_truth: Callable[[Any], Any]
    attach_truths: Callable[..., tuple[CandidateStateRecord, ...]]
    evaluate_g: Callable[..., EvaluationBatch]
    evaluate_c: Callable[..., EvaluationBatch]
    summarize: Callable[[Mapping[str, Any], IntegrityEvidence], dict[str, Any]]


def _record_payload(
    record: CandidateStateRecord, arm: _Arm, seed: int
) -> dict[str, Any]:
    normalized_pre = arm.normalization.normalize_pre(record.pre_state)
    normalized_next = arm.normalization.normalize_next(record.next_state)
    persistence = arm.normalization.normalize_next(record.pre_state)
    return {
        "seed": seed,
        "split_id": record.split_id,
        "world_id": record.world_id,
        "episode_id": record.episode_id,
        "t": record.t,
        "action_index": record.action_index,
        "pre_state": record.pre_state.tolist(),
        "next_state": record.next_state.tolist(),
        "normalized_state": normalized_pre.tolist(),
        "true_next_state": normalized_next.tolist(),
        "persistence_state": persistence.tolist(),
        "fixed_frame": record.fixed_frame.tolist(),
        "y_common": list(record.physical_target),
        "persistence_target": list(record.persistence_target),
        "candidate_unsafe": record.unsafe,
        "failure_delay": record.failure_delay,
    }


def _rows_payload(
    records: tuple[CandidateStateRecord, ...], arm: _Arm, seed: int
) -> tuple[dict[str, Any], ...]:
    brake_index = NONSTOP_ACTIONS.index("BRAKE")
    brake_by_origin = {
        (item.split_id, item.world_id, item.episode_id, item.t): item.unsafe
        for item in records
        if item.action_index == brake_index
    }
    rows = []
    for record in records:
        row = _record_payload(record, arm, seed)
        origin = (record.split_id, record.world_id, record.episode_id, record.t)
        if origin not in brake_by_origin:
            raise ValueError("candidate origin has no BRAKE consequence")
        row["brake_unsafe"] = brake_by_origin[origin]
        rows.append(row)
    return tuple(rows)


def _batch_payload(batch: EvaluationBatch) -> dict[str, Any]:
    """Persist records only; summaries recompute metrics from authenticated rows."""
    return {"records": list(batch.records)}


def _validate_worlds(
    split_worlds: Mapping[str, tuple[HazardWorldConfig, ...]], *, strict_registry: bool
) -> None:
    if set(split_worlds) != set(_TEST_SPLITS) or len(split_worlds) != len(_TEST_SPLITS):
        raise ValueError("registered evaluation requires exactly the three test splits")
    identifiers: set[str] = set()
    for name in _TEST_SPLITS:
        worlds = split_worlds[name]
        expected = split_spec(name).world_count
        if not worlds or (strict_registry and len(worlds) != expected):
            raise ValueError(f"registered world count differs for {name}")
        current = {world.world_id for world in worlds}
        if len(current) != len(worlds) or identifiers.intersection(current):
            raise ValueError("registered test world IDs must be globally unique")
        identifiers.update(current)


def _finite_json(value: object) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("registered evaluation result is not finite JSON") from error


def _evaluate_registered_test(
    root: str | Path,
    paths: PipelineSealPaths,
    split_worlds: Mapping[str, tuple[HazardWorldConfig, ...]],
    capability: TestOpenCapability,
    evidence: IntegrityEvidence,
    seal_sha256: str,
    *,
    dependencies: _Dependencies,
    strict_registry: bool,
) -> dict[str, object]:
    claim_evaluation_authority(capability, evidence, seal_sha256)
    _validate_worlds(split_worlds, strict_registry=strict_registry)

    # Load and strictly validate the complete sealed model matrix before any rollout.
    arms = {
        (kind, seed): dependencies.load_arm(Path(root), paths, kind, seed)
        for kind in BACKBONES
        for seed in TRAINING_SEEDS
    }
    prepared = {}
    for name in _TEST_SPLITS:
        spec = split_spec(name)
        prepared[name] = dependencies.prepare_origins(
            split_worlds[name],
            episodes_per_world=4,
            split_id=name,
            split_seed=spec.split_seed,
        )
        if not prepared[name]:
            raise ValueError(f"registered test origins are empty: {name}")

    # Causal barrier: every arm exports inputs before any privileged truth exists.
    exported_inputs = {
        (kind, seed, name): dependencies.export_inputs(
            prepared[name], arms[kind, seed].exporter, arms[kind, seed].projection
        )
        for kind in BACKBONES
        for seed in TRAINING_SEEDS
        for name in _TEST_SPLITS
    }
    truths = {
        name: tuple(dependencies.materialize_truth(origin) for origin in prepared[name])
        for name in _TEST_SPLITS
    }
    split_results: dict[str, dict[str, object]] = {
        name: {"systems": {}} for name in _TEST_SPLITS
    }
    for kind in BACKBONES:
        for seed in TRAINING_SEEDS:
            arm = arms[kind, seed]
            system_name = f"{kind}.seed{seed}"
            for name in _TEST_SPLITS:
                records = dependencies.attach_truths(
                    exported_inputs[kind, seed, name], truths[name]
                )
                rows = _rows_payload(records, arm, seed)
                systems = split_results[name]["systems"]
                assert isinstance(systems, dict)
                systems[system_name] = {
                    "G": _batch_payload(
                        dependencies.evaluate_g(rows, arm.g, arm.d, arm.g_calibration)
                    ),
                    "C": _batch_payload(
                        dependencies.evaluate_c(rows, arm.c, arm.c_calibration)
                    ),
                }
    result: dict[str, object] = {
        "kind": "attr_rtg_registered_evaluation",
        "schema_version": 1,
        "episodes_per_world": 4,
        "splits": split_results,
    }
    summary = dependencies.summarize(result, evidence)
    result["summary"] = summary
    result["gates"] = summary["gates"]
    _finite_json(result)
    return result


def _default_dependencies() -> _Dependencies:
    return _Dependencies(
        _load_arm,
        prepare_rtg_input_origins,
        export_candidate_state_inputs,
        materialize_origin_truth,
        attach_candidate_truths,
        evaluate_g_records,
        evaluate_c_records,
        summarize_registered_evaluation,
    )


def evaluate_registered_test(
    root: str | Path,
    paths: PipelineSealPaths,
    split_worlds: Mapping[str, tuple[HazardWorldConfig, ...]],
    capability: TestOpenCapability,
    evidence: IntegrityEvidence,
    seal_sha256: str,
) -> object:
    """Evaluate all registered arms using only sealed production artifacts.

    The caller must create ``split_worlds`` through the one-shot test capability.
    This function has no public callback or model-injection surface.
    """
    return _evaluate_registered_test(
        root,
        paths,
        split_worlds,
        capability,
        evidence,
        seal_sha256,
        dependencies=_default_dependencies(),
        strict_registry=True,
    )
