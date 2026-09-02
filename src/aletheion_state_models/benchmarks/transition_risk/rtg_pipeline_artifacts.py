"""Finalize the complete allowed ATTR-RTG artifact matrix before sealing."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .rtg_artifacts import (
    atomic_write_json,
    canonical_json,
    canonical_sha256,
    digest_files,
    digest_payload,
)
from .rtg_backbones import build_registered_backbone
from .rtg_checkpoint import load_terminal_checkpoint
from .rtg_config import TRAINING_SEEDS, load_registered_config, verify_preregistration
from .rtg_heads import DirectC, PhysicalD, TransitionG
from .rtg_pipeline_seal import (
    EVALUATOR_RELATIVE_PATH,
    GENERATOR_RELATIVE_PATH,
    SOURCE_RELATIVE_PATHS,
    PipelineSealPaths,
)
from .rtg_projection import load_projection
from .rtg_training_manifest_validation import validate_pipeline_training_manifest
from .rtg_truth_manifest_validation import validate_truth_manifest_relations

BACKBONES = ("asm", "transformer")
HEADS = ("G", "D", "C")
_METADATA = {"split": "train", "sealable": True, "protocol": "ATTR-RTG"}


def _arm(output: Path, kind: str, seed: int) -> Path:
    return output / f"{kind}_seed{seed}"


def _json(path: Path):
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if raw != canonical_json(value) + b"\n":
        raise ValueError(f"JSON artifact is not canonical: {path}")
    return value


def _finite(value: object) -> bool:
    if isinstance(value, bool | int | str) or value is None:
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _finite(item) for key, item in value.items()
        )
    return False


def _validate_normalization(path: Path) -> None:
    payload = _json(path)
    keys = ("pre_mean", "pre_std", "next_mean", "next_std")
    if not isinstance(payload, dict) or set(payload) != set(keys):
        raise ValueError("normalization schema differs")
    for key in keys:
        values = payload[key]
        if not isinstance(values, list) or len(values) != 28 or not _finite(values):
            raise ValueError("normalization vector differs")
        if key.endswith("std") and any(float(value) < 1e-6 for value in values):
            raise ValueError("normalization std is below frozen clamp")


def _validate_metrics(path: Path, kind: str, seed: int) -> None:
    payload = _json(path)
    if not isinstance(payload, dict) or not _finite(payload):
        raise ValueError("validation metrics must be a finite object")
    identity = (
        payload.get("kind"),
        payload.get("training_seed"),
        payload.get("split"),
        payload.get("terminal_update"),
    )
    if identity != (kind, seed, "validation", 1_000):
        raise ValueError("validation metrics identity differs")


def _validate_calibration(path: Path) -> None:
    payload = _json(path)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"temperature", "q95"}
        or not _finite(payload)
    ):
        raise ValueError("calibration parameter schema differs")
    if float(payload["temperature"]) <= 0 or not 0 <= float(payload["q95"]) <= 1:
        raise ValueError("calibration parameters are outside their domain")


def _validate_scores(path: Path, kind: str, seed: int) -> None:
    payload = _json(path)
    if not isinstance(payload, list) or not payload or not _finite(payload):
        raise ValueError("calibration scores must be a finite nonempty list")
    for row in payload:
        if not isinstance(row, dict) or row.get("seed") != seed:
            raise ValueError("calibration score seed differs")
        if not {
            "g_logit",
            "c_logit",
            "unsafe",
            "world_id",
            "episode_id",
            "t",
            "action_index",
        } <= set(row):
            raise ValueError("calibration score fields differ")
    if kind not in BACKBONES:
        raise ValueError("calibration score backbone differs")


def _validate_checkpoints(root: Path, output: Path) -> None:
    head_types = {"G": TransitionG, "D": PhysicalD, "C": DirectC}
    for kind in BACKBONES:
        for seed in TRAINING_SEEDS:
            arm = _arm(output, kind, seed)
            config = load_registered_config(root, kind, seed).to_dict()
            backbone = build_registered_backbone(root, kind, seed)
            load_terminal_checkpoint(
                arm / "backbone.pt",
                backbone,
                expected_kind=f"{kind}-backbone",
                expected_seed=seed,
                expected_update=1_000,
                expected_metadata=_METADATA,
                expected_config=config,
            )
            for name in HEADS:
                load_terminal_checkpoint(
                    arm / f"{name}.pt",
                    head_types[name](),
                    expected_kind=f"{kind}-{name}",
                    expected_seed=seed,
                    expected_update=1_000,
                    expected_metadata=_METADATA,
                )


def _source_paths(root: Path) -> tuple[dict[str, Path], Path, Path]:
    sources = {relative: root / relative for relative in SOURCE_RELATIVE_PATHS}
    generator = root / GENERATOR_RELATIVE_PATH
    evaluator = root / EVALUATOR_RELATIVE_PATH
    required = (*sources.values(), generator, evaluator)
    if any(not path.is_file() for path in required):
        raise ValueError("exact pipeline source inventory is incomplete")
    return sources, generator, evaluator


def _collect_paths(
    root: Path, output: Path
) -> tuple[PipelineSealPaths, dict[str, Path], dict[str, Path]]:
    backbones, heads, statistics, calibration = {}, {}, {}, {}
    validations, scores = {}, {}
    for kind in BACKBONES:
        for seed in TRAINING_SEEDS:
            prefix = f"{kind}.seed{seed}"
            arm = _arm(output, kind, seed)
            backbones[prefix] = arm / "backbone.pt"
            statistics[f"{prefix}.normalization"] = arm / "normalization.json"
            validations[prefix] = arm / "validation_metrics.json"
            scores[prefix] = arm / "calibration_scores.json"
            for name in HEADS:
                heads[f"{prefix}.{name}"] = arm / f"{name}.pt"
            for name in ("G", "C"):
                calibration[f"{prefix}.calibration_{name}"] = (
                    arm / f"calibration_{name}.json"
                )
    maps = {
        f"projection_{kind}": output / f"projection_{kind}.pt" for kind in BACKBONES
    }
    sources, generator, evaluator = _source_paths(root)
    manifests = {
        "train": output / "training_manifest.json",
        "validation": output / "validation_manifest.json",
        "calibration": output / "calibration_manifest.json",
    }
    state_records = {
        f"{kind}.seed{seed}.{prefix}_{split}": _arm(output, kind, seed)
        / f"{prefix}_{split}.pt"
        for kind in BACKBONES
        for seed in TRAINING_SEEDS
        for prefix in ("state_inputs", "states")
        for split in ("train", "validation", "calibration")
    }

    def relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                "pipeline artifact output must be below project root"
            ) from error

    def relatives(items: dict[str, Path]) -> dict[str, str]:
        return {name: relative(path) for name, path in items.items()}

    paths = PipelineSealPaths(
        root,
        relatives(sources),
        relative(generator),
        relative(evaluator),
        relatives(backbones),
        relatives(heads),
        relatives(maps),
        relatives(statistics),
        relatives(calibration),
        relatives(manifests),
        relatives(state_records),
    )
    return paths, validations, scores


def _manifest(schema: str, artifacts: dict[str, Path]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": schema,
        "artifacts": digest_payload(digest_files(artifacts)),
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def _paths_payload(paths: PipelineSealPaths) -> dict[str, object]:
    def strings(values):
        return {name: str(path) for name, path in values.items()}

    return {
        "schema": "PipelineSealPaths",
        "root": str(paths.root),
        "sources": strings(paths.sources),
        "generator": str(paths.generator),
        "evaluator": str(paths.evaluator),
        "backbone_checkpoints": strings(paths.backbone_checkpoints),
        "head_checkpoints": strings(paths.head_checkpoints),
        "maps": strings(paths.maps),
        "statistics": strings(paths.statistics),
        "calibration": strings(paths.calibration),
        "manifests": strings(paths.manifests),
        "state_records": strings(paths.state_records),
    }


def finalize_allowed_artifacts(root: str | Path, output: str | Path) -> Path:
    root, output = Path(root).resolve(), Path(output).resolve()
    verify_preregistration(root)
    paths, validations, scores = _collect_paths(root, output)
    allowed = _json(output / "allowed_manifest.json")
    if (
        not isinstance(allowed, dict)
        or allowed.get("schema") != "ATTR-RTG-ALLOWED-MANIFEST-V1"
    ):
        raise ValueError("allowed manifest schema differs")
    content = dict(allowed)
    claimed = content.pop("content_sha256", None)
    if claimed != canonical_sha256(content):
        raise ValueError("allowed manifest content digest differs")
    _validate_checkpoints(root, output)
    validate_truth_manifest_relations(output)
    validate_pipeline_training_manifest(output, paths)
    for kind in BACKBONES:
        load_projection(paths.resolve(paths.maps[f"projection_{kind}"]), kind)
    for kind in BACKBONES:
        for seed in TRAINING_SEEDS:
            prefix = f"{kind}.seed{seed}"
            _validate_normalization(
                paths.resolve(paths.statistics[f"{prefix}.normalization"])
            )
            _validate_metrics(validations[prefix], kind, seed)
            _validate_scores(scores[prefix], kind, seed)
            _validate_calibration(
                paths.resolve(paths.calibration[f"{prefix}.calibration_G"])
            )
            _validate_calibration(
                paths.resolve(paths.calibration[f"{prefix}.calibration_C"])
            )
    validation_manifest = _manifest("ATTR-RTG-VALIDATION-MANIFEST-V1", validations)
    calibration_files = scores | {
        name: paths.resolve(path) for name, path in paths.calibration.items()
    }
    calibration_manifest = _manifest(
        "ATTR-RTG-CALIBRATION-MANIFEST-V1", calibration_files
    )
    atomic_write_json(output / "validation_manifest.json", validation_manifest)
    atomic_write_json(output / "calibration_manifest.json", calibration_manifest)
    paths.artifact_groups()
    destination = output / "pipeline_paths.json"
    atomic_write_json(destination, _paths_payload(paths))
    return destination
