"""Relational validation for the canonical ATTR-RTG training manifest."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .rtg_artifacts import (
    canonical_json,
    canonical_sha256,
    digest_files,
    digest_payload,
)


def _training_state_name(logical_name: str) -> str:
    if ".state_inputs_" in logical_name:
        arm, split = logical_name.split(".state_inputs_", 1)
        return f"states.{arm}.input.{split}"
    if ".states_" in logical_name:
        arm, split = logical_name.split(".states_", 1)
        return f"states.{arm}.{split}"
    raise ValueError("unknown sealed state ledger name")


def validate_training_manifest(
    path: str | Path,
    *,
    allowed_manifest: str | Path,
    truth_manifest: str | Path,
    backbone_checkpoints: Mapping[str, str | Path],
    head_checkpoints: Mapping[str, str | Path],
    state_records: Mapping[str, str | Path],
) -> None:
    """Require canonical bytes and exact hashes for all 102 training relations."""
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid training manifest JSON") from error
    if raw != canonical_json(payload) + b"\n":
        raise ValueError("training manifest JSON is not canonical")
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "artifacts", "content_sha256",
    } or payload["schema"] != "ATTR-RTG-TRAINING-MANIFEST-V1":
        raise ValueError("training manifest schema differs")
    expected_paths = {
        "allowed_manifest": allowed_manifest,
        "truth_manifest": truth_manifest,
        **{f"backbone.{key}": value for key, value in backbone_checkpoints.items()},
        **{f"head.{key}": value for key, value in head_checkpoints.items()},
        **{
            _training_state_name(key): value
            for key, value in state_records.items()
        },
    }
    expected = digest_payload(digest_files(expected_paths))
    if payload["artifacts"] != expected:
        raise ValueError("training manifest artifact relations or hashes differ")
    body = {"schema": payload["schema"], "artifacts": payload["artifacts"]}
    if payload["content_sha256"] != canonical_sha256(body):
        raise ValueError("training manifest content digest differs")


def validate_pipeline_training_manifest(output: Path, paths: object) -> None:
    """Validate the training manifest against one resolved PipelineSealPaths matrix."""
    validate_training_manifest(
        output / "training_manifest.json",
        allowed_manifest=output / "allowed_manifest.json",
        truth_manifest=output / "truth_manifest.json",
        backbone_checkpoints={key: paths.resolve(value) for key, value in paths.backbone_checkpoints.items()},
        head_checkpoints={key: paths.resolve(value) for key, value in paths.head_checkpoints.items()},
        state_records={key: paths.resolve(value) for key, value in paths.state_records.items()},
    )


