"""Frozen configuration and preregistration validation for ATTR-RTG."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import TypeVar

import yaml

from drm_language_emitter import DRMConfig
from transformer.tiny_transformer import TinyTransformerConfig

TRAINING_SEEDS = (29, 43, 71, 89, 107)
BACKBONE_PARAMETER_COUNTS = {"asm": 30_122, "transformer": 30_120}
PREREGISTRATION_MANIFEST = Path("docs/ATTR_RTG_PREREGISTRATION_MANIFEST.json")
PREREGISTRATION_MANIFEST_SHA256 = (
    "4db4e22029431c6544a3d8c032cd75d45308250b3b601c0fe437ace4c51ee7f0"
)

POST_RTG_DRM_EXTENSION_FIELDS = frozenset({
    "asm_z_eta",
    "asm_z_lambda",
    "asm_z_metric_d_min",
    "asm_z_metric_d_max",
    "asm_z_metric_u_bound",
})

ConfigT = TypeVar("ConfigT", DRMConfig, TinyTransformerConfig)


def _config_path(kind: str, seed: int) -> Path:
    if kind == "asm":
        return Path(f"configs/rtg_asm_30k_seed{seed}.yaml")
    if kind == "transformer":
        return Path(f"transformer/rtg_transformer_30k_seed{seed}.yaml")
    raise ValueError(f"unknown ATTR-RTG backbone: {kind}")


def registered_config_paths() -> tuple[Path, ...]:
    """Return all ten registered YAML paths in stable arm/seed order."""
    return tuple(
        _config_path(kind, seed)
        for kind in ("asm", "transformer")
        for seed in TRAINING_SEEDS
    )


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_preregistration(root: str | Path) -> str:
    """Fail closed unless the frozen manifest and every registered artifact match."""
    root = Path(root)
    manifest_path = root / PREREGISTRATION_MANIFEST
    digest = file_sha256(manifest_path)
    if digest != PREREGISTRATION_MANIFEST_SHA256:
        raise ValueError("ATTR-RTG preregistration manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN PREREGISTRATION":
        raise ValueError("ATTR-RTG preregistration is not frozen")
    if tuple(manifest.get("scope", {}).get("seeds", ())) != TRAINING_SEEDS:
        raise ValueError("ATTR-RTG registered seeds differ")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise TypeError("ATTR-RTG artifact manifest is malformed")
    expected_paths = {Path("docs/ATTR_RTG_PREREGISTRATION.md"), *registered_config_paths()}
    actual_paths = {Path(record.get("path", "")) for record in records}
    if actual_paths != expected_paths or len(records) != len(expected_paths):
        raise ValueError("ATTR-RTG artifact set differs from preregistration")
    for record in records:
        path = root / record["path"]
        if path.stat().st_size != record.get("bytes"):
            raise ValueError(f"ATTR-RTG artifact size mismatch: {record['path']}")
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"ATTR-RTG artifact hash mismatch: {record['path']}")
    return digest


def load_registered_config(root: str | Path, kind: str, seed: int) -> ConfigT:
    """Load one complete YAML with literal dataclass fields and registered seed."""
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unregistered ATTR-RTG seed: {seed}")
    config_type = DRMConfig if kind == "asm" else TinyTransformerConfig
    path = Path(root) / _config_path(kind, seed)
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"ATTR-RTG config is not a mapping: {path}")
    field_names = {field.name for field in fields(config_type)}
    expected_fields = (
        field_names - POST_RTG_DRM_EXTENSION_FIELDS
        if kind == "asm"
        else field_names
    )
    if set(values) != expected_fields:
        missing = sorted(expected_fields - set(values))
        extra = sorted(set(values) - expected_fields)
        raise ValueError(f"ATTR-RTG config fields differ; missing={missing}, extra={extra}")
    if values.get("seed") != seed:
        raise ValueError("ATTR-RTG config seed differs from filename")
    config = config_type(**values)
    if kind == "asm":
        return config.validated_copy()
    if kind != "transformer":
        raise ValueError(f"unknown ATTR-RTG backbone: {kind}")
    return config
