"""Load and validate sealed artifacts for registered ATTR-RTG evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .rtg_backbones import load_terminal_backbone
from .rtg_calibration import RtgCalibration
from .rtg_checkpoint import load_terminal_checkpoint
from .rtg_heads import DirectC, PhysicalD, TransitionG
from .rtg_normalization import StateNormalization
from .rtg_pipeline_seal import PipelineSealPaths
from .rtg_projection import load_projection
from .rtg_state_export import (
    ASMStateExporter,
    CausalStateExporter,
    TransformerReadoutExporter,
)

_METADATA = {"split": "train", "sealable": True, "protocol": "ATTR-RTG"}


@dataclass(frozen=True)
class RegisteredArm:
    """One complete sealed model arm and its preprocessing artifacts."""

    exporter: CausalStateExporter
    projection: torch.Tensor
    normalization: StateNormalization
    g: nn.Module
    d: nn.Module
    c: nn.Module
    g_calibration: RtgCalibration
    c_calibration: RtgCalibration


def _read_object(path: Path, expected: set[str], name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} artifact fields differ from registered schema")
    return value


def _load_normalization(path: Path) -> StateNormalization:
    names = ("pre_mean", "pre_std", "next_mean", "next_std")
    value = _read_object(path, set(names), "normalization")
    tensors = tuple(torch.tensor(value[name], dtype=torch.float32) for name in names)
    if any(tensor.shape != (28,) for tensor in tensors):
        raise ValueError("normalization artifact must contain four 28-vectors")
    return StateNormalization(*tensors)


def _load_calibration(path: Path) -> RtgCalibration:
    value = _read_object(path, {"temperature", "q95"}, "calibration")
    temperature, q95 = value["temperature"], value["q95"]
    if type(temperature) not in {int, float} or type(q95) not in {int, float}:
        raise TypeError("calibration values must be JSON numbers")
    return RtgCalibration(float(temperature), float(q95))


def _load_head(path: Path, module: nn.Module, kind: str, seed: int, name: str) -> nn.Module:
    return load_terminal_checkpoint(
        path, module, expected_kind=f"{kind}-{name}", expected_seed=seed,
        expected_metadata=_METADATA,
    )


def load_registered_arm(
    root: Path, paths: PipelineSealPaths, kind: str, seed: int,
) -> RegisteredArm:
    """Load one arm only from its exact sealed artifact locations."""
    arm = f"{kind}.seed{seed}"
    model = load_terminal_backbone(
        root, kind, seed, paths.resolve(paths.backbone_checkpoints[arm])
    )
    exporter: CausalStateExporter = (
        ASMStateExporter(model) if kind == "asm" else TransformerReadoutExporter(model)
    )
    def head(name: str, module: nn.Module) -> nn.Module:
        return _load_head(
            paths.resolve(paths.head_checkpoints[f"{arm}.{name}"]),
            module, kind, seed, name,
        )
    return RegisteredArm(
        exporter, load_projection(paths.resolve(paths.maps[f"projection_{kind}"]), kind),
        _load_normalization(paths.resolve(paths.statistics[f"{arm}.normalization"])),
        head("G", TransitionG()), head("D", PhysicalD()), head("C", DirectC()),
        _load_calibration(paths.resolve(paths.calibration[f"{arm}.calibration_G"])),
        _load_calibration(paths.resolve(paths.calibration[f"{arm}.calibration_C"])),
    )
