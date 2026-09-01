"""Backbone adapters and registered factories for ATTR-TG1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from drm_language_emitter import DRMConfig, DRMEmitterModel
from transformer.tiny_transformer import TinyTransformerConfig, TinyTransformerLM

from .dataset import gather_step_representations
from .model_adapters import ASMModelAdapter, ModelAdapter, TransformerModelAdapter
from .trajectory_head import TrajectoryHead
from .trajectory_types import TRAINING_SEEDS

TG1_ARMS = ("asm_x_base", "transformer_base")
TG1_TRAINING_SEEDS = TRAINING_SEEDS


class TrajectoryModel(nn.Module):
    """Compose a causal backbone, a non-learned 72-wide bridge, and the common head."""

    def __init__(self, adapter: ModelAdapter, head: TrajectoryHead) -> None:
        super().__init__()
        self.adapter = adapter
        if adapter.representation_dim not in {64, 72}:
            raise ValueError("ATTR-TG1 supports only 64- or 72-wide backbones")
        self.head = head

    @property
    def model(self) -> nn.Module:
        return self.adapter.model

    def encode_steps(
        self, input_ids: torch.Tensor, step_positions: torch.Tensor
    ) -> torch.Tensor:
        states = self.adapter(input_ids)
        steps = gather_step_representations(states, step_positions)
        return F.pad(steps, (0, 8)) if steps.shape[-1] == 64 else steps

    def forward(
        self,
        input_ids: torch.Tensor,
        step_positions: torch.Tensor,
        plan_actions: torch.Tensor,
        targets: dict[str, torch.Tensor] | None = None,
        *,
        teacher_forcing: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        context = self.encode_steps(input_ids, step_positions)
        return self.head(
            context, plan_actions, targets, teacher_forcing=teacher_forcing
        )

    def sample(
        self,
        input_ids: torch.Tensor,
        step_positions: torch.Tensor,
        plan_actions: torch.Tensor,
        uniforms: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return self.head.sample(
            self.encode_steps(input_ids, step_positions), plan_actions, uniforms
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_trajectory_arm(
    root: str | Path,
    arm: str,
    seed: int,
    updates: int = 0,
) -> tuple[TrajectoryModel, dict[str, Any]]:
    """Build a registered arm with identical common-head initialization."""
    root = Path(root)
    torch.manual_seed(seed)
    if arm == "asm_x_base":
        values = _read_yaml(root / "configs/tiny_drm_stronger.yaml")
        config = DRMConfig.from_dict(
            values
            | {"sequence_mode": "directional_candidates", "use_powerlaw_risk": False}
        )
        backbone = DRMEmitterModel(config)
        adapter: ModelAdapter = ASMModelAdapter(backbone, global_step=updates)
        canonical_arm = "asm_x_base"
        risk_enabled = bool(config.use_powerlaw_risk)
    elif arm == "transformer_base":
        config = TinyTransformerConfig.from_dict(
            _read_yaml(root / "transformer/tiny_transformer_220k.yaml")
        )
        backbone = TinyTransformerLM(config)
        adapter = TransformerModelAdapter(backbone)
        canonical_arm = "transformer_base"
        risk_enabled = False
    else:
        raise ValueError(f"unknown ATTR-TG1 arm: {arm}")
    # Re-seeding makes the shared head independent of backbone construction.
    torch.manual_seed(seed + 50_000)
    head = TrajectoryHead(input_dim=72, hidden_dim=32)
    trajectory_model = TrajectoryModel(adapter, head)
    metadata = {
        "arm": canonical_arm,
        "protocol": "ATTR-TG1",
        "trajectory_steps": 8,
        "head_input_dim": 72,
        "hidden_dim": 32,
        "risk_enabled": risk_enabled,
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "backbone_parameters": sum(
            parameter.numel() for parameter in backbone.parameters()
        ),
    }
    return trajectory_model, metadata


def build_tg1_arm(root: str | Path, arm: str, seed: int, updates: int = 0):
    """Short factory alias used by experiment code."""
    return build_trajectory_arm(root, arm, seed, updates)


def verify_common_head_initialization(
    root: str | Path, seed: int = 29, updates: int = 0
) -> dict[str, Any]:
    """Fail if registered arms do not receive the exact same head tensors."""
    built = [build_trajectory_arm(root, arm, seed, updates) for arm in TG1_ARMS]
    states = [item[0].head.state_dict() for item in built]
    if states[0].keys() != states[1].keys() or any(
        not torch.equal(states[0][name], states[1][name]) for name in states[0]
    ):
        raise ValueError("ATTR-TG1 common heads differ between arms")
    counts = [
        sum(parameter.numel() for parameter in item[0].head.parameters())
        for item in built
    ]
    if counts[0] != counts[1]:
        raise ValueError("ATTR-TG1 common-head parameter counts differ")
    return {
        "identical_initial_tensors": True,
        "head_parameters": counts[0],
        "arms": TG1_ARMS,
    }


__all__ = [
    "TG1_ARMS",
    "TG1_TRAINING_SEEDS",
    "TrajectoryModel",
    "build_tg1_arm",
    "build_trajectory_arm",
    "verify_common_head_initialization",
]
