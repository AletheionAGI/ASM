"""Frozen model factories and parameter roles for ATTR P2."""

from __future__ import annotations
from pathlib import Path
import torch
import yaml
from drm_language_emitter import DRMConfig, DRMEmitterModel
from transformer.tiny_transformer import TinyTransformerConfig, TinyTransformerLM
from .model_adapters import ASMModelAdapter, TransformerModelAdapter
from .model_heads import TransitionRiskHeads
from .supplementary import _build_arm as build_supplementary_arm

MAIN_ARMS = ("asm_x_directional", "tiny_transformer_220k")
SUPPLEMENTARY_ARMS = (
    "asm_cm_durable",
    "asm_vr_s_full64",
    "asm_vr_s_fixed32",
    "asm_r_240k_control",
)
P2_ARMS = MAIN_ARMS + SUPPLEMENTARY_ARMS


def _yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def build_p2_arm(root: str | Path, arm: str, seed: int, updates: int):
    """Build one arm with a deterministic common-head initialization."""
    root = Path(root)
    if arm in SUPPLEMENTARY_ARMS:
        adapter, heads, metadata = build_supplementary_arm(arm, seed, updates)
        return adapter, heads, metadata
    torch.manual_seed(seed)
    if arm == "asm_x_directional":
        config = DRMConfig.from_dict(
            _yaml(root / "configs/tiny_drm_stronger.yaml")
            | {"sequence_mode": "directional_candidates"}
        )
        model = DRMEmitterModel(config)
        adapter = ASMModelAdapter(model, global_step=updates)
        dimension = config.d_state
    elif arm == "tiny_transformer_220k":
        config = TinyTransformerConfig.from_dict(
            _yaml(root / "transformer/tiny_transformer_220k.yaml")
        )
        model = TinyTransformerLM(config)
        adapter = TransformerModelAdapter(model)
        dimension = config.d_model
    else:
        raise ValueError(f"unknown ATTR P2 arm: {arm}")
    torch.manual_seed(seed + 50_000)
    heads = TransitionRiskHeads(dimension, 6, hidden_dim=32)
    metadata = {
        "comparison_role": "registered_main_pair",
        "logical_rank": None,
        "explicit_associative_memory": False,
        "backbone_trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
    return adapter, heads, metadata


def parameter_inventory(root: str | Path, seed: int = 29, updates: int = 1000) -> dict:
    output = {}
    for arm in P2_ARMS:
        adapter, heads, metadata = build_p2_arm(root, arm, seed, updates)
        total = sum(p.numel() for p in adapter.parameters()) + sum(
            p.numel() for p in heads.parameters()
        )
        trainable = sum(
            p.numel() for p in adapter.parameters() if p.requires_grad
        ) + sum(p.numel() for p in heads.parameters() if p.requires_grad)
        output[arm] = metadata | {
            "backbone_total": sum(p.numel() for p in adapter.model.parameters()),
            "head_total": sum(p.numel() for p in heads.parameters()),
            "total": total,
            "trainable": trainable,
        }
    return output


__all__ = [
    "MAIN_ARMS",
    "P2_ARMS",
    "SUPPLEMENTARY_ARMS",
    "build_p2_arm",
    "parameter_inventory",
]
