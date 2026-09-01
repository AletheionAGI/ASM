"""Matched ASM-VR-R and ASM-VR-S variants for Phase 3A.2."""
from __future__ import annotations
import torch
from drm_language_emitter import DRMConfig
from aletheion_state_models.variants import (
    build_variable_rank_phase3a1,
    selective_state_config,
)
from .phase3a_variants import phase3a_config

BASES = ("vr_r", "vr_s")
RANK_ARMS = ("full", "fixed_16", "fixed_32", "fixed_48", "adaptive_32")
PHASE3A2_VARIANTS = tuple(f"{base}_{arm}" for base in BASES for arm in RANK_ARMS)
ASM_S_MEMORY_HIDDEN_SIZE = 308


def _without_rank_losses(config: DRMConfig) -> DRMConfig:
    return DRMConfig.from_dict(config.to_dict() | {
        "lambda_variable_rank_budget": 0.0,
        "lambda_variable_rank_binary": 0.0,
        "lambda_variable_rank_switch": 0.0,
    })


def _fix_rank(model, rank: int) -> None:
    controller = model.variable_rank_core.controller
    with torch.no_grad():
        controller.score_head.weight.zero_(); controller.score_head.bias.fill_(-20.0)
        controller.score_head.bias[:rank].fill_(20.0)
    for parameter in controller.parameters(): parameter.requires_grad_(False)


def build_phase3a2_variant(variant: str, seed: int):
    """Build one rank arm on either the relational or selective core."""
    if variant not in PHASE3A2_VARIANTS: raise ValueError(f"unknown Phase 3A.2 variant: {variant}")
    base_name = next(base for base in BASES if variant.startswith(f"{base}_"))
    arm = variant.removeprefix(f"{base_name}_"); config = phase3a_config(seed)
    if base_name == "vr_s":
        config = selective_state_config(
            config,
            memory_hidden_size=ASM_S_MEMORY_HIDDEN_SIZE,
        )
    if arm == "adaptive_32":
        config = DRMConfig.from_dict(config.to_dict() | {"variable_rank_threshold": 0.5})
    else:
        config = _without_rank_losses(config)
    model = build_variable_rank_phase3a1(
        config,
        mixer=True,
        residual=True,
        selective_memory=(base_name == "vr_s"),
        relational_core=(base_name == "vr_r"),
    )
    fixed_rank = {"full": 64, "fixed_16": 16, "fixed_32": 32, "fixed_48": 48}.get(arm)
    if fixed_rank is not None: _fix_rank(model, fixed_rank)
    return model, fixed_rank


__all__ = ["ASM_S_MEMORY_HIDDEN_SIZE", "BASES", "PHASE3A2_VARIANTS", "RANK_ARMS", "build_phase3a2_variant"]
