"""Paired language-model variants for ASM-VR Phase 3A."""

from __future__ import annotations

import torch

from aletheion_state_models.variants import (
    build_relational_state,
    build_variable_rank_phase2,
)
from drm_language_emitter import DRMConfig

PHASE3A_VARIANTS = (
    "asm_r",
    "vr_full",
    "vr_fixed_16",
    "vr_fixed_32",
    "vr_fixed_48",
    "vr_adaptive_32",
)


def phase3a_config(seed: int) -> DRMConfig:
    """Return the frozen small-scale language recipe."""
    return DRMConfig(
        vocab_size=256,
        d_token=64,
        d_state=64,
        n_directions=8,
        metric_rank=8,
        hidden_size=128,
        sequence_mode="directional_block_cumsum",
        directional_cumsum_step_mode="velocity",
        directional_cumsum_block_size=32,
        directional_local_mixer="causal_conv",
        directional_local_mixer_hidden_size=64,
        directional_local_mixer_kernel_size=4,
        token_state_residual=True,
        selective_memory=True,
        selective_memory_hidden_size=64,
        bounded_state=False,
        dropout=0.0,
        seed=seed,
        lambda_action=0.0,
        lambda_dim_sparsity=0.0,
        lambda_dim_entropy=0.0,
        lambda_dim_variance=0.0,
        lambda_metric_reg=0.0,
        lambda_metric_diversity=0.0,
        lambda_active_fraction=0.0,
        lambda_condition=0.0,
        lambda_metric_u_floor=0.0,
        variable_rank_min_rank=16,
        variable_rank_threshold=0.8,
        variable_rank_target_fraction=0.5,
        variable_rank_warmup_steps=50,
        variable_rank_budget_ramp_steps=150,
        variable_rank_hardening_steps=200,
        variable_rank_temperature_initial=2.0,
        variable_rank_temperature_final=0.5,
        lambda_variable_rank_budget=0.0025,
        lambda_variable_rank_binary=0.01,
        lambda_variable_rank_switch=0.001,
        variable_rank_open_probability=0.95,
    )


def _fix_rank(model, rank: int) -> None:
    controller = model.variable_rank_core.controller
    with torch.no_grad():
        controller.score_head.weight.zero_()
        controller.score_head.bias.fill_(-20.0)
        controller.score_head.bias[:rank].fill_(20.0)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)


def build_phase3a_variant(variant: str, seed: int):
    """Build one Phase 3A arm and return model plus resolved logical rank."""
    if variant not in PHASE3A_VARIANTS:
        raise ValueError(f"unknown Phase 3A variant: {variant}")
    config = phase3a_config(seed)
    if variant == "asm_r":
        return build_relational_state(config), 64
    if variant != "vr_adaptive_32":
        config = DRMConfig.from_dict(
            config.to_dict()
            | {
                "lambda_variable_rank_budget": 0.0,
                "lambda_variable_rank_binary": 0.0,
                "lambda_variable_rank_switch": 0.0,
            }
        )
    model = build_variable_rank_phase2(config)
    fixed_rank = {
        "vr_full": 64,
        "vr_fixed_16": 16,
        "vr_fixed_32": 32,
        "vr_fixed_48": 48,
    }.get(variant)
    if fixed_rank is not None:
        _fix_rank(model, fixed_rank)
    return model, fixed_rank


__all__ = ["PHASE3A_VARIANTS", "build_phase3a_variant", "phase3a_config"]
