"""Projected-scaffold variants for ASM-VR Phase 3A.1."""
from __future__ import annotations
from functools import partial
import torch
from aletheion_state_models.variants import build_variable_rank_phase3a1
from drm_language_emitter import DRMConfig
from .phase3a_variants import phase3a_config

STAGE_A_COMPONENTS = {
    "strict": (False, False, False),
    "mixer": (True, False, False),
    "residual": (False, True, False),
    "selective": (False, False, True),
    "mixer_residual": (True, True, False),
    "mixer_selective": (True, False, True),
    "residual_selective": (False, True, True),
    "all_projected": (True, True, True),
}
STAGE_A_VARIANTS = tuple(STAGE_A_COMPONENTS)
STAGE_B_VARIANTS = (
    "selected_full",
    "selected_fixed_16",
    "selected_fixed_32",
    "selected_fixed_48",
    "selected_adaptive_32",
)


def _without_rank_losses(config: DRMConfig) -> DRMConfig:
    return DRMConfig.from_dict(config.to_dict() | {
        "lambda_variable_rank_budget": 0.0,
        "lambda_variable_rank_binary": 0.0,
        "lambda_variable_rank_switch": 0.0,
    })


def _fix_rank(model, rank: int) -> None:
    controller = model.variable_rank_core.controller
    with torch.no_grad():
        controller.score_head.weight.zero_()
        controller.score_head.bias.fill_(-20.0)
        controller.score_head.bias[:rank].fill_(20.0)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)


def _build(config: DRMConfig, components: tuple[bool, bool, bool]):
    mixer, residual, selective = components
    return build_variable_rank_phase3a1(
        config,
        mixer=mixer,
        residual=residual,
        selective_memory=selective,
    )


def build_stage_a_variant(variant: str, seed: int):
    """Build one full-rank factorial projected scaffold."""
    if variant not in STAGE_A_COMPONENTS:
        raise ValueError(f"unknown Phase 3A.1-A variant: {variant}")
    model = _build(_without_rank_losses(phase3a_config(seed)), STAGE_A_COMPONENTS[variant])
    _fix_rank(model, 64)
    return model, 64


def build_stage_b_variant(
    variant: str,
    seed: int,
    *,
    components: tuple[bool, bool, bool],
):
    """Build one rank arm over the frozen winning projected scaffold."""
    if variant not in STAGE_B_VARIANTS:
        raise ValueError(f"unknown Phase 3A.1-B variant: {variant}")
    config = phase3a_config(seed)
    if variant == "selected_adaptive_32":
        config = DRMConfig.from_dict(config.to_dict() | {"variable_rank_threshold": 0.5})
    else:
        config = _without_rank_losses(config)
    model = _build(config, components)
    fixed_rank = {
        "selected_full": 64,
        "selected_fixed_16": 16,
        "selected_fixed_32": 32,
        "selected_fixed_48": 48,
    }.get(variant)
    if fixed_rank is not None:
        _fix_rank(model, fixed_rank)
    return model, fixed_rank


def stage_b_builder(components: tuple[bool, bool, bool]):
    """Bind the selected scaffold for the generic Phase 3A trainer."""
    return partial(build_stage_b_variant, components=components)


__all__ = [
    "STAGE_A_COMPONENTS",
    "STAGE_A_VARIANTS",
    "STAGE_B_VARIANTS",
    "build_stage_a_variant",
    "build_stage_b_variant",
    "stage_b_builder",
]
