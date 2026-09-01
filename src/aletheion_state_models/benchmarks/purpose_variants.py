"""Parameter-matched ASM-CM and ASM-VR-S purpose-comparison variants."""
import torch
from drm_language_emitter import DRMConfig
from aletheion_state_models.variants import (
    build_compact_durable_fast_weight,
    build_variable_rank_phase3a1,
    selective_state_config,
)
from .phase3a_variants import phase3a_config

PURPOSE_VARIANTS = ("asm_cm", "asm_vr_s_full", "asm_vr_s_fixed_32")
VR_S_MEMORY_HIDDEN_SIZE = 465


def _selective_base(seed: int) -> DRMConfig:
    config = selective_state_config(
        phase3a_config(seed), memory_hidden_size=VR_S_MEMORY_HIDDEN_SIZE
    )
    return DRMConfig.from_dict(
        config.to_dict() | {"compact_streaming_inference": True}
    )


def build_purpose_variant(variant: str, seed: int):
    """Build one matched-purpose comparison arm."""
    if variant not in PURPOSE_VARIANTS:
        raise ValueError(f"unknown purpose-comparison variant: {variant}")
    if variant == "asm_cm":
        return build_compact_durable_fast_weight(phase3a_config(seed)), 64
    config = _selective_base(seed)
    config = DRMConfig.from_dict(
        config.to_dict()
        | {
            "lambda_variable_rank_budget": 0.0,
            "lambda_variable_rank_binary": 0.0,
            "lambda_variable_rank_switch": 0.0,
        }
    )
    model = build_variable_rank_phase3a1(
        config, mixer=True, residual=True, selective_memory=True,
        relational_core=False,
    )
    rank = 64 if variant == "asm_vr_s_full" else 32
    controller = model.variable_rank_core.controller
    with torch.no_grad():
        controller.score_head.weight.zero_(); controller.score_head.bias.fill_(-20.0)
        controller.score_head.bias[:rank].fill_(20.0)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return model, rank


def parameter_inventory(seed: int = 17):
    output = {}
    for variant in PURPOSE_VARIANTS:
        model, rank = build_purpose_variant(variant, seed)
        output[variant] = {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "logical_rank": rank,
        }
    return output


__all__ = ["PURPOSE_VARIANTS", "VR_S_MEMORY_HIDDEN_SIZE", "build_purpose_variant", "parameter_inventory"]
