"""Full-rank projected ASM-RS variant for Phase 3A.3."""
import torch
from aletheion_state_models.variants import (
    build_variable_rank_phase3a1,
    relational_selective_state_config,
)
from .phase3a_variants import phase3a_config


def build_phase3a3_variant(variant: str, seed: int):
    if variant != "vr_rs_full": raise ValueError(f"unknown Phase 3A.3 variant: {variant}")
    config = relational_selective_state_config(phase3a_config(seed))
    config = type(config).from_dict(config.to_dict() | {
        "lambda_variable_rank_budget": 0.0,
        "lambda_variable_rank_binary": 0.0,
        "lambda_variable_rank_switch": 0.0,
    })
    model = build_variable_rank_phase3a1(
        config, mixer=True, residual=True, selective_memory=True,
        relational_core=True,
    )
    controller = model.variable_rank_core.controller
    with torch.no_grad():
        controller.score_head.weight.zero_(); controller.score_head.bias.fill_(20.0)
    for parameter in controller.parameters(): parameter.requires_grad_(False)
    return model, 64


__all__ = ["build_phase3a3_variant"]
