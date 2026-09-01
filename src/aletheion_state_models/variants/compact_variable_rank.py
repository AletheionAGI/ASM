"""ASM-CM-VR fixed and adaptive durable associative memory builders."""
from __future__ import annotations
import math
import torch
from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from ._config import block_scan_overrides, configured


def _rank_aware_config(base: DRMConfig, target_rank: int, *, adaptive: bool) -> DRMConfig:
    if not 1 <= target_rank <= base.d_state:
        raise ValueError("target rank must lie in [1, d_state]")
    options = dict(
        use_drm_geometry=True, use_direction_field=False,
        use_relational_metric=True, use_metric_naturalization=True,
        directional_cumsum_step_mode="velocity",
        directional_metric_composition="post_naturalize",
        compact_streaming_inference=True, addressable_memory=True,
        addressable_memory_backend="fast_weight",
        addressable_memory_value_dim=base.d_state,
        variable_rank_memory_policy="project_io",
        fast_weight_durable_memory=True, fast_weight_state_fp32=True,
        fast_weight_compute_fp32=True, fast_weight_hard_write_threshold=0.5,
        variable_rank_mode="phase3a1_projected",
        variable_rank_scaffold_projection=True,
        variable_rank_target_fraction=target_rank / base.d_state,
    )
    if not adaptive:
        options.update(lambda_variable_rank_budget=0.0,
                       lambda_variable_rank_binary=0.0,
                       lambda_variable_rank_switch=0.0)
    return configured(base, **block_scan_overrides(base), **options)


def build_compact_memory_variable_rank(
    base: DRMConfig, *, fixed_rank: int = 32
) -> DRMEmitterModel:
    """Build strict ASM-CM-VR with one frozen prefix rank."""
    model = DRMEmitterModel(_rank_aware_config(base, fixed_rank, adaptive=False))
    controller = model.variable_rank_core.controller
    with torch.no_grad():
        controller.score_head.weight.zero_(); controller.score_head.bias.fill_(-20.0)
        controller.score_head.bias[:fixed_rank].fill_(20.0)
    for parameter in controller.parameters(): parameter.requires_grad_(False)
    return model


def build_compact_memory_adaptive_rank(
    base: DRMConfig, *, target_rank: int = 32
) -> DRMEmitterModel:
    """Build exploratory input-only adaptive ASM-CM-VR."""
    model = DRMEmitterModel(_rank_aware_config(base, target_rank, adaptive=True))
    controller = model.variable_rank_core.controller
    initial_bias = math.log(
        model.config.variable_rank_open_probability
        / (1.0 - model.config.variable_rank_open_probability)
    )
    with torch.no_grad():
        controller.score_head.weight.zero_(); controller.score_head.bias.fill_(initial_bias)
    return model


__all__ = ["build_compact_memory_adaptive_rank", "build_compact_memory_variable_rank"]
