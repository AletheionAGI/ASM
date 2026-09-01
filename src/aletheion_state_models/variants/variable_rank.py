"""ASM-VR builders over the ASM-R core."""

import math
import torch
from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from ._config import configured


def _variable_rank_config(
    base: DRMConfig,
    mode: str,
    *,
    mixer: bool = False,
    residual: bool = False,
    selective_memory: bool = False,
    relational_core: bool = True,
) -> DRMConfig:
    block_size = base.directional_cumsum_block_size or max(base.max_seq_len, 1)
    projected = mode == "phase3a1_projected"
    return configured(
        base,
        use_drm_geometry=True,
        use_direction_field=False,
        use_relational_metric=relational_core,
        use_metric_naturalization=relational_core,
        directional_cumsum_step_mode="velocity",
        directional_metric_composition="post_naturalize",
        sequence_mode="directional_block_cumsum",
        directional_cumsum_block_size=block_size,
        compact_streaming_inference=True,
        directional_local_mixer=("causal_conv" if mixer else "none"),
        token_state_residual=residual,
        selective_memory=selective_memory,
        addressable_memory=False,
        directional_refinement_layers=0,
        directional_endpoint_correction_weight=0.0,
        directional_fixed_point_iterations=0,
        directional_anderson_iterations=0,
        variable_rank_mode=mode,
        variable_rank_scaffold_projection=projected,
    )


def _initialize_open_controller(model: DRMEmitterModel) -> DRMEmitterModel:
    controller = model.variable_rank_core.controller
    probability = model.config.variable_rank_open_probability
    initial_bias = math.log(probability / (1.0 - probability))
    with torch.no_grad():
        controller.score_head.weight.zero_()
        controller.score_head.bias.fill_(initial_bias)
    return model


def build_variable_rank_phase1(base: DRMConfig) -> DRMEmitterModel:
    """Build the strict hard no-bypass Phase 1 variant."""
    return DRMEmitterModel(_variable_rank_config(base, "phase1_input_hard"))


def build_variable_rank_phase2(base: DRMConfig) -> DRMEmitterModel:
    """Build Phase 2 with hard-forward, soft-backward rank estimation."""
    model = DRMEmitterModel(_variable_rank_config(base, "phase2_input_ste"))
    return _initialize_open_controller(model)


def build_variable_rank_phase3a1(
    base: DRMConfig,
    *,
    mixer: bool = False,
    residual: bool = False,
    selective_memory: bool = False,
    relational_core: bool = True,
) -> DRMEmitterModel:
    """Build a projected ASM-R scaffold for the Phase 3A.1 ablations."""
    config = _variable_rank_config(
        base,
        "phase3a1_projected",
        mixer=mixer,
        residual=residual,
        selective_memory=selective_memory,
        relational_core=relational_core,
    )
    return _initialize_open_controller(DRMEmitterModel(config))


__all__ = [
    "build_variable_rank_phase1",
    "build_variable_rank_phase2",
    "build_variable_rank_phase3a1",
]
