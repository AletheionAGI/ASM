"""ASM-C2: compact ASM-R with fixed-capacity content-addressable memory."""

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel

from ._config import block_scan_overrides, configured


def build_compact_addressable(
    base: DRMConfig,
    *,
    slots: int = 32,
    read_enabled: bool = True,
    write_enabled: bool = True,
) -> DRMEmitterModel:
    config = configured(
        base,
        **block_scan_overrides(base),
        use_drm_geometry=True,
        use_direction_field=False,
        use_relational_metric=True,
        use_metric_naturalization=True,
        directional_cumsum_step_mode="velocity",
        directional_metric_composition="post_naturalize",
        compact_streaming_inference=True,
        addressable_memory=True,
        addressable_memory_slots=slots,
        addressable_memory_read_enabled=read_enabled,
        addressable_memory_write_enabled=write_enabled,
    )
    return DRMEmitterModel(config)


def build_compact_addressable_sparse(
    base: DRMConfig,
    *,
    slots: int = 32,
    read_top_k: int = 2,
    write_top_k: int = 1,
) -> DRMEmitterModel:
    config = configured(
        base,
        **block_scan_overrides(base),
        use_drm_geometry=True,
        use_direction_field=False,
        use_relational_metric=True,
        use_metric_naturalization=True,
        directional_cumsum_step_mode="velocity",
        directional_metric_composition="post_naturalize",
        compact_streaming_inference=True,
        addressable_memory=True,
        addressable_memory_slots=slots,
        addressable_memory_temperature=0.25,
        addressable_memory_read_top_k=read_top_k,
        addressable_memory_write_top_k=write_top_k,
        addressable_memory_use_previous_token_key=True,
        lambda_addressable_read_entropy=0.001,
        lambda_addressable_write_entropy=0.001,
    )
    return DRMEmitterModel(config)


def build_compact_fast_weight(
    base: DRMConfig,
    *,
    read_enabled: bool = True,
    write_enabled: bool = True,
    shuffle_on_eval: bool = False,
) -> DRMEmitterModel:
    """Build ASM-C2-FW with bounded delta-rule associative memory."""
    config = configured(
        base,
        **block_scan_overrides(base),
        use_drm_geometry=True,
        use_direction_field=False,
        use_relational_metric=True,
        use_metric_naturalization=True,
        directional_cumsum_step_mode="velocity",
        directional_metric_composition="post_naturalize",
        compact_streaming_inference=True,
        addressable_memory=True,
        addressable_memory_backend="fast_weight",
        addressable_memory_read_enabled=read_enabled,
        addressable_memory_write_enabled=write_enabled,
        addressable_memory_shuffle_on_eval=shuffle_on_eval,
    )
    return DRMEmitterModel(config)


def build_compact_durable_fast_weight(
    base: DRMConfig,
    *,
    read_enabled: bool = True,
    write_enabled: bool = True,
    shuffle_on_eval: bool = False,
) -> DRMEmitterModel:
    """Build durable ASM-C2-FW with selective fast/slow consolidation."""
    model = build_compact_fast_weight(
        base,
        read_enabled=read_enabled,
        write_enabled=write_enabled,
        shuffle_on_eval=shuffle_on_eval,
    )
    config = model.config
    config.fast_weight_durable_memory = True
    config.fast_weight_state_fp32 = True
    config.fast_weight_compute_fp32 = True
    config.fast_weight_hard_write_threshold = 0.5
    return DRMEmitterModel(config.validated_copy())


def build_compact_epistemic_memory(
    base: DRMConfig,
    *,
    read_enabled: bool = True,
    write_enabled: bool = True,
    shuffle_on_eval: bool = False,
) -> DRMEmitterModel:
    """Build ASM-CM-E: durable ASM-CM with epistemic read/write abstention."""
    model = build_compact_durable_fast_weight(
        base,
        read_enabled=read_enabled,
        write_enabled=write_enabled,
        shuffle_on_eval=shuffle_on_eval,
    )
    config = model.config
    config.epistemic_memory_gating = True
    return DRMEmitterModel(config.validated_copy())


__all__ = [
    "build_compact_addressable",
    "build_compact_addressable_sparse",
    "build_compact_fast_weight",
    "build_compact_durable_fast_weight",
    "build_compact_epistemic_memory",
]
