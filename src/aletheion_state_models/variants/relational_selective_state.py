"""ASM-RS: relational geometry with explicit selective state memory."""

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from ._config import block_scan_overrides, configured


def relational_selective_state_config(
    base: DRMConfig,
    *,
    memory_hidden_size: int | None = None,
) -> DRMConfig:
    """Return the canonical Relational Selective State Emitter config."""
    return configured(
        base,
        **block_scan_overrides(base),
        use_drm_geometry=True,
        use_direction_field=False,
        use_relational_metric=True,
        use_metric_naturalization=True,
        directional_cumsum_step_mode="velocity",
        directional_metric_composition="post_naturalize",
        selective_memory=True,
        selective_memory_hidden_size=(
            memory_hidden_size
            if memory_hidden_size is not None
            else base.selective_memory_hidden_size
        ),
    )


def build_relational_selective_state(
    base: DRMConfig,
    *,
    memory_hidden_size: int | None = None,
) -> DRMEmitterModel:
    """Build ASM-RS with no variable-rank controller."""
    config = relational_selective_state_config(
        base, memory_hidden_size=memory_hidden_size
    )
    return DRMEmitterModel(config)


__all__ = [
    "build_relational_selective_state",
    "relational_selective_state_config",
]
