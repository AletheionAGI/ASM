"""ASM-S: direct state transition with capacity allocated to selective memory."""

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from ._config import block_scan_overrides, configured


def selective_state_config(
    base: DRMConfig,
    *,
    memory_hidden_size: int | None = None,
) -> DRMConfig:
    """Return the canonical historical ASM-S configuration."""
    return configured(
        base,
        **block_scan_overrides(base),
        use_drm_geometry=True,
        use_direction_field=False,
        use_relational_metric=False,
        use_metric_naturalization=False,
        directional_cumsum_step_mode="velocity",
        directional_metric_composition="post_naturalize",
        selective_memory=True,
        selective_memory_hidden_size=(
            memory_hidden_size
            if memory_hidden_size is not None
            else base.selective_memory_hidden_size
        ),
    )


def build_selective_state(
    base: DRMConfig,
    *,
    memory_hidden_size: int | None = None,
) -> DRMEmitterModel:
    """Build the canonical ASM-S model."""
    return DRMEmitterModel(
        selective_state_config(base, memory_hidden_size=memory_hidden_size)
    )


__all__ = ["build_selective_state", "selective_state_config"]
