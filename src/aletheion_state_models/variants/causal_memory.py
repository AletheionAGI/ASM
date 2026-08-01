"""ASM-M: geometry-free causal mixer and selective-memory control."""

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel

from ._config import block_scan_overrides, configured


def build_causal_memory(base: DRMConfig) -> DRMEmitterModel:
    config = configured(
        base,
        **block_scan_overrides(base),
        use_drm_geometry=False,
        use_direction_field=False,
        use_relational_metric=False,
        use_metric_naturalization=False,
        directional_cumsum_step_mode="velocity",
        directional_metric_composition="post_naturalize",
        selective_memory=True,
        instantiate_disabled_risk=False,
    )
    return DRMEmitterModel(config)


__all__ = ["build_causal_memory"]
