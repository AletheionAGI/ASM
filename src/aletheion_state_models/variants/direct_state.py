"""ASM-D: direct contextual state transition without relational geometry."""

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel

from ._config import configured


def build_direct_state(base: DRMConfig) -> DRMEmitterModel:
    config = configured(
        base,
        use_drm_geometry=True,
        use_direction_field=False,
        use_relational_metric=False,
        use_metric_naturalization=False,
        directional_cumsum_step_mode="velocity",
        directional_metric_composition="post_naturalize",
    )
    return DRMEmitterModel(config)


__all__ = ["build_direct_state"]
