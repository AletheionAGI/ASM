"""ASM-R: direct contextual transition conditioned by relational geometry."""

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel

from ._config import configured


def build_relational_state(base: DRMConfig) -> DRMEmitterModel:
    config = configured(
        base,
        use_drm_geometry=True,
        use_direction_field=False,
        use_relational_metric=True,
        use_metric_naturalization=True,
        directional_cumsum_step_mode="velocity",
        directional_metric_composition="post_naturalize",
    )
    return DRMEmitterModel(config)


__all__ = ["build_relational_state"]
