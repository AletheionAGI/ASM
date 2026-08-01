"""ASM-F: directions orthonormalized in the learned relational metric."""

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel

from ._config import configured


def build_metric_frame(base: DRMConfig) -> DRMEmitterModel:
    config = configured(
        base,
        use_drm_geometry=True,
        use_direction_field=True,
        use_relational_metric=True,
        use_metric_naturalization=True,
        directional_metric_composition="metric_orthonormal",
    )
    return DRMEmitterModel(config)


__all__ = ["build_metric_frame"]
