"""ASM-C: checkpoint-compatible ASM-R with bounded streaming inference state."""

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel

from ._config import block_scan_overrides, configured


def build_compact_streaming(base: DRMConfig) -> DRMEmitterModel:
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
    )
    return DRMEmitterModel(config)


__all__ = ["build_compact_streaming"]
