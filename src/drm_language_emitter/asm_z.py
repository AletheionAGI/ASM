from __future__ import annotations

from dataclasses import replace

from .config import DRMConfig
from .model import DRMEmitterModel


def asm_z_config(base: DRMConfig, *, eta: float | None = None) -> DRMConfig:
    """Return a strict ASM-Z config without catalog-direction machinery."""
    return replace(
        base,
        sequence_mode="asm_z",
        asm_z_eta=float(base.asm_z_eta if eta is None else eta),
        bounded_state=False,
        use_drm_geometry=True,
        use_direction_field=False,
        use_relational_metric=False,
        use_metric_naturalization=False,
        compact_streaming_inference=True,
        dropout=0.0,
        addressable_memory=False,
        selective_memory=False,
        token_state_residual=False,
        directional_local_mixer="none",
        directional_refinement_layers=0,
    ).validated_copy()


def build_asm_z(base: DRMConfig, *, eta: float | None = None) -> DRMEmitterModel:
    return DRMEmitterModel(asm_z_config(base, eta=eta))
