"""Named constructor for the strict ASM-Z zero-choice recurrence."""

from __future__ import annotations

from drm_language_emitter.asm_z import asm_z_config
from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel


def zero_choice_config(base: DRMConfig, *, eta: float | None = None) -> DRMConfig:
    return asm_z_config(base, eta=eta)


def build_zero_choice(base: DRMConfig, *, eta: float | None = None) -> DRMEmitterModel:
    return DRMEmitterModel(zero_choice_config(base, eta=eta))
