"""Small configuration helpers used by named ASM constructors."""

from __future__ import annotations

from typing import Any

from drm_language_emitter.config import DRMConfig


def configured(base: DRMConfig, **overrides: Any) -> DRMConfig:
    data = base.to_dict()
    data.update(overrides)
    return DRMConfig.from_dict(data)
