"""Small configuration helpers used by named ASM constructors."""

from __future__ import annotations

from typing import Any

from drm_language_emitter.config import DRMConfig


def configured(base: DRMConfig, **overrides: Any) -> DRMConfig:
    data = base.to_dict()
    data.update(overrides)
    return DRMConfig.from_dict(data)


def block_scan_overrides(base: DRMConfig) -> dict[str, Any]:
    """Return a checkpoint-compatible block mode for direct transitions."""

    if base.sequence_mode in {
        "directional_cumsum",
        "directional_block_cumsum",
        "directional_superblock_cumsum",
    }:
        return {}
    return {"sequence_mode": "directional_block_cumsum"}
