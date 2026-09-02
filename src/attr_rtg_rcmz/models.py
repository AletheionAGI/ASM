"""Public model/config factory for ATTR-RTG-RCMZ-V1."""

from __future__ import annotations

from pathlib import Path

from .adapter import RiskAdapter
from .config import ModelConfig, load_config


def build_adapter(config: ModelConfig | str | Path) -> RiskAdapter:
    if not isinstance(config, ModelConfig):
        config = load_config(config)
    if config.arm == "R":
        from .r import build
    elif config.arm == "CM":
        from .cm import build
    elif config.arm == "Z":
        from .z import build
    elif config.arm == "T":
        from .t import build
    else:  # validated configs make this unreachable
        raise ValueError(f"unknown arm: {config.arm}")
    return build(config)
