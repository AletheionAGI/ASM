"""ASM-R adapter constructor."""

from .adapter import RiskAdapter
from .backbones import build_r
from .config import ModelConfig


def build(config: ModelConfig) -> RiskAdapter:
    if config.arm != "R":
        raise ValueError("R adapter requires arm R")
    return RiskAdapter(config, build_r(config))
