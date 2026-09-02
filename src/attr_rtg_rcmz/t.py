"""Transformer adapter constructor using the generic tiny Transformer."""

from .adapter import RiskAdapter
from .backbones import build_t
from .config import ModelConfig


def build(config: ModelConfig) -> RiskAdapter:
    if config.arm != "T":
        raise ValueError("T adapter requires arm T")
    return RiskAdapter(config, build_t(config))
