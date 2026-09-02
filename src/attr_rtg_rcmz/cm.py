"""ASM-CM adapter constructor using the generic compact fast-weight model."""

from .adapter import RiskAdapter
from .backbones import build_cm
from .config import ModelConfig


def build(config: ModelConfig) -> RiskAdapter:
    if config.arm != "CM":
        raise ValueError("CM adapter requires arm CM")
    return RiskAdapter(config, build_cm(config))
