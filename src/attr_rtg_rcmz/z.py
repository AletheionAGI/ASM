"""Strict zero-choice ASM-Z adapter and structural audit."""

from drm_language_emitter.asm_z_core import ASMZCore

from .adapter import RiskAdapter
from .backbones import build_z
from .config import ModelConfig


def audit_strict_z(adapter: RiskAdapter) -> None:
    config = adapter.config
    model = adapter.backbone
    forbidden = ("attention", "candidate", "trust", "gate", "side_write", "bypass")
    names = tuple(name.lower() for name, _ in model.named_modules())
    if (
        config.arm != "Z"
        or config.z_solves_per_input != 1
        or config.z_updates_per_input != 1
    ):
        raise ValueError("ASM-Z solve/update cardinality differs")
    if model.config.n_flow_steps != 1 or model.config.sequence_mode != "asm_z":
        raise ValueError("ASM-Z recurrence is not one strict step")
    if sum(isinstance(module, ASMZCore) for module in model.modules()) != 1:
        raise ValueError("ASM-Z must contain exactly one strict core")
    if any(term in name for name in names for term in forbidden):
        raise ValueError("ASM-Z contains forbidden internal mechanism")
    if model.config.addressable_memory or model.config.selective_memory:
        raise ValueError("ASM-Z contains memory/gating bypass")


def build(config: ModelConfig) -> RiskAdapter:
    if config.arm != "Z":
        raise ValueError("Z adapter requires arm Z")
    adapter = RiskAdapter(config, build_z(config))
    audit_strict_z(adapter)
    return adapter
