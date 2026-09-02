"""Construction and strict terminal-checkpoint loading for ATTR-RTG backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn

from drm_language_emitter import DRMConfig, DRMEmitterModel
from drm_language_emitter.emitter import RMSNorm
from transformer.tiny_transformer import TinyTransformerConfig, TinyTransformerLM

from .rtg_config import (
    BACKBONE_PARAMETER_COUNTS,
    load_registered_config,
    verify_preregistration,
)

BackboneKind = Literal["asm", "transformer"]

# These trainable ASM parameters are neither weights nor biases. DRMEmitterModel
# creates z0 from a config.seed-namespaced normal stream and the risk scalars from
# fixed constructor constants. The registered initializer deliberately preserves
# only this audited set; every matrix/vector affine parameter is reset below.
_PRESERVED_ASM_PARAMETERS = {
    "initializer.z0",
    "risk.alpha_b",
    "risk.alpha_d",
    "risk.beta_b",
    "risk.beta_d",
}
_PRESERVED_RISK_VALUES = {
    "risk.alpha_b": 0.1,
    "risk.alpha_d": 0.1,
    "risk.beta_b": 1.5,
    "risk.beta_d": 1.5,
}


def audit_preserved_asm_parameters(model: DRMEmitterModel) -> None:
    """Fail unless the only custom parameters have registered deterministic forms."""
    custom = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not (name.endswith(("weight", "bias")))
    }
    if set(custom) != _PRESERVED_ASM_PARAMETERS:
        raise ValueError(f"unregistered custom ASM parameters: {sorted(custom)}")
    z0 = custom["initializer.z0"]
    if z0.shape != (model.config.d_state,) or not torch.isfinite(z0).all():
        raise ValueError("ASM z0 is not the finite config-seeded state initializer")
    for name, expected in _PRESERVED_RISK_VALUES.items():
        value = custom[name]
        expected_tensor = torch.tensor(expected, dtype=value.dtype, device=value.device)
        if value.shape or not torch.equal(value.detach(), expected_tensor):
            raise ValueError(f"ASM custom scalar differs: {name}")


def initialize_registered_backbone(model: nn.Module, seed: int) -> None:
    """Apply the frozen initializer from one RNG sequence; preserve audited ASM state."""
    torch.manual_seed(seed)
    initialized: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            initialized.add(id(module.in_proj_weight))
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
                initialized.add(id(module.in_proj_bias))
            for parameter in (module.bias_k, module.bias_v):
                if parameter is not None:
                    nn.init.zeros_(parameter)
                    initialized.add(id(parameter))
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            initialized.add(id(module.weight))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                initialized.add(id(module.bias))
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
            initialized.add(id(module.weight))
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            initialized.add(id(module.weight))
            bias = getattr(module, "bias", None)
            if bias is not None:
                nn.init.zeros_(bias)
                initialized.add(id(bias))
    unhandled = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in initialized
    }
    if isinstance(model, DRMEmitterModel):
        audit_preserved_asm_parameters(model)
        unhandled -= _PRESERVED_ASM_PARAMETERS
    if unhandled:
        raise ValueError(f"registered initializer did not handle: {sorted(unhandled)}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_registered_backbone(
    root: str | Path,
    kind: BackboneKind,
    seed: int,
    *,
    verify_manifest: bool = True,
) -> nn.Module:
    """Build one frozen 30k backbone and verify its exact trainable budget."""
    if verify_manifest:
        verify_preregistration(root)
    torch.manual_seed(seed)
    config = load_registered_config(root, kind, seed)
    if kind == "asm":
        if not isinstance(config, DRMConfig):
            raise TypeError("ASM config type mismatch")
        model: nn.Module = DRMEmitterModel(config)
    else:
        if not isinstance(config, TinyTransformerConfig):
            raise TypeError("Transformer config type mismatch")
        model = TinyTransformerLM(config)
    initialize_registered_backbone(model, seed)
    actual = count_parameters(model)
    expected = BACKBONE_PARAMETER_COUNTS[kind]
    if actual != expected:
        raise ValueError(f"{kind} parameter budget differs: {actual} != {expected}")
    return model


def load_terminal_backbone(
    root: str | Path,
    kind: BackboneKind,
    seed: int,
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load a config-bearing terminal checkpoint strictly into a registered model."""
    model = build_registered_backbone(root, kind, seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    required = {
        "kind", "training_seed", "terminal_update", "metadata", "model", "config"
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("terminal checkpoint fields differ from the registered schema")
    expected_identity = (
        f"{kind}-backbone",
        seed,
        1_000,
        {"split": "train", "sealable": True, "protocol": "ATTR-RTG"},
    )
    actual_identity = (
        payload["kind"], payload["training_seed"], payload["terminal_update"],
        payload["metadata"],
    )
    expected_config = model.config.to_dict()
    if actual_identity != expected_identity or payload["config"] != expected_config:
        raise ValueError("terminal checkpoint identity/config differs from registration")
    if not isinstance(payload["model"], dict):
        raise TypeError("terminal checkpoint model state is malformed")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    return model
