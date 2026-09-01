"""ASM-X risk-mass variant for the post-hoc ATTR P2 diagnostic."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from drm_language_emitter import DRMConfig, DRMEmitterModel

from .model_adapters import ASMModelAdapter
from .model_heads import TransitionRiskHeads

RISK_MASS_ARM = "asm_x_directional_risk_mass"
BASELINE_ARM = "asm_x_directional"


def _configuration(root: Path, *, enabled: bool) -> DRMConfig:
    with (root / "configs/tiny_drm_stronger.yaml").open() as handle:
        values = yaml.safe_load(handle)
    return DRMConfig.from_dict(
        values
        | {
            "sequence_mode": "directional_candidates",
            "use_powerlaw_risk": enabled,
        }
    )


def build_risk_mass_arm(root: str | Path, seed: int, updates: int):
    """Build ASM-X with the sole config delta use_powerlaw_risk=True."""
    root = Path(root)
    torch.manual_seed(seed)
    config = _configuration(root, enabled=True)
    model = DRMEmitterModel(config)
    adapter = ASMModelAdapter(model, global_step=updates)
    torch.manual_seed(seed + 50_000)
    heads = TransitionRiskHeads(config.d_state, 6, hidden_dim=32)
    metadata = {
        "comparison_role": "posthoc_risk_mass_diagnostic",
        "logical_rank": None,
        "explicit_associative_memory": False,
        "risk_mass_enabled": True,
        "config_delta": {"use_powerlaw_risk": {"baseline": False, "variant": True}},
        "backbone_trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    return adapter, heads, metadata


def verify_parameter_and_initialization_parity(
    root: str | Path, seed: int = 29
) -> dict:
    """Verify that enabling risk mass changes no tensor shape or initialization."""
    root = Path(root)
    torch.manual_seed(seed)
    baseline = DRMEmitterModel(_configuration(root, enabled=False))
    torch.manual_seed(seed)
    variant = DRMEmitterModel(_configuration(root, enabled=True))
    baseline_state = baseline.state_dict()
    variant_state = variant.state_dict()
    if baseline_state.keys() != variant_state.keys():
        raise ValueError("risk-mass variant changed state-dict keys")
    if any(
        not torch.equal(baseline_state[name], variant_state[name])
        for name in baseline_state
    ):
        raise ValueError("risk-mass variant changed initial tensors")
    return {
        "baseline_parameters": sum(
            parameter.numel() for parameter in baseline.parameters()
        ),
        "variant_parameters": sum(
            parameter.numel() for parameter in variant.parameters()
        ),
        "identical_initial_tensors": True,
        "only_config_delta": "use_powerlaw_risk: false -> true",
    }


__all__ = [
    "BASELINE_ARM",
    "RISK_MASS_ARM",
    "build_risk_mass_arm",
    "verify_parameter_and_initialization_parity",
]
