from pathlib import Path

import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk.rtg_backbones import (
    audit_preserved_asm_parameters,
    build_registered_backbone,
    count_parameters,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_config import (
    load_registered_config,
)
from drm_language_emitter import DRMEmitterModel
from transformer.tiny_transformer import TinyTransformerLM

ROOT = Path(__file__).parents[1]


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_registered_backbone_budgets_and_seed_determinism():
    for kind, count in (("asm", 30_122), ("transformer", 30_120)):
        first = build_registered_backbone(ROOT, kind, 29, verify_manifest=False)
        second = build_registered_backbone(ROOT, kind, 29, verify_manifest=False)
        other = build_registered_backbone(ROOT, kind, 43, verify_manifest=False)
        assert count_parameters(first) == count
        assert count_parameters(other) == count
        assert all(torch.equal(value, _state(second)[name]) for name, value in _state(first).items())
        assert any(not torch.equal(value, _state(other)[name]) for name, value in _state(first).items())


def test_registered_initializer_zeroes_biases_and_sets_norm_scales():
    for kind in ("asm", "transformer"):
        model = build_registered_backbone(ROOT, kind, 29, verify_manifest=False)
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                assert torch.count_nonzero(parameter) == 0, name
        for module in model.modules():
            if isinstance(module, nn.LayerNorm) or type(module).__name__ == "RMSNorm":
                assert torch.equal(module.weight, torch.ones_like(module.weight))


def test_registered_xavier_matrices_replace_framework_defaults():
    asm_config = load_registered_config(ROOT, "asm", 29)
    raw_asm = DRMEmitterModel(asm_config)
    registered_asm = build_registered_backbone(ROOT, "asm", 29, verify_manifest=False)
    assert not torch.equal(raw_asm.token_embedding.embedding.weight, registered_asm.token_embedding.embedding.weight)
    raw_asm_linear = next(module for module in raw_asm.modules() if isinstance(module, nn.Linear))
    registered_asm_linear = next(
        module for module in registered_asm.modules() if isinstance(module, nn.Linear)
    )
    assert not torch.equal(raw_asm_linear.weight, registered_asm_linear.weight)

    transformer_config = load_registered_config(ROOT, "transformer", 29)
    torch.manual_seed(29)
    raw_transformer = TinyTransformerLM(transformer_config)
    registered_transformer = build_registered_backbone(ROOT, "transformer", 29, verify_manifest=False)
    assert not torch.equal(raw_transformer.token_embedding.weight, registered_transformer.token_embedding.weight)
    raw_attention = raw_transformer.encoder.layers[0].self_attn.in_proj_weight
    registered_attention = registered_transformer.encoder.layers[0].self_attn.in_proj_weight
    assert not torch.equal(raw_attention, registered_attention)


def test_custom_asm_parameters_are_config_seeded_or_fixed_scalars():
    first = build_registered_backbone(ROOT, "asm", 29, verify_manifest=False)
    second = build_registered_backbone(ROOT, "asm", 29, verify_manifest=False)
    other = build_registered_backbone(ROOT, "asm", 43, verify_manifest=False)
    audit_preserved_asm_parameters(first)
    assert torch.equal(first.initializer.z0, second.initializer.z0)
    assert not torch.equal(first.initializer.z0, other.initializer.z0)
    for name in ("alpha_b", "alpha_d", "beta_b", "beta_d"):
        assert torch.equal(getattr(first.risk, name), getattr(other.risk, name))
