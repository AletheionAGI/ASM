from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch
import yaml

from aletheion_state_models.variants import build_zero_choice
from drm_language_emitter import DRMConfig
from transformer.tiny_transformer import TinyTransformerConfig, TinyTransformerLM

SEEDS = (31, 47, 73, 97, 113)
PARAMETERS = 17_024
ROOT = Path(__file__).parents[1]


def _values(path: Path) -> dict:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    return values


def _z(seed: int):
    values = _values(ROOT / f"configs/rtg_z_asm_z_17k_seed{seed}.yaml")
    assert set(values) == {field.name for field in fields(DRMConfig)}
    assert values["seed"] == seed
    return build_zero_choice(DRMConfig(**values).validated_copy())


def _t(seed: int):
    values = _values(ROOT / f"transformer/rtg_z_transformer_17k_seed{seed}.yaml")
    assert set(values) == {field.name for field in fields(TinyTransformerConfig)}
    assert values["seed"] == seed
    return TinyTransformerLM(TinyTransformerConfig(**values))


def test_rtg_z_candidate_configs_are_complete_and_exactly_parameter_matched() -> None:
    for seed in SEEDS:
        z_model, t_model = _z(seed), _t(seed)
        z_count = sum(parameter.numel() for parameter in z_model.parameters())
        t_count = sum(parameter.numel() for parameter in t_model.parameters())
        assert z_count == t_count == PARAMETERS
        assert z_model.config.max_seq_len == t_model.config.max_seq_len == 256


def test_rtg_z_seed31_all_parameters_are_active() -> None:
    for model in (_z(31), _t(31)):
        model.train()
        ids = torch.arange(130).reshape(2, 65) % model.config.vocab_size
        loss = model(ids, targets=(ids + 1) % model.config.vocab_size)["loss"]
        loss.backward()
        assert all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in model.parameters()
        )


def test_rtg_z_seed31_context_256_is_finite_and_causal() -> None:
    torch.manual_seed(20260902)
    for model in (_z(31), _t(31)):
        model.eval()
        prefix = torch.randint(0, model.config.vocab_size, (1, 65))
        suffix = torch.randint(0, model.config.vocab_size, (1, 7))
        with torch.no_grad():
            prefix_logits = model(prefix)["logits"]
            combined = model(torch.cat((prefix, suffix), dim=1))["logits"]
            full = model(torch.arange(256).reshape(1, 256) % model.config.vocab_size)["logits"]
        prefix_error = (prefix_logits - combined[:, :65]).abs().max()
        assert float(prefix_error) < 1e-6
        assert full.shape == (1, 256, 64)
        assert torch.isfinite(full).all()
