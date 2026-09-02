"""Shared generic-backbone construction for the four registered arms."""

from __future__ import annotations

import torch
from torch import nn

from aletheion_state_models.variants import (
    build_compact_durable_fast_weight,
    build_relational_state,
    build_zero_choice,
)
from drm_language_emitter import DRMConfig
from transformer.tiny_transformer import TinyTransformerConfig, TinyTransformerLM

from .config import ModelConfig


def drm_base(config: ModelConfig) -> DRMConfig:
    return DRMConfig(
        vocab_size=config.vocab_size,
        d_token=config.d_token,
        d_state=config.d_state,
        n_directions=config.n_directions,
        metric_rank=config.metric_rank,
        hidden_size=config.hidden_size,
        n_flow_steps=1,
        max_seq_len=config.context_length,
        dropout=0.0,
        seed=config.training_seed,
        sequence_mode="local_step",
        instantiate_disabled_risk=False,
        emitter_layers=1,
        addressable_memory_dim=config.d_state,
        addressable_memory_value_dim=config.d_state,
        asm_z_eta=config.z_eta,
        asm_z_lambda=config.z_lambda,
        asm_z_metric_d_min=config.z_metric_d_min,
        asm_z_metric_d_max=config.z_metric_d_max,
        asm_z_metric_u_bound=config.z_metric_u_bound,
    )


def build_r(config: ModelConfig) -> nn.Module:
    return build_relational_state(drm_base(config))


def build_cm(config: ModelConfig) -> nn.Module:
    return build_compact_durable_fast_weight(drm_base(config))


def build_z(config: ModelConfig) -> nn.Module:
    return build_zero_choice(drm_base(config), eta=config.z_eta)


def build_t(config: ModelConfig) -> nn.Module:
    transformer = TinyTransformerConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_state,
        n_heads=config.transformer_heads,
        n_layers=config.transformer_layers,
        hidden_size=config.transformer_ffn,
        max_seq_len=config.context_length,
        dropout=0.0,
        seed=config.training_seed,
    )
    return TinyTransformerLM(transformer)


def initialize(model: nn.Module, seed: int) -> None:
    """Use a single deterministic CPU/CUDA-independent parameter stream."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            values = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
            values.normal_(0.0, 0.02, generator=generator)
            parameter.copy_(values.to(parameter.device))
