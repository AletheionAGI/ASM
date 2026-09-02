"""Complete configuration loading and fail-closed validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ARMS = ("R", "CM", "Z", "T")
TRAINING_SEEDS = (29, 43, 71, 89, 107)


@dataclass(frozen=True)
class ModelConfig:
    protocol: str
    status: str
    arm: str
    training_seed: int
    context_length: int
    vocab_size: int
    d_token: int
    d_state: int
    hidden_size: int
    n_directions: int
    metric_rank: int
    transformer_heads: int
    transformer_layers: int
    transformer_ffn: int
    common_state_dim: int
    candidate_count: int
    candidate_width: int
    target_trainable_parameters: int
    readout_hidden1: int
    readout_hidden2: int
    common_prelu: bool
    common_layernorm: bool
    synthetic_only: bool
    official_operations_allowed: bool
    native_export: bool
    common24_export: bool
    z_eta: float
    z_lambda: float
    z_metric_d_min: float
    z_metric_d_max: float
    z_metric_u_bound: float
    z_solves_per_input: int
    z_updates_per_input: int

    def validate(self) -> ModelConfig:
        if (
            self.protocol != "ATTR-RTG-RCMZ-V1"
            or self.status != "DRAFT V1 — LOCAL-ONLY — NOT LOCKED"
        ):
            raise ValueError("protocol identity/status differs")
        if self.arm not in ARMS or self.training_seed not in TRAINING_SEEDS:
            raise ValueError("unregistered arm or seed")
        if (
            self.context_length,
            self.candidate_count,
            self.candidate_width,
            self.common_state_dim,
        ) != (256, 6, 4, 24):
            raise ValueError("fixed context/candidate/common24 dimensions differ")
        if not self.synthetic_only or self.official_operations_allowed:
            raise ValueError("only synthetic pre-lock operations are allowed")
        if not self.native_export or not self.common24_export:
            raise ValueError("both registered state exports are required")
        if (
            min(
                self.d_token,
                self.d_state,
                self.hidden_size,
                self.target_trainable_parameters,
            )
            <= 0
        ):
            raise ValueError("model dimensions and budget must be positive")
        if self.arm == "Z" and (self.z_solves_per_input, self.z_updates_per_input) != (
            1,
            1,
        ):
            raise ValueError("ASM-Z requires exactly one solve and update per input")
        if not (
            0 < self.z_metric_d_min <= self.z_metric_d_max
            and self.z_eta > 0
            and self.z_lambda >= 0
        ):
            raise ValueError("invalid ASM-Z geometry constants")
        return self


def load_config(path: str | Path) -> ModelConfig:
    values: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("model config must be a mapping")
    expected = set(ModelConfig.__dataclass_fields__)
    if set(values) != expected:
        raise ValueError(
            f"config fields differ; missing={sorted(expected - set(values))}, extra={sorted(set(values) - expected)}"
        )
    return ModelConfig(**values).validate()


def registered_config_paths(root: str | Path) -> tuple[Path, ...]:
    base = Path(root) / "configs" / "attr_rtg_rcmz_v1"
    return tuple(
        base / f"{arm.lower()}_seed{seed}.yaml"
        for arm in ARMS
        for seed in TRAINING_SEEDS
    )
