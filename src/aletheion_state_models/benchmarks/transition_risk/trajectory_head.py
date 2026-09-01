"""Shared autoregressive physical-trajectory head for ATTR-TG1."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .trajectory_types import TARGET_CARDINALITIES

FIELD_CARDINALITIES = dict(TARGET_CARDINALITIES)
TRAJECTORY_STEPS = 8


def inverse_cdf_sample(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """Sample categorical logits with caller-owned common random numbers."""
    expected = logits.shape[:-1]
    if uniforms.shape != expected:
        raise ValueError(f"uniforms must have shape {expected}, got {uniforms.shape}")
    if (
        not torch.isfinite(uniforms).all()
        or (uniforms < 0).any()
        or (uniforms > 1).any()
    ):
        raise ValueError("uniforms must be finite values in [0, 1]")
    cdf = logits.softmax(dim=-1).cumsum(dim=-1)
    return (cdf < uniforms.unsqueeze(-1)).sum(dim=-1).clamp_max(logits.shape[-1] - 1)


class TrajectoryHead(nn.Module):
    """Predict only physical state distributions for an eight-step rollout."""

    def __init__(
        self,
        input_dim: int = 72,
        hidden_dim: int = 32,
        action_count: int = 7,
        action_dim: int = 16,
    ) -> None:
        super().__init__()
        if input_dim != 72 or hidden_dim != 32:
            raise ValueError("ATTR-TG1 fixes input_dim=72 and hidden_dim=32")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.trap_classifier = nn.Linear(hidden_dim, 3 * 81)
        self.action_embedding = nn.Embedding(action_count, action_dim)
        self.feedback_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size, hidden_dim)
                for name, size in FIELD_CARDINALITIES.items()
            }
        )
        self.decoder = nn.GRUCell(action_dim + hidden_dim, hidden_dim)
        self.classifiers = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, size)
                for name, size in FIELD_CARDINALITIES.items()
            }
        )

    @staticmethod
    def _check_inputs(context: torch.Tensor, plan_actions: torch.Tensor) -> None:
        if context.ndim != 3 or context.shape[-1] != 72:
            raise ValueError("context must have shape [B, T, 72]")
        if plan_actions.shape != (*context.shape[:2], TRAJECTORY_STEPS):
            raise ValueError("plan_actions must have shape [B, T, 8]")

    def _feedback(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        embedded = [
            self.feedback_embeddings[name](values[name].long())
            for name in FIELD_CARDINALITIES
        ]
        return torch.stack(embedded).mean(dim=0)

    def forward(
        self,
        context: torch.Tensor,
        plan_actions: torch.Tensor,
        targets: Mapping[str, torch.Tensor] | None = None,
        *,
        teacher_forcing: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return trap and future-state logits; training defaults to teacher forcing."""
        self._check_inputs(context, plan_actions)
        teacher_forcing = (
            self.training and targets is not None
            if teacher_forcing is None
            else teacher_forcing
        )
        if teacher_forcing and targets is None:
            raise ValueError("targets are required for teacher forcing")
        hidden = self.context(context)
        trap_logits = self.trap_classifier(hidden).reshape(*hidden.shape[:2], 3, 81)
        feedback = hidden.new_zeros(hidden.shape)
        outputs: dict[str, list[torch.Tensor]] = {
            name: [] for name in FIELD_CARDINALITIES
        }
        for step in range(TRAJECTORY_STEPS):
            action = self.action_embedding(plan_actions[..., step].long())
            hidden = self.decoder(
                torch.cat((action, feedback), dim=-1).flatten(0, 1),
                hidden.flatten(0, 1),
            )
            hidden = hidden.reshape(*context.shape[:2], self.hidden_dim)
            step_logits = {
                name: layer(hidden) for name, layer in self.classifiers.items()
            }
            for name, value in step_logits.items():
                outputs[name].append(value)
            values = (
                {name: targets[name][..., step] for name in FIELD_CARDINALITIES}
                if teacher_forcing
                else {name: value.argmax(dim=-1) for name, value in step_logits.items()}
            )
            feedback = self._feedback(values)
        return {"trap_cells": trap_logits} | {
            name: torch.stack(values, dim=-2) for name, values in outputs.items()
        }

    def sample(
        self,
        context: torch.Tensor,
        plan_actions: torch.Tensor,
        uniforms: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Free-run using externally supplied uniforms for paired rollouts."""
        self._check_inputs(context, plan_actions)
        missing = ({"trap_cells"} | set(FIELD_CARDINALITIES)) - set(uniforms)
        if missing:
            raise ValueError(f"missing uniforms for: {sorted(missing)}")
        hidden = self.context(context)
        trap_logits = self.trap_classifier(hidden).reshape(*hidden.shape[:2], 3, 81)
        samples = {
            "trap_cells": inverse_cdf_sample(trap_logits, uniforms["trap_cells"])
        }
        feedback = hidden.new_zeros(hidden.shape)
        field_samples: dict[str, list[torch.Tensor]] = {
            name: [] for name in FIELD_CARDINALITIES
        }
        for step in range(TRAJECTORY_STEPS):
            action = self.action_embedding(plan_actions[..., step].long())
            hidden = self.decoder(
                torch.cat((action, feedback), dim=-1).flatten(0, 1),
                hidden.flatten(0, 1),
            )
            hidden = hidden.reshape(*context.shape[:2], self.hidden_dim)
            current = {}
            for name, layer in self.classifiers.items():
                value = inverse_cdf_sample(layer(hidden), uniforms[name][..., step])
                current[name] = value
                field_samples[name].append(value)
            feedback = self._feedback(current)
        samples.update(
            {
                name: torch.stack(values, dim=-1)
                for name, values in field_samples.items()
            }
        )
        return samples


__all__ = [
    "FIELD_CARDINALITIES",
    "TRAJECTORY_STEPS",
    "TrajectoryHead",
    "inverse_cdf_sample",
]
