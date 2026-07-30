from __future__ import annotations

import torch
from torch import nn

from .config import DRMConfig


class DRMStateInitializer(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.z0 = nn.Parameter(torch.zeros(config.d_state))
        nn.init.normal_(self.z0, std=0.02)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.z0.unsqueeze(0).expand(batch_size, -1).to(device)


class CausalLocalMixer(nn.Module):
    """Cheap local causal state correction for testing the Anderson-as-mixer hypothesis."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        hidden = config.directional_local_mixer_hidden_size
        in_dim = config.d_state * 3 + config.d_token + 2
        self.input_proj = nn.Linear(in_dim, hidden)
        self.depthwise = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden,
                    hidden,
                    kernel_size=config.directional_local_mixer_kernel_size,
                    groups=hidden,
                    bias=True,
                )
                for _ in range(config.directional_local_mixer_layers)
            ]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel_size=1, bias=True) for _ in range(config.directional_local_mixer_layers)]
        )
        self.output_proj = nn.Linear(hidden, config.d_state)

    def forward(
        self,
        z_start: torch.Tensor,
        states: torch.Tensor,
        token_embeddings: torch.Tensor,
        local_delta: torch.Tensor,
        dim: torch.Tensor,
        risk_mass: torch.Tensor,
    ) -> torch.Tensor:
        previous = torch.cat([z_start.unsqueeze(1), states[:, :-1]], dim=1)
        residual = states - previous
        features = torch.cat(
            [
                states,
                residual,
                local_delta,
                token_embeddings,
                dim.unsqueeze(-1),
                risk_mass.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = torch.nn.functional.silu(self.input_proj(features)).transpose(1, 2)
        kernel = int(self.config.directional_local_mixer_kernel_size)
        for depthwise, pointwise in zip(self.depthwise, self.pointwise):
            padded = torch.nn.functional.pad(hidden, (kernel - 1, 0))
            mixed = pointwise(torch.nn.functional.silu(depthwise(padded)))
            hidden = hidden + mixed
        correction = self.output_proj(hidden.transpose(1, 2))
        return states + float(self.config.directional_local_mixer_scale) * correction
