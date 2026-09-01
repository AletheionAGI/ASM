"""Architecture-neutral prediction heads for the ATTR benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _projector(input_dim: int, output_dim: int, hidden_dim: int | None) -> nn.Module:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    if hidden_dim is None:
        return nn.Linear(input_dim, output_dim)
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class NextStateHead(nn.Module):
    """Predict a diagonal Gaussian distribution for the next observed state."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int | None = None,
        *,
        min_log_scale: float = -7.0,
        max_log_scale: float = 5.0,
    ) -> None:
        super().__init__()
        if min_log_scale >= max_log_scale:
            raise ValueError("min_log_scale must be less than max_log_scale")
        self.state_dim = state_dim
        self.min_log_scale = min_log_scale
        self.max_log_scale = max_log_scale
        self.projection = _projector(input_dim, state_dim * 2, hidden_dim)

    def forward(self, representations: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, log_scale = self.projection(representations).chunk(2, dim=-1)
        return {
            "mean": mean,
            "log_scale": log_scale.clamp(self.min_log_scale, self.max_log_scale),
        }


class HazardHead(nn.Module):
    """Predict independent hazard logits at registered forecast horizons."""

    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int] = (1, 4, 8, 16),
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        normalized = tuple(int(horizon) for horizon in horizons)
        if not normalized or any(horizon <= 0 for horizon in normalized):
            raise ValueError("horizons must contain positive integers")
        if len(set(normalized)) != len(normalized):
            raise ValueError("horizons must be unique")
        self.horizons = normalized
        self.projection = _projector(input_dim, len(normalized), hidden_dim)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return self.projection(representations)

    def probabilities(self, representations: torch.Tensor) -> torch.Tensor:
        return self(representations).sigmoid()


class SeverityHead(nn.Module):
    """Predict non-negative severity and time-to-hazard values."""

    def __init__(self, input_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.projection = _projector(input_dim, 2, hidden_dim)

    def forward(self, representations: torch.Tensor) -> dict[str, torch.Tensor]:
        severity, time_to_hazard = F.softplus(self.projection(representations)).unbind(
            dim=-1
        )
        return {"severity": severity, "time_to_hazard": time_to_hazard}


class TransitionRiskHeads(nn.Module):
    """Apply the same ATTR prediction surface to any adapted backbone."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        horizons: Sequence[int] = (1, 4, 8, 16),
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.next_state = NextStateHead(input_dim, state_dim, hidden_dim)
        self.hazard = HazardHead(input_dim, horizons, hidden_dim)
        self.severity = SeverityHead(input_dim, hidden_dim)

    def forward(self, representations: torch.Tensor) -> dict[str, Any]:
        return {
            "next_state": self.next_state(representations),
            "hazard_logits": self.hazard(representations),
            "severity": self.severity(representations),
        }


__all__ = ["HazardHead", "NextStateHead", "SeverityHead", "TransitionRiskHeads"]
