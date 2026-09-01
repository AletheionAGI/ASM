"""Strictly causal hard rank controller for ASM-VR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HardRankObservation:
    """Hard decision plus optional Phase 2 optimization tensors."""

    scores: Tensor
    active_mask: Tensor
    ranks: Tensor
    projection_mask: Tensor | None = None
    soft_gates: Tensor | None = None

    def mask_for_projection(self) -> Tensor:
        """Return hard-forward projection weights for the current mode."""
        return self.active_mask if self.projection_mask is None else self.projection_mask


class InputHardRankController(nn.Module):
    """Choose a per-example block mask from its first token only."""

    def __init__(
        self,
        input_dimension: int,
        frame_width: int,
        *,
        threshold: float = 0.5,
        minimum_rank: int = 1,
        estimator: str = "hard",
    ) -> None:
        super().__init__()
        if input_dimension < 1 or frame_width < 1:
            raise ValueError("input_dimension and frame_width must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if not 0 <= minimum_rank <= frame_width:
            raise ValueError("minimum_rank must lie in [0, frame_width]")
        if estimator not in {"hard", "ste"}:
            raise ValueError("estimator must be 'hard' or 'ste'")
        self.threshold = float(threshold)
        self.minimum_rank = minimum_rank
        self.estimator = estimator
        self.score_head = nn.Linear(input_dimension, frame_width)

    def forward(
        self, first_token: Tensor, *, temperature: float = 1.0
    ) -> HardRankObservation:
        """Return a hard decision and, in Phase 2, its STE surrogate."""
        if first_token.ndim != 2 or first_token.shape[-1] != self.score_head.in_features:
            raise ValueError("first_token must have shape [batch, input_dimension]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits = self.score_head(first_token)
        scores = torch.sigmoid(logits)
        mask = scores >= self.threshold
        ranks = mask.sum(dim=-1).clamp_min(self.minimum_rank)
        indices = torch.arange(mask.shape[-1], device=mask.device)
        mask = indices.unsqueeze(0) < ranks.unsqueeze(-1)
        projection_mask = None
        soft_gates = None
        if self.estimator == "ste":
            threshold = min(max(self.threshold, 1e-6), 1.0 - 1e-6)
            threshold_logit = torch.logit(logits.new_tensor(threshold))
            soft_gates = torch.sigmoid((logits - threshold_logit) / temperature)
            if self.training:
                hard = mask.to(dtype=soft_gates.dtype)
                projection_mask = hard + (soft_gates - soft_gates.detach())
        return HardRankObservation(
            scores=scores,
            active_mask=mask,
            ranks=ranks,
            projection_mask=projection_mask,
            soft_gates=soft_gates,
        )


__all__ = ["HardRankObservation", "InputHardRankController"]
