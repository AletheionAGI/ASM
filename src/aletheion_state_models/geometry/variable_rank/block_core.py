"""Block-boundary projection adapter for integrated ASM-VR Phase 1."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .batch_state import VariableRankBatchState
from .rank_controller import HardRankObservation, InputHardRankController


class VariableRankBlockCore(nn.Module):
    """Apply per-example hard collapse around one ASM-R block.

    Phase 1 uses the identity as a fixed global frame. This keeps ASM-R state
    coordinates canonical and avoids an O(D^3) QR factorization at model scale.
    """

    def __init__(
        self,
        token_dimension: int,
        state_dimension: int,
        *,
        threshold: float,
        minimum_rank: int,
        estimator: str = "hard",
    ) -> None:
        super().__init__()
        self.state_dimension = state_dimension
        self.controller = InputHardRankController(
            token_dimension,
            state_dimension,
            threshold=threshold,
            minimum_rank=minimum_rank,
            estimator=estimator,
        )

    def observe_block(
        self,
        token_embeddings: Tensor,
        block_index: int = 0,
        *,
        temperature: float = 1.0,
    ) -> HardRankObservation:
        """Choose one causal mask from the first token of each nonempty block."""
        if token_embeddings.ndim != 3 or token_embeddings.shape[1] < 1:
            raise ValueError("token_embeddings must contain a nonempty block")
        if block_index < 0:
            raise ValueError("block_index must be non-negative")
        return self.controller(token_embeddings[:, 0], temperature=temperature)

    def project(self, values: Tensor, active_mask: Tensor) -> Tensor:
        """Remove inactive identity-frame coordinates from states or sequences."""
        if values.shape[0] != active_mask.shape[0]:
            raise ValueError("values and active_mask must share the batch dimension")
        if values.shape[-1] != self.state_dimension:
            raise ValueError("values must end with state_dimension")
        if active_mask.shape != (values.shape[0], self.state_dimension):
            raise ValueError("active_mask must have shape [batch, state_dimension]")
        expanded_mask = active_mask
        while expanded_mask.ndim < values.ndim:
            expanded_mask = expanded_mask.unsqueeze(1)
        return values * expanded_mask.to(dtype=values.dtype)

    def begin_block(self, state: Tensor, observation: HardRankObservation) -> Tensor:
        """Collapse the carried state before ASM-R computes any token forcing."""
        return self.project(state, observation.mask_for_projection())

    def finish_block(self, states: Tensor, observation: HardRankObservation) -> Tensor:
        """Project all emission states after the ASM-R block computation."""
        return self.project(states, observation.mask_for_projection())

    def cache_state(self, state: Tensor, active_mask: Tensor) -> VariableRankBatchState:
        """Create the only latent payload allowed in a Phase 1 inference cache."""
        return VariableRankBatchState(self.project(state, active_mask), active_mask)


__all__ = ["VariableRankBlockCore"]
