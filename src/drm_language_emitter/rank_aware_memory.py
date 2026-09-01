"""Rank-aware adapter for bounded addressable memory."""
from __future__ import annotations
from typing import Any
import torch


def apply_rank_aware_memory(
    memory_module,
    states: torch.Tensor,
    token_embeddings: torch.Tensor,
    memory_state,
    *,
    variable_rank_core=None,
    rank_observation=None,
) -> tuple[torch.Tensor, Any, dict[str, torch.Tensor]]:
    """Project memory writes and reads under one block-rank observation.

    Tokens remain causal external input. State coordinates are projected before
    the memory can inspect or store them, and memory reads are projected before
    they can re-enter the recurrent state.
    """
    if (variable_rank_core is None) != (rank_observation is None):
        raise ValueError("rank core and observation must be provided together")
    memory_inputs = states
    if variable_rank_core is not None:
        memory_inputs = variable_rank_core.finish_block(states, rank_observation)
    memory_options = {}
    if rank_observation is not None:
        memory_options["value_mask"] = rank_observation.mask_for_projection()
    outputs, next_memory, diagnostics = memory_module.forward_sequence(
        memory_inputs, token_embeddings, memory_state, **memory_options
    )
    if variable_rank_core is not None:
        outputs = variable_rank_core.finish_block(outputs, rank_observation)
    return outputs, next_memory, diagnostics


__all__ = ["apply_rank_aware_memory"]
