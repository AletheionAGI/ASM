from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .addressable_memory import AddressableMemoryState
from .fast_weight_memory import FastWeightMemoryState
from .rank_aware_memory import apply_rank_aware_memory

MemoryState = AddressableMemoryState | FastWeightMemoryState

if TYPE_CHECKING:
    from aletheion_state_models.geometry.variable_rank.batch_state import VariableRankBatchState


@dataclass(frozen=True)
class InferenceState:
    """Causal inference state with an optional completed-block cache.

    ``input_ids`` is retained for checkpoint/debug compatibility and as a fallback
    for modes without fixed block boundaries. Fixed-block variants additionally
    retain the state before the open block and only recompute that bounded block.
    """

    input_ids: torch.Tensor
    completed_state: torch.Tensor | None = None
    block_tokens: torch.Tensor | None = None
    block_index: int = 0
    block_size: int = 0
    tokens_seen: int = 0
    compact: bool = False
    addressable_memory: MemoryState | None = None
    variable_rank_state: VariableRankBatchState | None = None

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return self.tokens_seen if self.compact else int(self.input_ids.shape[1])

    @property
    def uses_block_cache(self) -> bool:
        return (
            self.block_size > 0
            and (self.completed_state is not None or self.variable_rank_state is not None)
            and self.block_tokens is not None
        )


class InferenceMixin:
    def init_inference_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> InferenceState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return InferenceState(
            torch.empty(batch_size, 0, dtype=torch.long, device=device),
            compact=bool(self.config.compact_streaming_inference),
        )

    def _inference_block_size(self) -> int:
        if self.config.sequence_mode == "directional_block_cumsum":
            return max(int(self.config.directional_cumsum_block_size), 0)
        if self.config.sequence_mode == "directional_superblock_cumsum":
            return max(
                int(
                    self.config.directional_superblock_size
                    or self.config.directional_cumsum_block_size
                ),
                0,
            )
        return 0

    def _inference_state_from_forward(
        self,
        prefix: torch.Tensor,
        states: torch.Tensor,
        final_memory: MemoryState | None = None,
        memory_before_last_block: MemoryState | None = None,
        variable_rank_masks: torch.Tensor | None = None,
    ) -> InferenceState:
        block_size = self._inference_block_size()
        compact = bool(self.config.compact_streaming_inference)
        if compact and (block_size <= 0 or not self.config.use_drm_geometry):
            raise RuntimeError(
                "compact streaming requires DRM geometry with fixed block boundaries"
            )
        if block_size <= 0 or not self.config.use_drm_geometry:
            return InferenceState(prefix)
        block_start = (prefix.shape[1] // block_size) * block_size
        if block_start == 0:
            completed_state = self.initializer(prefix.shape[0], prefix.device)
        else:
            completed_state = states[:, block_start - 1]
        completed_memory = (
            final_memory
            if prefix.shape[1] % block_size == 0
            else memory_before_last_block
        )
        variable_rank_state = None
        if self.variable_rank_core is not None:
            if variable_rank_masks is None:
                raise RuntimeError("ASM-VR forward did not return hard rank masks")
            if prefix.shape[1] % block_size == 0:
                cache_source = states[:, -1]
                cache_mask = variable_rank_masks[:, -1]
            else:
                cache_source = completed_state
                cache_mask = variable_rank_masks[:, block_start]
            variable_rank_state = self.variable_rank_core.cache_state(
                cache_source, cache_mask
            ).detach()
            completed_state = None
        return InferenceState(
            input_ids=(prefix[:, :0].detach() if compact else prefix),
            completed_state=(completed_state.detach() if completed_state is not None else None),
            block_tokens=prefix[:, block_start:].detach(),
            block_index=block_start // block_size,
            block_size=block_size,
            tokens_seen=int(prefix.shape[1]),
            compact=compact,
            addressable_memory=(completed_memory.detach() if completed_memory is not None else None),
            variable_rank_state=variable_rank_state,
        )

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        state: InferenceState | None = None,
    ) -> tuple[torch.Tensor, InferenceState]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError(
                "input_ids must have shape [batch, sequence] with sequence > 0"
            )
        if state is None:
            prefix = input_ids
        else:
            if state.batch_size != input_ids.shape[0]:
                raise ValueError(
                    "inference state batch size does not match input_ids"
                )
            if state.compact and state.tokens_seen:
                raise RuntimeError(
                    "compact streaming state cannot be extended with prefill; use decode_step"
                )
            prefix = torch.cat([state.input_ids, input_ids], dim=1)
        output = self(
            prefix,
            return_states=True,
            collect_diagnostics=False,
        )
        return output["logits"], self._inference_state_from_forward(
            prefix,
            output["states"],
            output.get("addressable_final_memory"),
            output.get("addressable_before_last_block"),
            output.get("variable_rank_masks"),
        )

    @torch.no_grad()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        state: InferenceState,
    ) -> tuple[torch.Tensor, InferenceState]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step expects one token per batch item")
        if state.batch_size != input_ids.shape[0]:
            raise ValueError("inference state batch size does not match input_ids")
        if not state.uses_block_cache:
            if self.variable_rank_core is not None:
                raise RuntimeError("ASM-VR decode requires its compact block cache")
            logits, next_state = self.prefill(input_ids, state)
            return logits[:, -1], next_state

        open_tokens = torch.cat([state.block_tokens, input_ids], dim=1)
        token_embeddings = self.token_embedding(open_tokens)
        rank_observation = None
        block_start_state = state.completed_state
        if self.variable_rank_core is not None:
            if state.variable_rank_state is None:
                raise RuntimeError("ASM-VR cache is missing its effective rank state")
            rank_observation = self.variable_rank_core.observe_block(
                token_embeddings, state.block_index
            )
            if state.block_tokens.shape[1] and not torch.equal(
                rank_observation.active_mask, state.variable_rank_state.active_mask
            ):
                raise RuntimeError("ASM-VR open-block mask changed during replay")
            block_start_state = self.variable_rank_core.begin_block(
                state.variable_rank_state.effective_coordinates,
                rank_observation,
            )
        assert block_start_state is not None
        observed_block_length = token_embeddings.shape[1]
        block_input = token_embeddings
        if self.config.variable_rank_scaffold_projection:
            block_size = self.config.directional_cumsum_block_size
            if observed_block_length < block_size:
                block_input = torch.nn.functional.pad(
                    token_embeddings,
                    (0, 0, 0, block_size - observed_block_length),
                )
        block_output = self._directional_cumsum_block(
            block_start_state,
            block_input,
            global_step=None,
            block_index=state.block_index,
            collect_diagnostics=False,
            projection_mask=(
                rank_observation.mask_for_projection()
                if rank_observation is not None
                and self.config.variable_rank_scaffold_projection
                else None
            ),
        )
        block_states = block_output[0][:, :observed_block_length]
        if rank_observation is not None:
            block_states = self.variable_rank_core.finish_block(
                block_states, rank_observation
            )
        next_memory = state.addressable_memory
        if self.addressable_memory is not None:
            if next_memory is None:
                next_memory = self.addressable_memory.initial_state(
                    input_ids.shape[0], input_ids.device, block_states.dtype
                )
            block_states, recomputed_memory, _ = apply_rank_aware_memory(
                self.addressable_memory,
                block_states,
                token_embeddings,
                next_memory,
                variable_rank_core=self.variable_rank_core,
                rank_observation=rank_observation,
            )
        if state.compact:
            prefix = state.input_ids
            next_logits = self.emitter(block_states[:, -1:])[:, -1]
        else:
            prefix = torch.cat([state.input_ids, input_ids], dim=1)
            # Compatibility path: preserve the full BF16 GEMM shape.
            emitter_states = block_states.new_zeros(
                block_states.shape[0],
                prefix.shape[1],
                block_states.shape[-1],
            )
            emitter_states[:, -block_states.shape[1] :] = block_states
            next_logits = self.emitter(emitter_states)[:, -1]
        next_variable_rank_state = state.variable_rank_state
        if open_tokens.shape[1] == state.block_size:
            completed_state = block_states[:, -1].detach()
            if rank_observation is not None:
                next_variable_rank_state = self.variable_rank_core.cache_state(
                    completed_state, rank_observation.active_mask
                ).detach()
                completed_state = None
            block_tokens = open_tokens[:, :0].detach()
            block_index = state.block_index + 1
            next_memory = recomputed_memory.detach() if self.addressable_memory is not None else next_memory
        else:
            completed_state = state.completed_state
            if rank_observation is not None and state.block_tokens.shape[1] == 0:
                next_variable_rank_state = self.variable_rank_core.cache_state(
                    block_start_state, rank_observation.active_mask
                ).detach()
            block_tokens = open_tokens.detach()
            block_index = state.block_index
        next_state = InferenceState(
            input_ids=prefix,
            completed_state=completed_state,
            block_tokens=block_tokens,
            block_index=block_index,
            block_size=state.block_size,
            tokens_seen=state.sequence_length + 1,
            compact=state.compact,
            addressable_memory=next_memory,
            variable_rank_state=next_variable_rank_state,
        )
        return next_logits, next_state
