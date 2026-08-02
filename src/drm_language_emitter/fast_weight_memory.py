"""Fixed-capacity causal fast-weight memory for ASM-C2-FW."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FastWeightMemoryState:
    matrix: torch.Tensor
    consolidated: torch.Tensor
    previous_token: torch.Tensor

    def detach(self) -> "FastWeightMemoryState":
        return FastWeightMemoryState(
            self.matrix.detach(), self.consolidated.detach(), self.previous_token.detach()
        )


class FastWeightMemory(nn.Module):
    """Causal delta-rule memory with bounded state independent of sequence length."""

    def __init__(self, config) -> None:
        super().__init__()
        self.d_key = int(config.addressable_memory_dim)
        self.d_value = int(config.addressable_memory_dim)
        self.d_token = int(config.d_token)
        self.d_state = int(config.d_state)
        self.read_scale = float(config.addressable_memory_read_scale)
        self.read_enabled = bool(config.addressable_memory_read_enabled)
        self.write_enabled = bool(config.addressable_memory_write_enabled)
        self.shuffle_on_eval = bool(config.addressable_memory_shuffle_on_eval)
        self.durable = bool(config.fast_weight_durable_memory)
        self.state_fp32 = bool(config.fast_weight_state_fp32)
        self.compute_fp32 = bool(config.fast_weight_compute_fp32)
        self.hard_write_threshold = float(config.fast_weight_hard_write_threshold)
        self.consolidation_scale = float(config.fast_weight_consolidation_scale)
        self.slow_read_scale = float(config.fast_weight_slow_read_scale)
        feature_dim = self.d_state + 2 * self.d_token
        # The same encoder defines stored and queried keys.
        self.key = nn.Linear(self.d_token, self.d_key, bias=False)
        self.value = nn.Linear(self.d_state + self.d_token, self.d_value)
        self.write_gate = nn.Linear(feature_dim, 1)
        self.read_gate = nn.Linear(feature_dim, 1)
        self.forget_gate = nn.Linear(feature_dim, 1)
        self.consolidate_gate = nn.Linear(feature_dim, 1)
        self.read_output = nn.Linear(self.d_value, self.d_state, bias=False)
        write_bias = 0.0 if self.durable else float(config.addressable_memory_write_bias)
        nn.init.constant_(self.write_gate.bias, write_bias)
        nn.init.constant_(self.forget_gate.bias, 4.0)
        nn.init.zeros_(self.read_output.weight)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> FastWeightMemoryState:
        memory_dtype = torch.float32 if self.state_fp32 else dtype
        matrix = torch.zeros(
            batch_size, self.d_key, self.d_value, device=device, dtype=memory_dtype
        )
        consolidated = torch.zeros_like(matrix)
        previous = torch.zeros(
            batch_size, self.d_token, device=device, dtype=memory_dtype
        )
        return FastWeightMemoryState(matrix, consolidated, previous)

    def _selective_gate(self, gate: torch.Tensor) -> torch.Tensor:
        if self.hard_write_threshold <= 0:
            return gate
        hard = (gate >= self.hard_write_threshold).to(gate.dtype)
        return hard if not self.training else hard + gate - gate.detach()

    def step(
        self,
        state: torch.Tensor,
        token_embedding: torch.Tensor,
        memory: FastWeightMemoryState,
    ) -> tuple[torch.Tensor, FastWeightMemoryState, dict[str, torch.Tensor]]:
        output_dtype = state.dtype
        if self.compute_fp32:
            # Fast-weight recurrence is numerically sensitive: execute projections,
            # gates, delta updates, and accumulators outside mixed precision. The
            # returned model state is restored to the caller dtype.
            with torch.autocast(device_type=state.device.type, enabled=False):
                return self._step_impl(
                    state.float(), token_embedding.float(), memory, output_dtype
                )
        return self._step_impl(state, token_embedding, memory, output_dtype)

    def _step_impl(
        self,
        state: torch.Tensor,
        token_embedding: torch.Tensor,
        memory: FastWeightMemoryState,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, FastWeightMemoryState, dict[str, torch.Tensor]]:
        previous_token = memory.previous_token.to(token_embedding.dtype)
        features = torch.cat([state, token_embedding, previous_token], dim=-1)
        query = F.normalize(
            self.key(token_embedding.to(self.key.weight.dtype)).float(), dim=-1
        )
        write_key = F.normalize(
            self.key(previous_token.to(self.key.weight.dtype)).float(), dim=-1
        )
        matrix = (
            torch.roll(memory.matrix, shifts=1, dims=1)
            if self.shuffle_on_eval and not self.training
            else memory.matrix
        )
        read = torch.einsum("bk,bkv->bv", query, matrix.float())
        if self.durable:
            read = read + self.slow_read_scale * torch.einsum(
                "bk,bkv->bv", query, memory.consolidated.float()
            )
        read_gate = (
            torch.sigmoid(self.read_gate(features))
            if self.read_enabled
            else state.new_zeros(state.shape[0], 1)
        )
        next_state = state + self.read_scale * read_gate * self.read_output(read.to(state.dtype))

        write_gate = (
            torch.sigmoid(self.write_gate(features))
            if self.write_enabled
            else state.new_zeros(state.shape[0], 1)
        )
        write_gate = self._selective_gate(write_gate)
        retention = torch.sigmoid(self.forget_gate(features)).float()
        candidate = torch.tanh(
            self.value(torch.cat([state, token_embedding], dim=-1))
        ).float()
        predicted = torch.einsum("bk,bkv->bv", write_key, memory.matrix.float())
        delta = candidate - predicted
        update = torch.einsum("bk,bv->bkv", write_key, delta)
        next_matrix = (
            retention.unsqueeze(-1) * memory.matrix.float()
            + write_gate.float().unsqueeze(-1) * update
        )
        consolidated = memory.consolidated.float()
        if self.durable and self.write_enabled:
            consolidation = self._selective_gate(
                torch.sigmoid(self.consolidate_gate(features))
            ).float()
            slow_prediction = torch.einsum("bk,bkv->bv", write_key, consolidated)
            slow_update = torch.einsum(
                "bk,bv->bkv", write_key, candidate - slow_prediction
            )
            consolidated = consolidated + (
                self.consolidation_scale
                * write_gate.float()
                * consolidation
            ).unsqueeze(-1) * slow_update
        else:
            consolidation = write_gate.new_zeros(write_gate.shape)
        if not self.write_enabled:
            next_matrix = memory.matrix.float()
            consolidated = memory.consolidated.float()
        memory_dtype = torch.float32 if self.state_fp32 else state.dtype
        next_memory = FastWeightMemoryState(
            next_matrix.to(memory_dtype),
            consolidated.to(memory_dtype),
            token_embedding.to(memory_dtype),
        )
        zeros = read_gate.squeeze(-1).float().new_zeros(read_gate.shape[0])
        diagnostics = {
            "read_entropy": zeros,
            "write_entropy": zeros,
            "write_gate": write_gate.squeeze(-1).float(),
            "read_gate": read_gate.squeeze(-1).float(),
            "forget_gate": retention.squeeze(-1).float(),
            "consolidation_gate": consolidation.squeeze(-1).float(),
            "slot_usage": next_matrix.float().square().mean(dim=(-2, -1)).sqrt(),
            "consolidated_norm": consolidated.square().mean(dim=(-2, -1)).sqrt(),
            "read_norm": read.float().norm(dim=-1),
        }
        return next_state.to(output_dtype), next_memory, diagnostics

    def forward_sequence(
        self,
        states: torch.Tensor,
        token_embeddings: torch.Tensor,
        memory: FastWeightMemoryState | None = None,
    ) -> tuple[torch.Tensor, FastWeightMemoryState, dict[str, torch.Tensor]]:
        if memory is None:
            memory = self.initial_state(states.shape[0], states.device, states.dtype)
        outputs: list[torch.Tensor] = []
        diagnostic_rows: dict[str, list[torch.Tensor]] = {}
        for position in range(states.shape[1]):
            current, memory, diagnostics = self.step(
                states[:, position], token_embeddings[:, position], memory
            )
            outputs.append(current)
            for name, value in diagnostics.items():
                diagnostic_rows.setdefault(name, []).append(value)
        stacked = {
            name: torch.stack(values, dim=1) for name, values in diagnostic_rows.items()
        }
        return torch.stack(outputs, dim=1), memory, stacked


__all__ = ["FastWeightMemory", "FastWeightMemoryState"]
