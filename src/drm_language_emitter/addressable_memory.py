"""Fixed-capacity content-addressable memory for compact ASM streaming."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AddressableMemoryState:
    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    age: torch.Tensor
    previous_token: torch.Tensor

    def detach(self) -> "AddressableMemoryState":
        return AddressableMemoryState(
            *(tensor.detach() for tensor in (self.keys, self.values, self.usage, self.age, self.previous_token))
        )


class AddressableMemory(nn.Module):
    """Differentiable read/write memory whose persistent size is fixed by slots."""

    def __init__(self, config) -> None:
        super().__init__()
        self.slots = int(config.addressable_memory_slots)
        self.d_memory = int(config.addressable_memory_dim)
        self.read_scale = float(config.addressable_memory_read_scale)
        self.temperature = float(config.addressable_memory_temperature)
        self.usage_decay = float(config.addressable_memory_usage_decay)
        self.age_bias = float(config.addressable_memory_age_bias)
        self.read_enabled = bool(config.addressable_memory_read_enabled)
        self.write_enabled = bool(config.addressable_memory_write_enabled)
        self.shuffle_on_eval = bool(config.addressable_memory_shuffle_on_eval)
        self.read_top_k = int(config.addressable_memory_read_top_k)
        self.write_top_k = int(config.addressable_memory_write_top_k)
        self.use_previous_token_key = bool(config.addressable_memory_use_previous_token_key)
        feature_dim = int(config.d_state + config.d_token)
        self.query = nn.Linear(feature_dim, self.d_memory)
        self.write_query = nn.Linear(feature_dim, self.d_memory)
        self.key_candidate = nn.Linear(feature_dim, self.d_memory)
        self.value_candidate = nn.Linear(feature_dim, self.d_memory)
        self.erase = nn.Linear(feature_dim, self.d_memory)
        self.read_gate = nn.Linear(feature_dim, 1)
        self.write_gate = nn.Linear(feature_dim, 1)
        self.read_output = nn.Linear(self.d_memory, config.d_state, bias=False)
        self.initial_keys = nn.Parameter(torch.empty(self.slots, self.d_memory))
        self.initial_values = nn.Parameter(torch.zeros(self.slots, self.d_memory))
        nn.init.normal_(self.initial_keys, std=self.d_memory ** -0.5)
        nn.init.constant_(self.write_gate.bias, float(config.addressable_memory_write_bias))
        nn.init.zeros_(self.read_output.weight)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> AddressableMemoryState:
        keys = self.initial_keys.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        values = self.initial_values.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        zeros = torch.zeros(batch_size, self.slots, device=device, dtype=dtype)
        previous_token = torch.zeros(batch_size, self.query.in_features - self.read_output.out_features, device=device, dtype=dtype)
        return AddressableMemoryState(keys, values, zeros, zeros, previous_token)

    @staticmethod
    def _sparse_weights(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        dense = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        if top_k <= 0 or top_k >= logits.shape[-1]:
            return dense
        _, indices = torch.topk(logits, k=top_k, dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, indices, True)
        sparse_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        sparse = torch.softmax(sparse_logits.float(), dim=-1).to(logits.dtype)
        # Sparse forward values with dense-softmax gradients through slot selection.
        return sparse + dense - dense.detach()

    def step(
        self,
        state: torch.Tensor,
        token_embedding: torch.Tensor,
        memory: AddressableMemoryState,
    ) -> tuple[torch.Tensor, AddressableMemoryState, dict[str, torch.Tensor]]:
        features = torch.cat([state, token_embedding], dim=-1)
        query = F.normalize(self.query(features).float(), dim=-1).to(state.dtype)
        keys_norm = F.normalize(memory.keys.float(), dim=-1).to(state.dtype)
        read_logits = torch.einsum("bd,bsd->bs", query, keys_norm) / self.temperature
        read_weights = self._sparse_weights(read_logits, self.read_top_k)
        read_values = (
            torch.roll(memory.values, shifts=1, dims=1)
            if self.shuffle_on_eval and not self.training
            else memory.values
        )
        read = torch.einsum("bs,bsd->bd", read_weights, read_values)
        read_gate = torch.sigmoid(self.read_gate(features)) if self.read_enabled else state.new_zeros(state.shape[0], 1)
        next_state = state + self.read_scale * read_gate * self.read_output(read)

        key_features = (
            torch.cat([state, memory.previous_token], dim=-1)
            if self.use_previous_token_key
            else features
        )
        candidate_key = torch.tanh(self.key_candidate(key_features))
        write_query = F.normalize(
            candidate_key.float() if self.use_previous_token_key else self.write_query(features).float(),
            dim=-1,
        ).to(state.dtype)
        write_logits = torch.einsum("bd,bsd->bs", write_query, keys_norm) / self.temperature
        write_logits = write_logits - memory.usage + self.age_bias * memory.age
        write_weights = self._sparse_weights(write_logits, self.write_top_k)
        write_gate = torch.sigmoid(self.write_gate(features)) if self.write_enabled else state.new_zeros(state.shape[0], 1)
        allocation = write_gate * write_weights
        candidate_value = torch.tanh(self.value_candidate(features))
        erase = torch.sigmoid(self.erase(features))
        allocation_3d = allocation.unsqueeze(-1)
        next_keys = memory.keys * (1.0 - allocation_3d * erase.unsqueeze(1)) + allocation_3d * candidate_key.unsqueeze(1)
        next_values = memory.values * (1.0 - allocation_3d * erase.unsqueeze(1)) + allocation_3d * candidate_value.unsqueeze(1)
        next_usage = (self.usage_decay * memory.usage + allocation).clamp(0.0, 1.0)
        next_age = torch.where(
            allocation > (1.0 / self.slots),
            torch.zeros_like(memory.age),
            self.usage_decay * memory.age + (1.0 - self.usage_decay),
        ).clamp(0.0, 1.0)
        next_memory = AddressableMemoryState(
            next_keys, next_values, next_usage, next_age, token_embedding
        )
        eps = torch.finfo(torch.float32).eps
        diagnostics = {
            "read_entropy": -(read_weights.float() * (read_weights.float() + eps).log()).sum(-1),
            "write_entropy": -(write_weights.float() * (write_weights.float() + eps).log()).sum(-1),
            "write_gate": write_gate.squeeze(-1).float(),
            "read_gate": read_gate.squeeze(-1).float(),
            "slot_usage": next_usage.float().mean(-1),
            "read_norm": read.float().norm(dim=-1),
        }
        return next_state, next_memory, diagnostics

    def forward_sequence(
        self,
        states: torch.Tensor,
        token_embeddings: torch.Tensor,
        memory: AddressableMemoryState | None = None,
    ) -> tuple[torch.Tensor, AddressableMemoryState, dict[str, torch.Tensor]]:
        if memory is None:
            memory = self.initial_state(states.shape[0], states.device, states.dtype)
        output = []
        diagnostic_rows: dict[str, list[torch.Tensor]] = {}
        for position in range(states.shape[1]):
            current, memory, diagnostics = self.step(states[:, position], token_embeddings[:, position], memory)
            output.append(current)
            for name, value in diagnostics.items():
                diagnostic_rows.setdefault(name, []).append(value)
        stacked = {name: torch.stack(values, dim=1) for name, values in diagnostic_rows.items()}
        return torch.stack(output, dim=1), memory, stacked


__all__ = ["AddressableMemory", "AddressableMemoryState"]
