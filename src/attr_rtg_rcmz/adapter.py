"""True history-snapshot/candidate-fork risk adapter."""

from __future__ import annotations

from dataclasses import fields

import torch
from torch import nn

from .backbones import initialize
from .config import ModelConfig
from .contracts import InferenceMessage, InferenceResult


class RiskAdapter(nn.Module):
    """Process each history once, then consume each four-byte frame in its fork."""

    def __init__(self, config: ModelConfig, backbone: nn.Module) -> None:
        super().__init__()
        self.config, self.backbone = config, backbone
        self.common24 = nn.Linear(config.d_state, config.common_state_dim)
        self.common_activation = (
            nn.PReLU(config.common_state_dim) if config.common_prelu else nn.Identity()
        )
        self.common_norm = (
            nn.LayerNorm(config.common_state_dim)
            if config.common_layernorm
            else nn.Identity()
        )
        self.readout = nn.Sequential(
            nn.Linear(config.common_state_dim, config.readout_hidden1),
            nn.GELU(),
            nn.Linear(config.readout_hidden1, config.readout_hidden2),
            nn.GELU(),
            nn.Linear(config.readout_hidden2, 1),
        )
        initialize(self, config.training_seed)
        actual = self.trainable_parameter_count()
        if actual != config.target_trainable_parameters:
            raise ValueError(
                f"trainable budget differs: {actual} != {config.target_trainable_parameters}"
            )

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def graph_active_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.grad is not None)

    def _history_snapshot(self, tokens: torch.Tensor, lengths: torch.Tensor):
        """Run a fixed 256-position device schedule with per-row state masks."""
        if self.config.arm == "T":
            return self._transformer_history(tokens, lengths), [None] * tokens.shape[0]
        state = self.backbone.initializer(tokens.shape[0], tokens.device)
        embeddings = self.backbone.token_embedding(tokens.long())
        memory = None
        if self.config.arm == "CM":
            memory = self.backbone.addressable_memory.initial_state(
                tokens.shape[0], tokens.device, state.dtype
            )
        for position in range(self.config.context_length):
            token = embeddings[:, position]
            if self.config.arm == "Z":
                proposed, _ = self.backbone.asm_z_core(state, token)
            else:
                proposed = self._drm_fork(state, token)
                if memory is not None:
                    proposed, proposed_memory, _ = (
                        self.backbone.addressable_memory.step(proposed, token, memory)
                    )
            active = (lengths > position).unsqueeze(-1)
            state = torch.where(active, proposed, state)
            if memory is not None:
                memory = _masked_memory(memory, proposed_memory, active)
        return state, memory

    def _transformer_history(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(self.config.context_length, device=tokens.device)
        values = self.backbone.token_embedding(tokens.long())
        values = values + self.backbone.position_embedding(positions)
        causal = torch.triu(
            torch.ones(
                self.config.context_length,
                self.config.context_length,
                device=tokens.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        padding = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        states = self.backbone.norm(
            self.backbone.encoder(values, mask=causal, src_key_padding_mask=padding)
        )
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        return states[rows, lengths.long() - 1]

    def _frame_embedding(self, frames: torch.Tensor) -> torch.Tensor:
        # The four bytes form one candidate-frame input. ASM-Z therefore performs
        # one solve/update for the frame, not four byte-token updates.
        byte_tokens = (
            frames.round().long().remainder(self.config.vocab_size).reshape(-1, 4)
        )
        return self.backbone.token_embedding(byte_tokens).mean(dim=1)

    def _drm_fork(self, state: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        diagonal, low_rank = self.backbone.metric(state)
        if self.backbone.direction_field is None:
            raw = self.backbone.direct_transition(state, token)
        else:
            directions, gates = self.backbone.direction_field(state)
            raw, _ = self.backbone.flow(state, token, directions, gates)
        step = self.backbone.metric.naturalize(
            raw,
            diagonal,
            low_rank,
            strength=self.backbone.config.metric_naturalization_strength,
            damping=self.backbone.config.metric_damping,
        )
        return self.backbone.updater(state, step)

    def _candidate_forks(self, snapshot: torch.Tensor, frames: torch.Tensor, memories):
        batch, candidates, _ = frames.shape
        state = (
            snapshot[:, None, :]
            .expand(-1, candidates, -1)
            .reshape(-1, snapshot.shape[-1])
        )
        token = self._frame_embedding(frames)
        if self.config.arm == "Z":
            post, _ = self.backbone.asm_z_core(state, token)
        elif self.config.arm in {"R", "CM"}:
            post = self._drm_fork(state, token)
            if self.config.arm == "CM":
                memory = _repeat_memory(memories, candidates)
                post, _, _ = self.backbone.addressable_memory.step(post, token, memory)
        else:
            positions = torch.tensor([0, 1], device=state.device)
            pair = torch.stack(
                (state, token), dim=1
            ) + self.backbone.position_embedding(positions)
            causal = torch.triu(
                torch.ones(2, 2, device=state.device, dtype=torch.bool), diagonal=1
            )
            post = self.backbone.norm(self.backbone.encoder(pair, mask=causal)[:, -1])
        return post.reshape(batch, candidates, -1)

    def forward(self, message: InferenceMessage) -> InferenceResult:
        if not isinstance(message, InferenceMessage):
            raise TypeError("adapter accepts one validated InferenceMessage")
        message.validate(self.config.context_length)
        snapshot, memories = self._history_snapshot(
            message.history_bytes, message.logical_lengths
        )
        native = self._candidate_forks(snapshot, message.candidate4s, memories)
        common = self.common24(native)
        common = self.common_activation(
            common.reshape(-1, common.shape[-1])
        ).reshape_as(common)
        common = self.common_norm(common)
        logits = self.readout(common).squeeze(-1)
        flat_native = native.reshape(-1, native.shape[-1])
        if self.config.arm == "T":
            evidence = self.backbone.lm_head(flat_native).mean(-1)
        else:
            evidence = self.backbone.emitter(flat_native[:, None, :])[:, 0].mean(-1)
            if self.backbone.risk is not None:
                evidence = evidence + self.backbone.risk(flat_native)["risk_mass"]
        logits = logits + evidence.reshape_as(logits)
        return InferenceResult(logits.masked_fill(~message.masks, 0.0), common, native)


def _repeat_memory(memory: object, candidates: int):
    return type(memory)(
        *(
            getattr(memory, field.name).repeat_interleave(candidates, dim=0)
            for field in fields(memory)
        )
    )


def _masked_memory(previous: object, proposed: object, active: torch.Tensor):
    values = []
    for field in fields(previous):
        old, new = getattr(previous, field.name), getattr(proposed, field.name)
        mask = active
        while mask.ndim < old.ndim:
            mask = mask.unsqueeze(-1)
        values.append(torch.where(mask, new, old))
    return type(previous)(*values)
