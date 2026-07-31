from __future__ import annotations

from typing import Any

import torch

from .losses import next_token_cross_entropy, recurrence_proxy, stability_proxy


class SelectiveControlMixin:
    """Geometry-free selective-memory control used for causal ablations."""

    def _forward_selective_control(
        self,
        z0: torch.Tensor,
        token_embeddings: torch.Tensor,
        targets: torch.Tensor | None,
        return_states: bool,
        collect_diagnostics: bool,
    ) -> dict[str, Any]:
        if self.selective_memory is None:
            raise RuntimeError("geometry-free control requires selective_memory")

        batch, seq_len, _ = token_embeddings.shape
        block_size = self.config.directional_cumsum_block_size or seq_len
        block_size = max(min(int(block_size), seq_len), 1)
        blocks = []
        z_block = z0
        for block_start in range(0, seq_len, block_size):
            block_tokens = token_embeddings[:, block_start : block_start + block_size]
            block_len = block_tokens.shape[1]
            states = z_block.unsqueeze(1).expand(-1, block_len, -1)
            local_delta = torch.zeros_like(states)
            zeros = states.new_zeros(batch, block_len)
            if self.local_mixer is not None:
                states = self._bound_state(
                    self.local_mixer(
                        z_block,
                        states,
                        block_tokens,
                        local_delta,
                        zeros,
                        zeros,
                    )
                )
            if self.token_state_residual is not None:
                states = self._bound_state(
                    self.token_state_residual(states, block_tokens)
                )
            states = self._bound_state(
                self.selective_memory(z_block, states, block_tokens)
            )
            blocks.append(states)
            z_block = states[:, -1]

        states = torch.cat(blocks, dim=1)
        logits = self.emitter(states)
        zero = logits.new_tensor(0.0)
        ce_loss = (
            next_token_cross_entropy(logits, targets)
            if targets is not None
            else zero
        )
        recurrence = (
            recurrence_proxy(states)
            if return_states or collect_diagnostics
            else zero
        )
        stability = stability_proxy(logits) if collect_diagnostics else zero
        aux_losses = {
            "ce": ce_loss,
            "action": zero,
            "dim_sparsity": zero,
            "dim_entropy": zero,
            "metric_reg": zero,
            "metric_diversity": zero,
            "recurrence": recurrence,
            "stability": stability,
            "blindspot": zero,
            "total": ce_loss,
        }
        diagnostics = {
            "geometry_enabled": zero,
            "recurrence_proxy": recurrence,
            "stability_proxy": stability,
        }
        out: dict[str, Any] = {
            "logits": logits,
            "loss": ce_loss,
            "aux_losses": aux_losses,
            "diagnostics": diagnostics,
        }
        if return_states:
            out["states"] = states
        return out
