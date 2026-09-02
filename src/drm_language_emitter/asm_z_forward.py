from __future__ import annotations

from typing import Any

import torch

from .losses import next_token_cross_entropy


class ASMZForwardMixin:
    """Causal sequence orchestration for the strict ASM-Z recurrence."""

    def _asm_z_forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        return_states: bool,
        collect_diagnostics: bool,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        batch, seq_len = input_ids.shape
        state = (
            self.initializer(batch, input_ids.device)
            if initial_state is None
            else initial_state
        )
        embeddings = self.token_embedding(input_ids)
        states = []
        potentials = []
        gradient_norms = []
        condition_values = []
        for position in range(seq_len):
            token = embeddings[:, position]
            for _ in range(self.config.n_flow_steps):
                state, geometry = self.asm_z_core(state, token)
            states.append(state)
            if collect_diagnostics:
                potentials.append(geometry.potential)
                gradient_norms.append(geometry.gradient.norm(dim=-1))
                eigenvalues = torch.linalg.eigvalsh(geometry.metric.float())
                condition_values.append(eigenvalues[:, -1] / eigenvalues[:, 0])
        state_tensor = torch.stack(states, dim=1)
        logits = self.emitter(state_tensor)
        loss = next_token_cross_entropy(logits, targets) if targets is not None else None
        diagnostics: dict[str, torch.Tensor] = {}
        if collect_diagnostics:
            diagnostics = {
                "asm_z_potential_mean": torch.stack(potentials, dim=1).mean(),
                "asm_z_gradient_norm_mean": torch.stack(gradient_norms, dim=1).mean(),
                "asm_z_condition_mean": torch.stack(condition_values, dim=1).mean(),
            }
        output: dict[str, Any] = {
            "logits": logits,
            "loss": loss,
            "aux_losses": {},
            "diagnostics": diagnostics,
        }
        if return_states:
            output["states"] = state_tensor
        return output

    def _decode_asm_z(self, input_ids: torch.Tensor, inference_state):
        from .inference import InferenceState

        state = inference_state.completed_state
        if state is None:
            state = self.initializer(input_ids.shape[0], input_ids.device)
        token = self.token_embedding(input_ids)[:, 0]
        for _ in range(self.config.n_flow_steps):
            state, _ = self.asm_z_core(state, token)
        logits = self.emitter(state.unsqueeze(1))[:, 0]
        next_state = InferenceState(
            input_ids=inference_state.input_ids,
            completed_state=state.detach(),
            tokens_seen=inference_state.sequence_length + 1,
            compact=True,
        )
        return logits, next_state
