from __future__ import annotations

from typing import Any

import torch

from .losses import combine_losses, metric_diversity, next_token_cross_entropy, recurrence_proxy, stability_proxy


class DirectionalForwardMixin:
    def _forward_directional_cumsum(
        self,
        z0: torch.Tensor,
        token_embeddings: torch.Tensor,
        targets: torch.Tensor | None,
        return_states: bool,
        global_step: int | None,
        collect_diagnostics: bool,
    ) -> dict[str, Any]:
        """Approximate the recurrent trajectory by parallel local deltas."""

        batch, seq_len, d_token = token_embeddings.shape
        if self.config.sequence_mode == "directional_block_cumsum":
            block_size = self.config.directional_cumsum_block_size or seq_len
            block_size = max(min(block_size, seq_len), 1)
        elif self.config.sequence_mode == "directional_superblock_cumsum":
            block_size = self.config.directional_superblock_size or self.config.directional_cumsum_block_size or seq_len
            block_size = max(min(block_size, seq_len), 1)
        else:
            block_size = seq_len
        block_states = []
        action_values = []
        dim_values = []
        entropy_values = []
        metric_regs = []
        metric_diag_steps = []
        condition_values = []
        active_050_values = []
        gate_values = []
        u_norm_values = []
        risk_values = []
        consistency_values = []
        sampled_consistency_values = []
        z_block = z0
        for block_index, block_start in enumerate(range(0, seq_len, block_size)):
            block_tokens = token_embeddings[:, block_start : block_start + block_size]
            (
                states_block,
                action_block,
                dim_block,
                entropy_block,
                metric_reg_block,
                metric_diag_block,
                condition_block,
                active_050_block,
                gates_block,
                u_norm_block,
                risk_block,
                consistency_block,
                sampled_consistency_block,
            ) = self._directional_cumsum_block(
                z_block,
                block_tokens,
                global_step,
                block_index,
                collect_diagnostics,
            )
            block_states.append(states_block)
            action_values.append(action_block)
            dim_values.append(dim_block)
            entropy_values.append(entropy_block)
            metric_regs.append(metric_reg_block)
            metric_diag_steps.append(metric_diag_block)
            condition_values.append(condition_block)
            active_050_values.append(active_050_block)
            if collect_diagnostics:
                gate_values.append(gates_block)
            u_norm_values.append(u_norm_block)
            risk_values.append(risk_block)
            if consistency_block is not None:
                consistency_values.append(consistency_block)
            if sampled_consistency_block is not None:
                sampled_consistency_values.append(sampled_consistency_block)
            z_block = states_block[:, -1]

        states = torch.cat(block_states, dim=1)
        logits = self.emitter(states)

        action_loss = torch.cat(action_values, dim=1).mean()
        dim_tensor = torch.cat(dim_values, dim=1)
        dim_sparsity = dim_tensor.mean()
        dim_std_value = dim_tensor.std(unbiased=False)
        dim_entropy_value = torch.stack(entropy_values).mean()
        metric_reg = torch.stack(metric_regs).mean()
        metric_diag_tensor = torch.cat(metric_diag_steps, dim=1)
        metric_div_value = metric_diversity(metric_diag_tensor) if self.config.lambda_metric_diversity != 0 else action_loss.new_tensor(0.0)
        recurrence_value = recurrence_proxy(states) if (return_states or collect_diagnostics or self.config.lambda_recurrence != 0) else action_loss.new_tensor(0.0)
        stability_value = stability_proxy(logits) if (collect_diagnostics or self.config.lambda_stability != 0) else action_loss.new_tensor(0.0)
        risk_tensor = torch.cat(risk_values, dim=1)
        blindspot_value = risk_tensor.mean() if (collect_diagnostics or self.config.lambda_blindspot != 0) else action_loss.new_tensor(0.0)
        active_050_tensor = torch.cat(active_050_values, dim=1)
        hard_active_050_value = active_050_tensor.mean()
        soft_active_value = dim_sparsity / self.config.n_directions
        condition_value = torch.cat(condition_values, dim=1).mean()
        u_norm = torch.cat(u_norm_values, dim=1)
        metric_u_norm_value = u_norm.mean()
        if self.config.metric_rank > 0:
            metric_u_floor_loss = (self.config.metric_u_min_norm - u_norm).clamp_min(0.0).pow(2).mean()
        else:
            metric_u_floor_loss = action_loss.new_tensor(0.0)
        ce_loss = next_token_cross_entropy(logits, targets) if targets is not None else None
        total_loss, aux_losses = combine_losses(
            self.config,
            ce_loss,
            action_loss,
            dim_sparsity,
            dim_entropy_value,
            metric_reg,
            metric_div_value,
            recurrence_value,
            stability_value,
            blindspot_value,
            soft_active_value,
            dim_std_value,
            condition_value,
            metric_u_norm_value,
        )
        if self.config.lambda_metric_u_floor and self.config.metric_rank > 0:
            total_loss = total_loss + self.config.lambda_metric_u_floor * metric_u_floor_loss
            aux_losses["metric_u_floor"] = metric_u_floor_loss
        if self.config.lambda_block_consistency and consistency_values:
            block_consistency = torch.cat(consistency_values, dim=1).mean()
            total_loss = total_loss + self.config.lambda_block_consistency * block_consistency
            aux_losses["block_consistency"] = block_consistency
        else:
            block_consistency = action_loss.new_tensor(0.0)
        if self.config.lambda_sampled_block_consistency and sampled_consistency_values:
            sampled_block_consistency = torch.cat(sampled_consistency_values, dim=1).mean()
            total_loss = total_loss + self.config.lambda_sampled_block_consistency * sampled_block_consistency
            aux_losses["sampled_block_consistency"] = sampled_block_consistency
        else:
            sampled_block_consistency = action_loss.new_tensor(0.0)

        diagnostics = {
            "dimD_mean": dim_sparsity,
            "dimD_std": dim_std_value,
            "soft_active_fraction": soft_active_value,
            "active_fraction": hard_active_050_value,
            "hard_active_fraction_050": hard_active_050_value,
            "gate_entropy": dim_entropy_value,
            "action_mean": action_loss,
            "metric_U_norm_mean": metric_u_norm_value,
            "metric_U_variance": u_norm.var(unbiased=False),
            "condition_proxy": condition_value,
            "recurrence_proxy": recurrence_value,
            "stability_proxy": stability_value,
            "risk_mass_mean": blindspot_value,
            "metric_u_floor_loss": metric_u_floor_loss,
            "block_consistency": block_consistency,
            "sampled_block_consistency": sampled_block_consistency,
            "metric_naturalization_strength": token_embeddings.new_tensor(float(self._naturalization_strength(global_step))),
        }
        if collect_diagnostics:
            all_gates = torch.cat(gate_values, dim=1)
            flat_gates = all_gates.reshape(-1).float()
            gate_quantiles = torch.quantile(
                flat_gates,
                torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=states.device),
            )
            diagnostics.update(
                {
                    "hard_active_fraction_025": (all_gates > 0.25).float().mean(),
                    "hard_active_fraction_075": (all_gates > 0.75).float().mean(),
                    "hard_active_fraction_090": (all_gates > 0.90).float().mean(),
                    "gate_min": all_gates.min(),
                    "gate_max": all_gates.max(),
                    "gate_q10": gate_quantiles[0],
                    "gate_q25": gate_quantiles[1],
                    "gate_q50": gate_quantiles[2],
                    "gate_q75": gate_quantiles[3],
                    "gate_q90": gate_quantiles[4],
                    "risk_mass_std": risk_tensor.std(unbiased=False),
                    "risk_mass_max": risk_tensor.max(),
                }
            )

        out: dict[str, Any] = {
            "logits": logits,
            "loss": total_loss,
            "aux_losses": aux_losses,
            "diagnostics": diagnostics,
        }
        if return_states:
            out["states"] = states
        return out

    def _directional_cumsum_block(
        self,
        z_start: torch.Tensor,
        token_embeddings: torch.Tensor,
        global_step: int | None,
        block_index: int = 0,
        collect_diagnostics: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        inner_block_size = int(self.config.directional_cumsum_inner_block_size)
        block_len = token_embeddings.shape[1]
        if self.config.sequence_mode == "directional_superblock_cumsum":
            return self._directional_superblock_cumsum_block(
                z_start,
                token_embeddings,
                global_step,
                block_index,
                collect_diagnostics,
            )
        if 0 < inner_block_size < block_len:
            states_parts = []
            action_parts = []
            dim_parts = []
            entropy_parts = []
            metric_reg_parts = []
            metric_diag_parts = []
            condition_parts = []
            active_parts = []
            gate_parts = []
            u_norm_parts = []
            risk_parts = []
            consistency_parts = []
            sampled_consistency_parts = []
            z_inner = z_start
            inner_count = max((block_len + inner_block_size - 1) // inner_block_size, 1)
            for inner_index, inner_start in enumerate(range(0, block_len, inner_block_size)):
                inner_tokens = token_embeddings[:, inner_start : inner_start + inner_block_size]
                (
                    states_inner,
                    action_inner,
                    dim_inner,
                    entropy_inner,
                    metric_reg_inner,
                    metric_diag_inner,
                    condition_inner,
                    active_inner,
                    gates_inner,
                    u_norm_inner,
                    risk_inner,
                    consistency_inner,
                    sampled_consistency_inner,
                ) = self._directional_cumsum_block_base(
                    z_inner,
                    inner_tokens,
                    global_step,
                    block_index * inner_count + inner_index,
                    collect_diagnostics,
                )
                states_parts.append(states_inner)
                action_parts.append(action_inner)
                dim_parts.append(dim_inner)
                entropy_parts.append(entropy_inner)
                metric_reg_parts.append(metric_reg_inner)
                metric_diag_parts.append(metric_diag_inner)
                condition_parts.append(condition_inner)
                active_parts.append(active_inner)
                gate_parts.append(gates_inner)
                u_norm_parts.append(u_norm_inner)
                risk_parts.append(risk_inner)
                if consistency_inner is not None:
                    consistency_parts.append(consistency_inner)
                if sampled_consistency_inner is not None:
                    sampled_consistency_parts.append(sampled_consistency_inner)
                z_inner = states_inner[:, -1]
            consistency = torch.cat(consistency_parts, dim=1) if consistency_parts else None
            sampled_consistency = torch.cat(sampled_consistency_parts, dim=1) if sampled_consistency_parts else None
            return (
                torch.cat(states_parts, dim=1),
                torch.cat(action_parts, dim=1),
                torch.cat(dim_parts, dim=1),
                torch.stack(entropy_parts).mean(),
                torch.stack(metric_reg_parts).mean(),
                torch.cat(metric_diag_parts, dim=1),
                torch.cat(condition_parts, dim=1),
                torch.cat(active_parts, dim=1),
                torch.cat(gate_parts, dim=1),
                torch.cat(u_norm_parts, dim=1),
                torch.cat(risk_parts, dim=1),
                consistency,
                sampled_consistency,
            )
        return self._directional_cumsum_block_base(
            z_start,
            token_embeddings,
            global_step,
            block_index,
            collect_diagnostics,
        )
