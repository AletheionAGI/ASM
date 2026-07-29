from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .config import DRMConfig
from .deer import causal_anderson_solve, fixed_point_solve, sequential_rollout
from .direction_field import DirectionField
from .dynamics import DRMFlow, StateUpdater
from .emitter import LanguageEmitter, TokenEmbedding
from .losses import (
    combine_losses,
    dimension_entropy,
    metric_diversity,
    next_token_cross_entropy,
    recurrence_proxy,
    stability_proxy,
)
from .metric import RelationalMetric
from .risk import RiskField


class DRMStateInitializer(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.z0 = nn.Parameter(torch.zeros(config.d_state))
        nn.init.normal_(self.z0, std=0.02)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.z0.unsqueeze(0).expand(batch_size, -1).to(device)


class CausalLocalMixer(nn.Module):
    """Cheap local causal state correction for testing the Anderson-as-mixer hypothesis."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        hidden = config.directional_local_mixer_hidden_size
        in_dim = config.d_state * 3 + config.d_token + 2
        self.input_proj = nn.Linear(in_dim, hidden)
        self.depthwise = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden,
                    hidden,
                    kernel_size=config.directional_local_mixer_kernel_size,
                    groups=hidden,
                    bias=True,
                )
                for _ in range(config.directional_local_mixer_layers)
            ]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel_size=1, bias=True) for _ in range(config.directional_local_mixer_layers)]
        )
        self.output_proj = nn.Linear(hidden, config.d_state)

    def forward(
        self,
        z_start: torch.Tensor,
        states: torch.Tensor,
        token_embeddings: torch.Tensor,
        local_delta: torch.Tensor,
        dim: torch.Tensor,
        risk_mass: torch.Tensor,
    ) -> torch.Tensor:
        previous = torch.cat([z_start.unsqueeze(1), states[:, :-1]], dim=1)
        residual = states - previous
        features = torch.cat(
            [
                states,
                residual,
                local_delta,
                token_embeddings,
                dim.unsqueeze(-1),
                risk_mass.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = torch.nn.functional.silu(self.input_proj(features)).transpose(1, 2)
        kernel = int(self.config.directional_local_mixer_kernel_size)
        for depthwise, pointwise in zip(self.depthwise, self.pointwise):
            padded = torch.nn.functional.pad(hidden, (kernel - 1, 0))
            mixed = pointwise(torch.nn.functional.silu(depthwise(padded)))
            hidden = hidden + mixed
        correction = self.output_proj(hidden.transpose(1, 2))
        return states + float(self.config.directional_local_mixer_scale) * correction


class DRMEmitterModel(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config)
        self.initializer = DRMStateInitializer(config)
        self.direction_field = DirectionField(config)
        self.metric = RelationalMetric(config)
        self.flow = DRMFlow(config)
        self.updater = StateUpdater(config)
        self.risk = RiskField(config)
        self.emitter = LanguageEmitter(config)
        self.local_mixer = CausalLocalMixer(config) if config.directional_local_mixer != "none" else None
        self._compiled_forward = None
        if config.use_torch_compile and hasattr(torch, "compile"):
            try:
                self._compiled_forward = torch.compile(self._forward_impl)
            except Exception:
                self._compiled_forward = None

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_states: bool = False,
        global_step: int | None = None,
        collect_diagnostics: bool = True,
    ) -> dict[str, Any]:
        batch, seq_len = input_ids.shape
        z = self.initializer(batch, input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        if self.config.sequence_mode in {
            "directional_cumsum",
            "directional_block_cumsum",
            "directional_superblock_cumsum",
        }:
            return self._forward_directional_cumsum(
                z,
                token_embeddings,
                targets,
                return_states,
                global_step,
                collect_diagnostics,
            )
        need_states = return_states or collect_diagnostics or self.config.lambda_recurrence != 0
        need_stability = collect_diagnostics or self.config.lambda_stability != 0
        need_metric_diversity = collect_diagnostics or self.config.lambda_metric_diversity != 0
        emission_states = []
        states = []
        action_values = []
        dim_values = []
        entropy_values = []
        metric_regs = []
        metric_diag_steps = []
        condition_values = []
        active_025_values = []
        active_050_values = []
        active_075_values = []
        active_090_values = []
        gate_min_values = []
        gate_max_values = []
        gate_flat_values = []
        u_norm_values = []
        risk_values = []
        naturalization_strength = self._naturalization_strength(global_step)
        geometry_interval = max(int(self.config.geometry_update_interval), 1)
        geometry_tick = 0
        cached_directions: torch.Tensor | None = None
        cached_gates: torch.Tensor | None = None
        cached_metric_diag: torch.Tensor | None = None
        cached_metric_u: torch.Tensor | None = None
        cached_risk: dict[str, torch.Tensor] | None = None
        truncate_interval = int(self.config.bptt_truncate_interval)

        for t in range(seq_len):
            e_t = token_embeddings[:, t]
            for _ in range(self.config.n_flow_steps):
                if geometry_tick % geometry_interval == 0 or cached_directions is None:
                    cached_directions, cached_gates = self.direction_field(z)
                    cached_metric_diag, cached_metric_u = self.metric(z)
                    cached_risk = self.risk(z)
                directions = cached_directions
                gates = cached_gates
                metric_diag = cached_metric_diag
                metric_u = cached_metric_u
                risk = cached_risk
                assert gates is not None
                assert metric_diag is not None
                assert metric_u is not None
                assert risk is not None
                dz_raw, _coefficients = self.flow(z, e_t, directions, gates)
                dz = self.metric.naturalize(
                    dz_raw,
                    metric_diag,
                    metric_u,
                    strength=naturalization_strength,
                    damping=self.config.metric_damping,
                )
                if self.config.sequence_mode == "geodesic_step" and self.config.geodesic_solver_steps > 0:
                    z_next = self._geodesic_step(z, dz, metric_diag, metric_u)
                    dz = (z_next - z) / max(self.config.dt, 1e-8)
                elif self.config.sequence_mode == "directional_candidates":
                    z_next = self._directional_candidate_step(z, dz, directions, gates, metric_diag, metric_u)
                    dz = (z_next - z) / max(self.config.dt, 1e-8)
                energy = self.metric.metric_energy(
                    z, dz, metric_diag, metric_u, risk_mass=risk["risk_mass"]
                )
                action_values.append(energy)
                dim_values.append(gates.sum(dim=-1))
                entropy_values.append(dimension_entropy(gates))
                u_norm = metric_u.norm(dim=(1, 2)) if metric_u.numel() else metric_diag.new_zeros(batch)
                metric_regs.append(metric_diag.pow(2).mean() + metric_u.pow(2).mean())
                if need_metric_diversity:
                    metric_diag_steps.append(metric_diag)
                condition_values.append(self.metric.condition_proxy(metric_diag, metric_u))
                active_050_values.append((gates > 0.50).float().mean(dim=-1))
                if collect_diagnostics:
                    active_025_values.append((gates > 0.25).float().mean(dim=-1))
                    active_075_values.append((gates > 0.75).float().mean(dim=-1))
                    active_090_values.append((gates > 0.90).float().mean(dim=-1))
                    gate_min_values.append(gates.min(dim=-1).values)
                    gate_max_values.append(gates.max(dim=-1).values)
                    gate_flat_values.append(gates.reshape(-1))
                u_norm_values.append(u_norm)
                if collect_diagnostics or self.config.lambda_blindspot != 0:
                    risk_values.append(risk["risk_mass"])
                if self.config.sequence_mode in {"geodesic_step", "directional_candidates"} and "z_next" in locals():
                    z = z_next
                else:
                    z = self.updater(z, dz)
                if "z_next" in locals():
                    del z_next
                geometry_tick += 1
            emission_states.append(z)
            if need_states:
                states.append(z)
            should_truncate = (
                self.training
                and truncate_interval > 0
                and (t + 1) % truncate_interval == 0
                and (t + 1) < seq_len
            )
            if should_truncate:
                z = z.detach()
                cached_directions = None
                cached_gates = None
                cached_metric_diag = None
                cached_metric_u = None
                cached_risk = None

        emission_state_tensor = torch.stack(emission_states, dim=1)
        logits = self.emitter(emission_state_tensor)
        state_tensor = torch.stack(states, dim=1) if need_states else None
        metric_diag_tensor = torch.stack(metric_diag_steps, dim=1) if need_metric_diversity else None
        action_loss = torch.stack(action_values, dim=1).mean()
        dim_tensor = torch.stack(dim_values, dim=1)
        dim_sparsity = dim_tensor.mean()
        dim_std_value = dim_tensor.std(unbiased=False)
        dim_entropy_value = torch.stack(entropy_values).mean()
        metric_reg = torch.stack(metric_regs).mean()
        metric_u_norm_steps = torch.stack(u_norm_values, dim=1)
        if self.config.metric_rank > 0:
            metric_u_floor_loss = (
                (self.config.metric_u_min_norm - metric_u_norm_steps)
                .clamp_min(0.0)
                .pow(2)
                .mean()
            )
        else:
            metric_u_floor_loss = action_loss.new_tensor(0.0)
        metric_div_value = metric_diversity(metric_diag_tensor) if metric_diag_tensor is not None else action_loss.new_tensor(0.0)
        recurrence_value = recurrence_proxy(state_tensor) if state_tensor is not None else action_loss.new_tensor(0.0)
        stability_value = stability_proxy(logits) if need_stability else action_loss.new_tensor(0.0)
        blindspot_value = torch.stack(risk_values, dim=1).mean() if risk_values else action_loss.new_tensor(0.0)
        hard_active_050_value = torch.stack(active_050_values, dim=1).mean()
        soft_active_value = dim_sparsity / self.config.n_directions
        condition_value = torch.stack(condition_values, dim=1).mean()
        metric_u_norm_value = metric_u_norm_steps.mean()
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

        diagnostics = {
            "dimD_mean": dim_sparsity,
            "dimD_std": dim_std_value,
            "soft_active_fraction": soft_active_value,
            "active_fraction": hard_active_050_value,
            "hard_active_fraction_050": hard_active_050_value,
            "gate_entropy": dim_entropy_value,
            "action_mean": action_loss,
            "metric_U_norm_mean": metric_u_norm_value,
            "metric_U_variance": torch.stack(u_norm_values, dim=1).var(unbiased=False),
            "condition_proxy": condition_value,
            "recurrence_proxy": recurrence_value,
            "stability_proxy": stability_value,
            "risk_mass_mean": blindspot_value,
            "metric_u_floor_loss": metric_u_floor_loss,
            "metric_naturalization_strength": input_ids.new_tensor(float(naturalization_strength), dtype=torch.float32),
        }
        if collect_diagnostics:
            hard_active_025_value = torch.stack(active_025_values, dim=1).mean()
            hard_active_075_value = torch.stack(active_075_values, dim=1).mean()
            hard_active_090_value = torch.stack(active_090_values, dim=1).mean()
            all_gates = torch.cat(gate_flat_values).float()
            gate_quantiles = torch.quantile(
                all_gates,
                torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=all_gates.device, dtype=all_gates.dtype),
            )
            risk_tensor = torch.stack(risk_values, dim=1) if risk_values else action_loss.new_zeros(batch, 1)
            diagnostics.update(
                {
                    "hard_active_fraction_025": hard_active_025_value,
                    "hard_active_fraction_075": hard_active_075_value,
                    "hard_active_fraction_090": hard_active_090_value,
                    "gate_min": torch.stack(gate_min_values, dim=1).min(),
                    "gate_max": torch.stack(gate_max_values, dim=1).max(),
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
        if return_states and state_tensor is not None:
            out["states"] = state_tensor
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_states: bool = False,
        global_step: int | None = None,
        collect_diagnostics: bool = True,
    ):
        if self._compiled_forward is not None:
            try:
                return self._compiled_forward(input_ids, targets, return_states, global_step, collect_diagnostics)
            except Exception:
                self._compiled_forward = None
        return self._forward_impl(input_ids, targets, return_states, global_step, collect_diagnostics)

    def state_dict_with_config(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "model": self.state_dict()}

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
                u_norm_block,
                risk_block,
                consistency_block,
                sampled_consistency_block,
            ) = self._directional_cumsum_block(z_block, block_tokens, global_step, block_index)
            block_states.append(states_block)
            action_values.append(action_block)
            dim_values.append(dim_block)
            entropy_values.append(entropy_block)
            metric_regs.append(metric_reg_block)
            metric_diag_steps.append(metric_diag_block)
            condition_values.append(condition_block)
            active_050_values.append(active_050_block)
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
            flat_dim = dim_tensor.reshape(-1)
            gate_quantiles = torch.quantile(
                flat_dim.float(),
                torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=states.device),
            )
            diagnostics.update(
                {
                    "hard_active_fraction_025": (dim_tensor > 0.25).float().mean(),
                    "hard_active_fraction_075": (dim_tensor > 0.75).float().mean(),
                    "hard_active_fraction_090": (dim_tensor > 0.90).float().mean(),
                    "gate_min": dim_tensor.min(),
                    "gate_max": dim_tensor.max(),
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
            return self._directional_superblock_cumsum_block(z_start, token_embeddings, global_step, block_index)
        if 0 < inner_block_size < block_len:
            states_parts = []
            action_parts = []
            dim_parts = []
            entropy_parts = []
            metric_reg_parts = []
            metric_diag_parts = []
            condition_parts = []
            active_parts = []
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
                    u_norm_inner,
                    risk_inner,
                    consistency_inner,
                    sampled_consistency_inner,
                ) = self._directional_cumsum_block_base(
                    z_inner,
                    inner_tokens,
                    global_step,
                    block_index * inner_count + inner_index,
                )
                states_parts.append(states_inner)
                action_parts.append(action_inner)
                dim_parts.append(dim_inner)
                entropy_parts.append(entropy_inner)
                metric_reg_parts.append(metric_reg_inner)
                metric_diag_parts.append(metric_diag_inner)
                condition_parts.append(condition_inner)
                active_parts.append(active_inner)
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
                torch.cat(u_norm_parts, dim=1),
                torch.cat(risk_parts, dim=1),
                consistency,
                sampled_consistency,
            )
        return self._directional_cumsum_block_base(z_start, token_embeddings, global_step, block_index)

    def _directional_superblock_cumsum_block(
        self,
        z_start: torch.Tensor,
        token_embeddings: torch.Tensor,
        global_step: int | None,
        block_index: int = 0,
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
        local_size = int(self.config.directional_superblock_local_size)
        block_len = token_embeddings.shape[1]
        if block_len <= local_size or block_len % local_size != 0:
            return self._directional_cumsum_block_base(z_start, token_embeddings, global_step, block_index)

        batch, _block_len, d_token = token_embeddings.shape
        segment_count = block_len // local_size

        coarse_states = self._directional_superblock_coarse_states(z_start, token_embeddings, global_step)
        if segment_count == 1:
            segment_starts = z_start.unsqueeze(1)
        else:
            previous_endpoints = coarse_states[:, local_size - 1 : block_len - 1 : local_size]
            segment_starts = torch.cat([z_start.unsqueeze(1), previous_endpoints], dim=1)

        flat_starts = segment_starts.reshape(batch * segment_count, z_start.shape[-1])
        segment_tokens = token_embeddings.reshape(batch, segment_count, local_size, d_token)
        flat_tokens = segment_tokens.reshape(batch * segment_count, local_size, d_token)
        (
            flat_states,
            flat_action,
            flat_dim,
            entropy,
            metric_reg,
            flat_metric_diag,
            flat_condition,
            flat_active,
            flat_u_norm,
            flat_risk,
            flat_consistency,
            flat_sampled_consistency,
        ) = self._directional_cumsum_block_base(flat_starts, flat_tokens, global_step, block_index)

        states = flat_states.reshape(batch, segment_count, local_size, -1).reshape(batch, block_len, -1)
        action = flat_action.reshape(batch, segment_count, local_size).reshape(batch, block_len)
        dim = flat_dim.reshape(batch, segment_count, local_size).reshape(batch, block_len)
        metric_diag = flat_metric_diag.reshape(batch, segment_count, local_size, -1).reshape(batch, block_len, -1)
        condition = flat_condition.reshape(batch, segment_count, local_size).reshape(batch, block_len)
        active = flat_active.reshape(batch, segment_count, local_size).reshape(batch, block_len)
        u_norm = flat_u_norm.reshape(batch, segment_count, local_size).reshape(batch, block_len)
        risk = flat_risk.reshape(batch, segment_count, local_size).reshape(batch, block_len)
        consistency = (
            flat_consistency.reshape(batch, segment_count, local_size).reshape(batch, block_len)
            if flat_consistency is not None
            else None
        )
        sampled_consistency = (
            flat_sampled_consistency.reshape(batch, segment_count, local_size).reshape(batch, block_len)
            if flat_sampled_consistency is not None
            else None
        )
        return (
            states,
            action,
            dim,
            entropy,
            metric_reg,
            metric_diag,
            condition,
            active,
            u_norm,
            risk,
            consistency,
            sampled_consistency,
        )

    def _directional_superblock_coarse_states(
        self,
        z_start: torch.Tensor,
        token_embeddings: torch.Tensor,
        global_step: int | None,
    ) -> torch.Tensor:
        batch, block_len, d_token = token_embeddings.shape
        flat_z = z_start.unsqueeze(1).expand(-1, block_len, -1).reshape(batch * block_len, z_start.shape[-1])
        flat_tokens = token_embeddings.reshape(batch * block_len, d_token)
        base_directions, base_gates = self.direction_field(z_start)
        base_metric_diag, base_metric_u = self.metric(z_start)
        directions = (
            base_directions.unsqueeze(1)
            .expand(-1, block_len, -1, -1)
            .reshape(batch * block_len, self.config.n_directions, self.config.d_state)
        )
        gates = (
            base_gates.unsqueeze(1)
            .expand(-1, block_len, -1)
            .reshape(batch * block_len, self.config.n_directions)
        )
        metric_diag = (
            base_metric_diag.unsqueeze(1)
            .expand(-1, block_len, -1)
            .reshape(batch * block_len, self.config.d_state)
        )
        metric_u = (
            base_metric_u.unsqueeze(1)
            .expand(-1, block_len, -1, -1)
            .reshape(batch * block_len, self.config.d_state, self.config.metric_rank)
        )
        dz_raw, _coefficients = self.flow(flat_z, flat_tokens, directions, gates)
        dz = self.metric.naturalize(
            dz_raw,
            metric_diag,
            metric_u,
            strength=self._naturalization_strength(global_step),
            damping=self.config.metric_damping,
        )
        if self.config.directional_cumsum_step_mode == "velocity":
            flat_next = self.updater(flat_z, dz)
        else:
            flat_next = self._directional_candidate_step(flat_z, dz, directions, gates, metric_diag, metric_u)
        local_delta = (flat_next - flat_z).reshape(batch, block_len, -1)
        return self._bound_state(z_start.unsqueeze(1) + torch.cumsum(local_delta, dim=1))

    def _directional_cumsum_block_base(
        self,
        z_start: torch.Tensor,
        token_embeddings: torch.Tensor,
        global_step: int | None,
        block_index: int = 0,
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
        batch, block_len, d_token = token_embeddings.shape
        flat_z = z_start.unsqueeze(1).expand(-1, block_len, -1).reshape(batch * block_len, z_start.shape[-1])
        flat_tokens = token_embeddings.reshape(batch * block_len, d_token)
        base_directions, base_gates = self.direction_field(z_start)
        base_metric_diag, base_metric_u = self.metric(z_start)
        base_risk = self.risk(z_start)
        directions = (
            base_directions.unsqueeze(1)
            .expand(-1, block_len, -1, -1)
            .reshape(batch * block_len, self.config.n_directions, self.config.d_state)
        )
        gates = (
            base_gates.unsqueeze(1)
            .expand(-1, block_len, -1)
            .reshape(batch * block_len, self.config.n_directions)
        )
        metric_diag = (
            base_metric_diag.unsqueeze(1)
            .expand(-1, block_len, -1)
            .reshape(batch * block_len, self.config.d_state)
        )
        metric_u = (
            base_metric_u.unsqueeze(1)
            .expand(-1, block_len, -1, -1)
            .reshape(batch * block_len, self.config.d_state, self.config.metric_rank)
        )
        risk_mass_flat = base_risk["risk_mass"].unsqueeze(1).expand(-1, block_len).reshape(batch * block_len)
        dz_raw, _coefficients = self.flow(flat_z, flat_tokens, directions, gates)
        dz = self.metric.naturalize(
            dz_raw,
            metric_diag,
            metric_u,
            strength=self._naturalization_strength(global_step),
            damping=self.config.metric_damping,
        )
        if self.config.directional_cumsum_step_mode == "velocity":
            flat_next = self.updater(flat_z, dz)
        else:
            flat_next = self._directional_candidate_step(flat_z, dz, directions, gates, metric_diag, metric_u)
        local_delta = (flat_next - flat_z).reshape(batch, block_len, -1)
        states = self._bound_state(z_start.unsqueeze(1) + torch.cumsum(local_delta, dim=1))
        states = self._apply_endpoint_correction(z_start, token_embeddings, states, global_step)
        states = self._apply_block_fixed_point(z_start, token_embeddings, states, global_step)
        states = self._apply_block_anderson(z_start, token_embeddings, states, global_step, block_index)
        velocity = (flat_next - flat_z) / max(self.config.dt, 1e-8)
        action = self.metric.metric_energy(flat_z, velocity, metric_diag, metric_u, risk_mass=risk_mass_flat).reshape(batch, block_len)
        dim = gates.sum(dim=-1).reshape(batch, block_len)
        risk_mass = risk_mass_flat.reshape(batch, block_len)
        if self.local_mixer is not None:
            states = self._bound_state(self.local_mixer(z_start, states, token_embeddings, local_delta, dim, risk_mass))
        consistency = self._block_consistency(z_start, token_embeddings, states, global_step)
        sampled_consistency = self._sampled_block_consistency(
            z_start,
            token_embeddings,
            states,
            global_step,
            block_index,
        )
        entropy = dimension_entropy(gates)
        metric_reg = metric_diag.pow(2).mean() + metric_u.pow(2).mean()
        metric_diag_step = metric_diag.reshape(batch, block_len, -1)
        condition = self.metric.condition_proxy(metric_diag, metric_u).reshape(batch, block_len)
        active_050 = (gates > 0.50).float().mean(dim=-1).reshape(batch, block_len)
        u_norm = (
            metric_u.norm(dim=(1, 2)).reshape(batch, block_len)
            if metric_u.numel()
            else metric_diag.new_zeros(batch, block_len)
        )
        return (
            states,
            action,
            dim,
            entropy,
            metric_reg,
            metric_diag_step,
            condition,
            active_050,
            u_norm,
            risk_mass,
            consistency,
            sampled_consistency,
        )

    def _apply_endpoint_correction(
        self,
        z_start: torch.Tensor,
        token_embeddings: torch.Tensor,
        states: torch.Tensor,
        global_step: int | None,
    ) -> torch.Tensor:
        weight = float(self.config.directional_endpoint_correction_weight)
        if weight <= 0.0:
            return states
        previous = z_start if states.shape[1] == 1 else states[:, -2]
        corrected_endpoint = self.directional_transition(previous, token_embeddings[:, -1], global_step)
        correction = corrected_endpoint - states[:, -1]
        positions = torch.linspace(
            1.0 / states.shape[1],
            1.0,
            states.shape[1],
            device=states.device,
            dtype=states.dtype,
        )
        power = float(self.config.directional_endpoint_correction_power)
        weights = positions.pow(power).view(1, -1, 1)
        return self._bound_state(states + weight * weights * correction.unsqueeze(1))

    def _apply_block_fixed_point(
        self,
        z_start: torch.Tensor,
        token_embeddings: torch.Tensor,
        states: torch.Tensor,
        global_step: int | None,
    ) -> torch.Tensor:
        iterations = int(self.config.directional_fixed_point_iterations)
        if iterations <= 0:
            return states

        def transition(z: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
            return self.directional_transition(z, token_embedding, global_step)

        solved, _residuals = fixed_point_solve(
            transition,
            z_start,
            token_embeddings,
            iterations=iterations,
            relaxation=self.config.directional_fixed_point_relaxation,
            initial_trajectory=states,
        )
        return self._bound_state(solved)

    def _apply_block_anderson(
        self,
        z_start: torch.Tensor,
        token_embeddings: torch.Tensor,
        states: torch.Tensor,
        global_step: int | None,
        block_index: int = 0,
    ) -> torch.Tensor:
        iterations = int(self.config.directional_anderson_iterations)
        if iterations <= 0:
            return states
        stride = int(self.config.directional_anderson_block_stride)
        if stride > 1 and block_index % stride != 0:
            return states

        def transition(z: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
            if self.config.directional_anderson_transition_mode == "velocity":
                return self.directional_velocity_transition(z, token_embedding, global_step)
            return self.directional_transition(z, token_embedding, global_step)

        if self.config.directional_anderson_scope == "endpoint":
            previous = z_start if states.shape[1] == 1 else states[:, -2]
            endpoint, _residuals = causal_anderson_solve(
                transition,
                previous,
                token_embeddings[:, -1:],
                iterations=iterations,
                history_size=self.config.directional_anderson_history_size,
                ridge=self.config.directional_anderson_ridge,
                relaxation=self.config.directional_anderson_relaxation,
                initial_trajectory=states[:, -1:],
            )
            return self._bound_state(torch.cat([states[:, :-1], endpoint], dim=1))

        solved, _residuals = causal_anderson_solve(
            transition,
            z_start,
            token_embeddings,
            iterations=iterations,
            history_size=self.config.directional_anderson_history_size,
            ridge=self.config.directional_anderson_ridge,
            relaxation=self.config.directional_anderson_relaxation,
            initial_trajectory=states,
        )
        return self._bound_state(solved)

    def _block_consistency(
        self,
        z_start: torch.Tensor,
        token_embeddings: torch.Tensor,
        states: torch.Tensor,
        global_step: int | None,
    ) -> torch.Tensor | None:
        if self.config.lambda_block_consistency <= 0.0:
            return None

        def transition(z: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
            return self.directional_transition(z, token_embedding, global_step)

        target = sequential_rollout(transition, z_start, token_embeddings).detach()
        return self.config.block_consistency_weight * (states - target).pow(2).mean(dim=-1)

    def _sampled_block_consistency(
        self,
        z_start: torch.Tensor,
        token_embeddings: torch.Tensor,
        states: torch.Tensor,
        global_step: int | None,
        block_index: int = 0,
    ) -> torch.Tensor | None:
        if self.config.lambda_sampled_block_consistency <= 0.0:
            return None

        interval = max(int(self.config.sampled_block_consistency_interval), 1)
        if interval > 1 and block_index % interval != 0:
            return None

        block_len = token_embeddings.shape[1]
        local_size = min(max(int(self.config.sampled_block_consistency_local_size), 1), block_len)
        segment_count = max((block_len + local_size - 1) // local_size, 1)
        segment_index = (block_index // interval) % segment_count
        start = min(segment_index * local_size, max(block_len - local_size, 0))
        end = min(start + local_size, block_len)
        if end <= start:
            return None

        teacher_start = z_start if start == 0 else states[:, start - 1].detach()
        teacher_tokens = token_embeddings[:, start:end].detach()
        fast_segment = states[:, start:end]

        def transition(z: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
            if self.config.sampled_block_consistency_teacher_mode == "velocity":
                return self.directional_velocity_transition(z, token_embedding, global_step)
            return self.directional_transition(z, token_embedding, global_step)

        with torch.no_grad():
            target = sequential_rollout(transition, teacher_start.detach(), teacher_tokens)
        return self.config.block_consistency_weight * (fast_segment - target).pow(2).mean(dim=-1)

    def directional_transition(
        self,
        z: torch.Tensor,
        token_embedding: torch.Tensor,
        global_step: int | None = None,
    ) -> torch.Tensor:
        """Apply one target-free directional-candidate DRM transition."""

        directions, gates = self.direction_field(z)
        metric_diag, metric_u = self.metric(z)
        dz_raw, _coefficients = self.flow(z, token_embedding, directions, gates)
        dz = self.metric.naturalize(
            dz_raw,
            metric_diag,
            metric_u,
            strength=self._naturalization_strength(global_step),
            damping=self.config.metric_damping,
        )
        return self._directional_candidate_step(z, dz, directions, gates, metric_diag, metric_u)

    def directional_velocity_transition(
        self,
        z: torch.Tensor,
        token_embedding: torch.Tensor,
        global_step: int | None = None,
    ) -> torch.Tensor:
        """Apply the cheaper naturalized-velocity transition used by fast Anderson probes."""

        directions, gates = self.direction_field(z)
        metric_diag, metric_u = self.metric(z)
        dz_raw, _coefficients = self.flow(z, token_embedding, directions, gates)
        dz = self.metric.naturalize(
            dz_raw,
            metric_diag,
            metric_u,
            strength=self._naturalization_strength(global_step),
            damping=self.config.metric_damping,
        )
        return self._bound_state(self.updater(z, dz))

    def _bound_state(self, z: torch.Tensor) -> torch.Tensor:
        if not self.config.bounded_state:
            return z
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        clip = torch.clamp(self.config.state_clip_norm / norm, max=1.0)
        z = z * clip
        return self.config.state_clip_norm * torch.tanh(z / self.config.state_clip_norm)

    def _geodesic_step(
        self,
        z: torch.Tensor,
        dz_context: torch.Tensor,
        metric_diag: torch.Tensor,
        metric_u: torch.Tensor,
    ) -> torch.Tensor:
        """Refine the local DRM proposal by minimizing target-free geometric energy.

        This is an experimental solver path. The supervised next-token target is
        intentionally not part of the inner energy; CE remains the outer loss.
        """

        z_start = z
        z_anchor = self.updater(z_start, dz_context)
        z_candidate = z_anchor
        dt = max(self.config.dt, 1e-8)
        create_graph = torch.is_grad_enabled() and self.training

        for _ in range(self.config.geodesic_solver_steps):
            with torch.enable_grad():
                z_candidate = (
                    z_candidate.detach().requires_grad_(True)
                    if not create_graph
                    else z_candidate.requires_grad_(True)
                )
                displacement = z_candidate - z_start
                anchor_loss = (z_candidate - z_anchor).pow(2).sum(dim=-1)
                metric_loss = self.metric.metric_energy(
                    z_start,
                    displacement / dt,
                    metric_diag,
                    metric_u,
                )
                risk_loss = self.risk(z_candidate)["risk_mass"]
                energy = (
                    self.config.geodesic_anchor_weight * anchor_loss
                    + self.config.geodesic_metric_weight * metric_loss
                    + self.config.geodesic_risk_weight * risk_loss
                ).sum()
                grad = torch.autograd.grad(energy, z_candidate, create_graph=create_graph)[0]
            z_candidate = self._bound_state(z_candidate - self.config.geodesic_lr * grad)
        return z_candidate if create_graph else z_candidate.detach()

    def _directional_candidate_step(
        self,
        z: torch.Tensor,
        dz_context: torch.Tensor,
        directions: torch.Tensor,
        gates: torch.Tensor,
        metric_diag: torch.Tensor,
        metric_u: torch.Tensor,
    ) -> torch.Tensor:
        """Choose the next state from parallel directional endpoint candidates."""

        dt = max(self.config.dt, 1e-8)
        z_anchor = self.updater(z, dz_context)
        candidate_delta = dt * self.config.directional_candidate_scale * gates.unsqueeze(-1) * directions
        anchor_delta = (z_anchor - z).unsqueeze(1)
        candidates = self._bound_state(z.unsqueeze(1) + candidate_delta + anchor_delta)
        candidate_velocity = (candidates - z.unsqueeze(1)) / dt

        diag_energy = (metric_diag.unsqueeze(1) * candidate_velocity.pow(2)).sum(dim=-1)
        if metric_u.shape[-1] > 0:
            low_rank = torch.einsum("bnr,bnrk->bnk", candidate_velocity, metric_u.unsqueeze(1))
            metric_energy = diag_energy + low_rank.pow(2).sum(dim=-1)
        else:
            metric_energy = diag_energy
        candidate_risk = self.risk(candidates.reshape(-1, candidates.shape[-1]))["risk_mass"].view(candidates.shape[:2])
        anchor_energy = (candidates - z_anchor.unsqueeze(1)).pow(2).sum(dim=-1)
        scores = -(
            self.config.geodesic_metric_weight * metric_energy
            + self.config.geodesic_risk_weight * candidate_risk
            + self.config.geodesic_anchor_weight * anchor_energy
        )
        weights = torch.softmax(scores / self.config.directional_candidate_temperature, dim=-1)
        return torch.einsum("bn,bnd->bd", weights, candidates)

    def _naturalization_strength(self, global_step: int | None) -> float:
        if not self.config.use_metric_naturalization:
            return 0.0
        max_strength = self.config.metric_naturalization_strength
        warmup = self.config.metric_naturalization_warmup_steps
        if global_step is None or warmup <= 0:
            return max_strength
        return max_strength * min(max(global_step, 0) / warmup, 1.0)
