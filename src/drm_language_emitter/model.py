from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .config import DRMConfig
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


class GeometryEncoder(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        h = config.hidden_size
        self.net = nn.Sequential(
            nn.Linear(config.d_state, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.GELU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DRMEmitterModel(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config)
        self.initializer = DRMStateInitializer(config)
        self.geometry_encoder = GeometryEncoder(config) if config.use_shared_geometry_trunk else None
        self.direction_field = DirectionField(config)
        self.metric = RelationalMetric(config)
        self.flow = DRMFlow(config)
        self.updater = StateUpdater(config)
        self.risk = RiskField(config)
        self.emitter = LanguageEmitter(config)
        self._compiled_step = None
        if config.compile_drm_step and hasattr(torch, "compile"):
            try:
                self._compiled_step = torch.compile(self._drm_step)
            except Exception:
                self._compiled_step = None
        self._compiled_forward = None
        if config.use_torch_compile and hasattr(torch, "compile"):
            try:
                self._compiled_forward = torch.compile(self._forward_impl)
            except Exception:
                self._compiled_forward = None

    def _geometry(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.geometry_encoder(z) if self.geometry_encoder is not None else None
        directions, gates = self.direction_field(z, hidden)
        metric_diag, metric_u = self.metric(z, hidden)
        risk = self.risk(z)
        return directions, gates, metric_diag, metric_u, risk

    def _drm_step(
        self,
        z: torch.Tensor,
        e_t: torch.Tensor,
        directions: torch.Tensor,
        gates: torch.Tensor,
        metric_diag: torch.Tensor,
        metric_u: torch.Tensor,
        naturalization_strength: float,
        apply_naturalization: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dz_raw, coefficients = self.flow(z, e_t, directions, gates)
        if apply_naturalization:
            dz = self.metric.naturalize(
                dz_raw,
                metric_diag,
                metric_u,
                strength=naturalization_strength,
                damping=self.config.metric_damping,
            )
        else:
            dz = dz_raw
        return self.updater(z, dz), dz, coefficients

    def _run_step(
        self,
        z: torch.Tensor,
        e_t: torch.Tensor,
        directions: torch.Tensor,
        gates: torch.Tensor,
        metric_diag: torch.Tensor,
        metric_u: torch.Tensor,
        naturalization_strength: float,
        apply_naturalization: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._compiled_step is not None:
            try:
                return self._compiled_step(
                    z,
                    e_t,
                    directions,
                    gates,
                    metric_diag,
                    metric_u,
                    naturalization_strength,
                    apply_naturalization,
                )
            except Exception:
                self._compiled_step = None
        return self._drm_step(
            z,
            e_t,
            directions,
            gates,
            metric_diag,
            metric_u,
            naturalization_strength,
            apply_naturalization,
        )

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
        need_states = return_states or collect_diagnostics or self.config.lambda_recurrence != 0
        need_stability = collect_diagnostics or self.config.lambda_stability != 0
        need_metric_diversity = collect_diagnostics or self.config.lambda_metric_diversity != 0
        emission_states = []
        states = []
        metric_diag_steps = []
        active_025_values = []
        active_075_values = []
        active_090_values = []
        gate_min_values = []
        gate_max_values = []
        gate_flat_values = []
        risk_values = []
        action_sum: torch.Tensor | None = None
        dim_sum: torch.Tensor | None = None
        dim_square_sum: torch.Tensor | None = None
        entropy_sum: torch.Tensor | None = None
        metric_reg_sum: torch.Tensor | None = None
        condition_sum: torch.Tensor | None = None
        active_050_sum: torch.Tensor | None = None
        u_norm_sum: torch.Tensor | None = None
        u_norm_square_sum: torch.Tensor | None = None
        u_floor_sum: torch.Tensor | None = None
        sampled_value_count = 0
        sampled_step_count = 0
        naturalization_strength = self._naturalization_strength(global_step)
        geometry_interval = max(int(self.config.geometry_update_interval), 1)
        aux_loss_interval = max(int(self.config.aux_loss_interval), 1)
        naturalization_interval = max(int(self.config.naturalization_interval), 1)
        forward_chunk_size = int(self.config.forward_chunk_size)
        geometry_tick = 0
        cached_directions: torch.Tensor | None = None
        cached_gates: torch.Tensor | None = None
        cached_metric_diag: torch.Tensor | None = None
        cached_metric_u: torch.Tensor | None = None
        cached_risk: dict[str, torch.Tensor] | None = None
        truncate_interval = int(self.config.bptt_truncate_interval)

        chunk_size = seq_len if forward_chunk_size <= 0 else max(forward_chunk_size, 1)
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            for t in range(chunk_start, chunk_end):
                e_t = token_embeddings[:, t]
                for _ in range(self.config.n_flow_steps):
                    if geometry_tick % geometry_interval == 0 or cached_directions is None:
                        cached_directions, cached_gates, cached_metric_diag, cached_metric_u, cached_risk = self._geometry(z)
                    directions = cached_directions
                    gates = cached_gates
                    metric_diag = cached_metric_diag
                    metric_u = cached_metric_u
                    risk = cached_risk
                    assert directions is not None
                    assert gates is not None
                    assert metric_diag is not None
                    assert metric_u is not None
                    assert risk is not None
                    apply_naturalization = naturalization_strength > 0 and geometry_tick % naturalization_interval == 0
                    z_before = z
                    z, dz, _coefficients = self._run_step(
                        z,
                        e_t,
                        directions,
                        gates,
                        metric_diag,
                        metric_u,
                        naturalization_strength,
                        apply_naturalization,
                    )
                    sample_aux = geometry_tick % aux_loss_interval == 0
                    if sample_aux:
                        energy = self.metric.metric_energy(
                            z_before, dz, metric_diag, metric_u, risk_mass=risk["risk_mass"]
                        )
                        dim_values = gates.sum(dim=-1)
                        entropy_value = dimension_entropy(gates)
                        u_norm = metric_u.norm(dim=(1, 2)) if metric_u.numel() else metric_diag.new_zeros(batch)
                        metric_reg_value = metric_diag.pow(2).mean() + metric_u.pow(2).mean()
                        condition_values = self.metric.condition_proxy(metric_diag, metric_u)
                        active_050_values = (gates > 0.50).float().mean(dim=-1)

                        action_sum = energy.sum() if action_sum is None else action_sum + energy.sum()
                        dim_sum = dim_values.sum() if dim_sum is None else dim_sum + dim_values.sum()
                        dim_square_sum = dim_values.pow(2).sum() if dim_square_sum is None else dim_square_sum + dim_values.pow(2).sum()
                        entropy_sum = entropy_value if entropy_sum is None else entropy_sum + entropy_value
                        metric_reg_sum = metric_reg_value if metric_reg_sum is None else metric_reg_sum + metric_reg_value
                        condition_sum = condition_values.sum() if condition_sum is None else condition_sum + condition_values.sum()
                        active_050_sum = active_050_values.sum() if active_050_sum is None else active_050_sum + active_050_values.sum()
                        u_norm_sum = u_norm.sum() if u_norm_sum is None else u_norm_sum + u_norm.sum()
                        u_norm_square_sum = u_norm.pow(2).sum() if u_norm_square_sum is None else u_norm_square_sum + u_norm.pow(2).sum()
                        if self.config.metric_rank > 0:
                            floor_values = (self.config.metric_u_min_norm - u_norm).clamp_min(0.0).pow(2)
                            u_floor_sum = floor_values.sum() if u_floor_sum is None else u_floor_sum + floor_values.sum()
                        if need_metric_diversity:
                            metric_diag_steps.append(metric_diag)
                        sampled_value_count += batch
                        sampled_step_count += 1
                    if collect_diagnostics:
                        active_025_values.append((gates > 0.25).float().mean(dim=-1))
                        active_075_values.append((gates > 0.75).float().mean(dim=-1))
                        active_090_values.append((gates > 0.90).float().mean(dim=-1))
                        gate_min_values.append(gates.min(dim=-1).values)
                        gate_max_values.append(gates.max(dim=-1).values)
                        gate_flat_values.append(gates.reshape(-1))
                    if collect_diagnostics or self.config.lambda_blindspot != 0:
                        risk_values.append(risk["risk_mass"])
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
            if forward_chunk_size > 0:
                cached_directions = None
                cached_gates = None
                cached_metric_diag = None
                cached_metric_u = None
                cached_risk = None

        emission_state_tensor = torch.stack(emission_states, dim=1)
        logits = self.emitter(emission_state_tensor)
        state_tensor = torch.stack(states, dim=1) if need_states else None
        metric_diag_tensor = torch.stack(metric_diag_steps, dim=1) if need_metric_diversity else None
        if sampled_value_count == 0 or action_sum is None:
            raise RuntimeError("DRM forward did not sample any auxiliary loss steps")
        action_loss = action_sum / sampled_value_count
        dim_sparsity = dim_sum / sampled_value_count
        dim_variance = (dim_square_sum / sampled_value_count - dim_sparsity.pow(2)).clamp_min(0.0)
        dim_std_value = dim_variance.sqrt()
        dim_entropy_value = entropy_sum / sampled_step_count
        metric_reg = metric_reg_sum / sampled_step_count
        u_norm_mean_for_variance = u_norm_sum / sampled_value_count
        metric_u_norm_variance = (u_norm_square_sum / sampled_value_count - u_norm_mean_for_variance.pow(2)).clamp_min(0.0)
        if self.config.metric_rank > 0:
            metric_u_floor_loss = u_floor_sum / sampled_value_count if u_floor_sum is not None else action_loss.new_tensor(0.0)
        else:
            metric_u_floor_loss = action_loss.new_tensor(0.0)
        metric_div_value = metric_diversity(metric_diag_tensor) if metric_diag_tensor is not None else action_loss.new_tensor(0.0)
        recurrence_value = recurrence_proxy(state_tensor) if state_tensor is not None else action_loss.new_tensor(0.0)
        stability_value = stability_proxy(logits) if need_stability else action_loss.new_tensor(0.0)
        blindspot_value = torch.stack(risk_values, dim=1).mean() if risk_values else action_loss.new_tensor(0.0)
        hard_active_050_value = active_050_sum / sampled_value_count
        soft_active_value = dim_sparsity / self.config.n_directions
        condition_value = condition_sum / sampled_value_count
        metric_u_norm_value = u_norm_mean_for_variance
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
            "metric_U_variance": metric_u_norm_variance,
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

    def _naturalization_strength(self, global_step: int | None) -> float:
        if not self.config.use_metric_naturalization:
            return 0.0
        max_strength = self.config.metric_naturalization_strength
        warmup = self.config.metric_naturalization_warmup_steps
        if global_step is None or warmup <= 0:
            return max_strength
        return max_strength * min(max(global_step, 0) / warmup, 1.0)
