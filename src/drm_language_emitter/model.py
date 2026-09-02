from __future__ import annotations

import warnings
from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .addressable_memory import AddressableMemory
from .asm_z_core import ASMZCore
from .asm_z_forward import ASMZForwardMixin
from .config import DRMConfig
from .direction_field import DirectionField
from .directional_blocks import DirectionalBlocksMixin
from .directional_forward import DirectionalForwardMixin
from .directional_solvers import DirectionalSolversMixin
from .dynamics import DRMFlow, StateUpdater
from .emitter import LanguageEmitter, TokenEmbedding
from .fast_weight_memory import FastWeightMemory
from .geometric_steps import GeometricStepsMixin
from .inference import InferenceMixin
from .losses import (
    combine_losses,
    dimension_entropy,
    metric_diversity,
    next_token_cross_entropy,
    recurrence_proxy,
    stability_proxy,
)
from .metric import RelationalMetric
from .model_components import (
    CausalLocalMixer,
    DirectStateTransition,
    DRMRefinementLayer,
    DRMStateInitializer,
    SelectiveStateMemory,
    TokenStateResidual,
)
from .risk import RiskField
from .selective_control import SelectiveControlMixin


def _seeded_module(config: DRMConfig, offset: int, factory):
    """Initialize a component from a stable stream independent of optional peers."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.seed) + offset)
        return factory()


class DRMEmitterModel(
    ASMZForwardMixin,
    DirectionalForwardMixin,
    DirectionalBlocksMixin,
    DirectionalSolversMixin,
    GeometricStepsMixin,
    InferenceMixin,
    SelectiveControlMixin,
    nn.Module,
):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = _seeded_module(config, 101, lambda: TokenEmbedding(config))
        self.initializer = _seeded_module(config, 102, lambda: DRMStateInitializer(config))
        self.direction_field = (
            _seeded_module(config, 103, lambda: DirectionField(config))
            if config.use_drm_geometry and config.use_direction_field and config.sequence_mode != "asm_z"
            else None
        )
        self.metric = (
            _seeded_module(config, 104, lambda: RelationalMetric(config))
            if config.use_drm_geometry and config.use_relational_metric and config.sequence_mode != "asm_z"
            else None
        )
        self.flow = (
            _seeded_module(config, 105, lambda: DRMFlow(config))
            if config.use_drm_geometry and config.use_direction_field and config.sequence_mode != "asm_z"
            else None
        )
        self.direct_transition = (
            _seeded_module(config, 112, lambda: DirectStateTransition(config))
            if config.use_drm_geometry and not config.use_direction_field and config.sequence_mode != "asm_z"
            else None
        )
        self.asm_z_core = (
            _seeded_module(config, 115, lambda: ASMZCore(config))
            if config.sequence_mode == "asm_z"
            else None
        )
        self.updater = None if config.sequence_mode == "asm_z" else StateUpdater(config)
        self.risk = (
            _seeded_module(config, 106, lambda: RiskField(config))
            if config.use_drm_geometry and config.sequence_mode != "asm_z"
            else None
        )
        self.emitter = _seeded_module(config, 107, lambda: LanguageEmitter(config))
        self.local_mixer = (
            _seeded_module(config, 108, lambda: CausalLocalMixer(config))
            if config.directional_local_mixer != "none"
            else None
        )
        self.token_state_residual = (
            _seeded_module(config, 109, lambda: TokenStateResidual(config))
            if config.token_state_residual
            else None
        )
        self.selective_memory = (
            _seeded_module(config, 110, lambda: SelectiveStateMemory(config))
            if config.selective_memory
            else None
        )
        self.addressable_memory = (
            _seeded_module(
                config,
                113,
                lambda: (
                    FastWeightMemory(config)
                    if config.addressable_memory_backend == "fast_weight"
                    else AddressableMemory(config)
                ),
            )
            if config.addressable_memory
            else None
        )
        self.refinement_layers = _seeded_module(
            config,
            111,
            lambda: nn.ModuleList(
                DRMRefinementLayer(config) for _ in range(config.directional_refinement_layers)
            ),
        )
        self.variable_rank_core = None
        if config.variable_rank_mode != "off":
            from aletheion_state_models.geometry.variable_rank.block_core import (
                VariableRankBlockCore,
            )

            self.variable_rank_core = _seeded_module(
                config,
                114,
                lambda: VariableRankBlockCore(
                    config.d_token,
                    config.d_state,
                    threshold=config.variable_rank_threshold,
                    minimum_rank=config.variable_rank_min_rank,
                    estimator=(
                        "ste" if config.variable_rank_mode in {"phase2_input_ste", "phase3a1_projected"} else "hard"
                    ),
                ),
            )
        self._compiled_forward = None
        if config.use_torch_compile and hasattr(torch, "compile"):
            try:
                self._compiled_forward = torch.compile(self._forward_impl)
            except Exception as exc:
                warnings.warn(
                    f"torch.compile initialization failed; using eager execution: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
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
        if self.config.sequence_mode == "asm_z":
            return self._asm_z_forward(
                input_ids,
                targets,
                return_states,
                collect_diagnostics,
                initial_state=z,
            )
        token_embeddings = self.token_embedding(input_ids)
        if not self.config.use_drm_geometry:
            return self._forward_selective_control(
                z,
                token_embeddings,
                targets,
                return_states,
                collect_diagnostics,
            )
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
            except Exception as exc:
                warnings.warn(
                    f"compiled forward failed; disabling torch.compile and retrying eagerly: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._compiled_forward = None
        return self._forward_impl(input_ids, targets, return_states, global_step, collect_diagnostics)

    def state_dict_with_config(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.config.schema_version),
            "config": asdict(self.config),
            "model": self.state_dict(),
        }
