from __future__ import annotations

import torch

from .losses import dimension_entropy


class DirectionalBlocksMixin:
    def _directional_superblock_cumsum_block(
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
        local_size = int(self.config.directional_superblock_local_size)
        block_len = token_embeddings.shape[1]
        if block_len <= local_size or block_len % local_size != 0:
            return self._directional_cumsum_block_base(
                z_start,
                token_embeddings,
                global_step,
                block_index,
                collect_diagnostics,
            )

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
            flat_gates,
            flat_u_norm,
            flat_risk,
            flat_consistency,
            flat_sampled_consistency,
        ) = self._directional_cumsum_block_base(
            flat_starts,
            flat_tokens,
            global_step,
            block_index,
            collect_diagnostics,
        )

        states = flat_states.reshape(batch, segment_count, local_size, -1).reshape(batch, block_len, -1)
        action = flat_action.reshape(batch, segment_count, local_size).reshape(batch, block_len)
        dim = flat_dim.reshape(batch, segment_count, local_size).reshape(batch, block_len)
        metric_diag = flat_metric_diag.reshape(batch, segment_count, local_size, -1).reshape(batch, block_len, -1)
        condition = flat_condition.reshape(batch, segment_count, local_size).reshape(batch, block_len)
        active = flat_active.reshape(batch, segment_count, local_size).reshape(batch, block_len)
        gate_values = flat_gates.reshape(
            batch,
            segment_count,
            local_size,
            self.config.n_directions,
        ).reshape(batch, block_len, self.config.n_directions)
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
            gate_values,
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
        _dz_raw, coefficients = self.flow(flat_z, flat_tokens, directions, gates)
        dz = self._compose_metric_and_directions(
            base_directions,
            base_gates,
            coefficients.reshape(batch, block_len, self.config.n_directions),
            base_metric_diag,
            base_metric_u,
            global_step,
        ).reshape(batch * block_len, self.config.d_state)
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
        batch, block_len, d_token = token_embeddings.shape
        flat_z = z_start.unsqueeze(1).expand(-1, block_len, -1).reshape(batch * block_len, z_start.shape[-1])
        flat_tokens = token_embeddings.reshape(batch * block_len, d_token)
        if self.direction_field is not None:
            base_directions, base_gates = self.direction_field(z_start)
            directions = (
                base_directions.unsqueeze(1)
                .expand(-1, block_len, -1, -1)
                .reshape(
                    batch * block_len,
                    self.config.n_directions,
                    self.config.d_state,
                )
            )
            gates = (
                base_gates.unsqueeze(1)
                .expand(-1, block_len, -1)
                .reshape(batch * block_len, self.config.n_directions)
            )
            assert self.flow is not None
            dz_raw, coefficients = self.flow(
                flat_z,
                flat_tokens,
                directions,
                gates,
            )
        else:
            assert self.direct_transition is not None
            directions = flat_z.new_empty(
                batch * block_len,
                0,
                self.config.d_state,
            )
            gates = flat_z.new_zeros(
                batch * block_len,
                self.config.n_directions,
            )
            dz_raw = self.direct_transition(flat_z, flat_tokens)

        if self.metric is not None:
            base_metric_diag, base_metric_u = self.metric(z_start)
            metric_diag = (
                base_metric_diag.unsqueeze(1)
                .expand(-1, block_len, -1)
                .reshape(batch * block_len, self.config.d_state)
            )
            metric_u = (
                base_metric_u.unsqueeze(1)
                .expand(-1, block_len, -1, -1)
                .reshape(
                    batch * block_len,
                    self.config.d_state,
                    self.config.metric_rank,
                )
            )
            if self.direction_field is not None:
                dz = self._compose_metric_and_directions(
                    base_directions,
                    base_gates,
                    coefficients.reshape(
                        batch,
                        block_len,
                        self.config.n_directions,
                    ),
                    base_metric_diag,
                    base_metric_u,
                    global_step,
                ).reshape(batch * block_len, self.config.d_state)
            else:
                dz = self.metric.naturalize(
                    dz_raw,
                    metric_diag,
                    metric_u,
                    strength=self._naturalization_strength(global_step),
                    damping=self.config.metric_damping,
                )
        else:
            metric_diag = flat_z.new_ones(
                batch * block_len,
                self.config.d_state,
            )
            metric_u = flat_z.new_zeros(
                batch * block_len,
                self.config.d_state,
                0,
            )
            dz = dz_raw

        assert self.risk is not None
        need_risk = (
            self.config.use_powerlaw_risk
            or self.config.lambda_blindspot != 0
            or collect_diagnostics
        )
        if need_risk:
            base_risk = self.risk(z_start)
            risk_mass_flat = (
                base_risk["risk_mass"]
                .unsqueeze(1)
                .expand(-1, block_len)
                .reshape(batch * block_len)
            )
        else:
            risk_mass_flat = flat_z.new_zeros(batch * block_len)
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
        need_action = self.config.lambda_action != 0 or collect_diagnostics
        if need_action and self.metric is not None:
            action = self.metric.metric_energy(
                flat_z,
                velocity,
                metric_diag,
                metric_u,
                risk_mass=risk_mass_flat,
            ).reshape(batch, block_len)
        elif need_action:
            action = velocity.pow(2).sum(dim=-1).reshape(batch, block_len)
        else:
            action = velocity.new_zeros(batch, block_len)
        dim = gates.sum(dim=-1).reshape(batch, block_len)
        risk_mass = risk_mass_flat.reshape(batch, block_len)
        if self.local_mixer is not None:
            states = self._bound_state(self.local_mixer(z_start, states, token_embeddings, local_delta, dim, risk_mass))
        if self.token_state_residual is not None:
            states = self._bound_state(self.token_state_residual(states, token_embeddings))
        if self.selective_memory is not None:
            states = self._bound_state(
                self.selective_memory(z_start, states, token_embeddings)
            )
        for refinement_layer in self.refinement_layers:
            states = self._bound_state(
                refinement_layer(
                    z_start,
                    states,
                    token_embeddings,
                    self._naturalization_strength(global_step),
                )
            )
        consistency = self._block_consistency(z_start, token_embeddings, states, global_step)
        sampled_consistency = self._sampled_block_consistency(
            z_start,
            token_embeddings,
            states,
            global_step,
            block_index,
        )
        need_dim = (
            self.config.lambda_dim_sparsity != 0
            or self.config.lambda_dim_entropy != 0
            or self.config.lambda_dim_variance != 0
            or self.config.lambda_active_fraction != 0
            or collect_diagnostics
        )
        entropy = dimension_entropy(gates) if need_dim else gates.new_tensor(0.0)
        need_metric_reg = self.config.lambda_metric_reg != 0 or collect_diagnostics
        metric_reg = (
            metric_diag.pow(2).mean() + metric_u.pow(2).mean()
            if self.metric is not None and need_metric_reg
            else metric_diag.new_tensor(0.0)
        )
        metric_diag_step = metric_diag.reshape(batch, block_len, -1)
        need_condition = self.config.lambda_condition != 0 or collect_diagnostics
        condition = (
            self.metric.condition_proxy(metric_diag, metric_u).reshape(
                batch,
                block_len,
            )
            if self.metric is not None and need_condition
            else metric_diag.new_ones(batch, block_len)
        )
        active_050 = (
            (gates > 0.50).float().mean(dim=-1).reshape(batch, block_len)
            if collect_diagnostics
            else metric_diag.new_zeros(batch, block_len)
        )
        gate_values = gates.reshape(batch, block_len, self.config.n_directions)
        need_u_norm = (
            self.config.lambda_metric_u_floor != 0
            or self.config.lambda_metric_u_target != 0
            or collect_diagnostics
        )
        u_norm = (
            metric_u.norm(dim=(1, 2)).reshape(batch, block_len)
            if metric_u.numel() and need_u_norm
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
            gate_values,
            u_norm,
            risk_mass,
            consistency,
            sampled_consistency,
        )
