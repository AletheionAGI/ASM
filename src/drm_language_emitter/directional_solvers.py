from __future__ import annotations

import torch

from .deer import causal_anderson_solve, fixed_point_solve, sequential_rollout


class DirectionalSolversMixin:
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
