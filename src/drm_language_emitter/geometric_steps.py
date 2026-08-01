from __future__ import annotations

import torch


class GeometricStepsMixin:
    @staticmethod
    def _stable_metric_frame(
        directions: torch.Tensor,
        regularized: torch.Tensor,
        identity: torch.Tensor,
        damping: float,
    ) -> torch.Tensor:
        """Solve metric frames per sample with adaptive Cholesky jitter.

        A batched spectral fallback lets one ill-conditioned sample abort the
        entire batch. Retrying only failed samples is both safer and keeps the
        primary lower-triangular frame convention used during training.
        """

        cholesky, info = torch.linalg.cholesky_ex(
            regularized,
            check_errors=False,
        )
        if bool((info == 0).all()):
            return torch.linalg.solve_triangular(
                cholesky,
                directions,
                upper=False,
            )

        frames = []
        eye = identity[0]
        for batch_index in range(regularized.shape[0]):
            matrix = regularized[batch_index]
            vectors = directions[batch_index]
            if int(info[batch_index]) == 0:
                factor = cholesky[batch_index]
                frame = torch.linalg.solve_triangular(
                    factor,
                    vectors,
                    upper=False,
                )
                frames.append(frame)
                continue

            frame = None
            if bool(torch.isfinite(matrix).all()) and bool(
                torch.isfinite(vectors).all()
            ):
                for multiplier in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
                    retry = matrix + (damping * multiplier) * eye
                    factor, retry_info = torch.linalg.cholesky_ex(
                        retry,
                        check_errors=False,
                    )
                    if int(retry_info) == 0:
                        frame = torch.linalg.solve_triangular(
                            factor,
                            vectors,
                            upper=False,
                        )
                        break
            if frame is None:
                # A non-finite or irrecoverable sample must not terminate a long
                # frozen rescore. Falling back to its finite Euclidean directions
                # is conservative and affects only that pathological sample.
                frame = torch.nan_to_num(vectors)
            frames.append(frame)
        return torch.stack(frames)

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

    def _compose_metric_and_directions(
        self,
        directions: torch.Tensor,
        gates: torch.Tensor,
        coefficients: torch.Tensor,
        metric_diag: torch.Tensor,
        metric_u: torch.Tensor,
        global_step: int | None,
    ) -> torch.Tensor:
        """Compose a block's directions and metric without duplicating its Gram matrix.

        Shapes are BxNxD for directions, BxN for gates, BxLxN for
        coefficients, BxD for the diagonal, and BxDxR for the low-rank factor.
        Both metric-first modes preserve the span of the original directions.
        """

        active = coefficients * gates.unsqueeze(1)
        raw = torch.einsum("bln,bnd->bld", active, directions)
        mode = self.config.directional_metric_composition
        strength = self._naturalization_strength(global_step)
        if mode == "post_naturalize":
            batch, block_len, d_state = raw.shape
            expanded_diag = (
                metric_diag.unsqueeze(1)
                .expand(-1, block_len, -1)
                .reshape(batch * block_len, d_state)
            )
            expanded_u = (
                metric_u.unsqueeze(1)
                .expand(-1, block_len, -1, -1)
                .reshape(
                    batch * block_len,
                    d_state,
                    metric_u.shape[-1],
                )
            )
            naturalized = self.metric.naturalize(
                raw.reshape(batch * block_len, d_state),
                expanded_diag,
                expanded_u,
                strength=strength,
                damping=self.config.metric_damping,
            )
            return naturalized.reshape(batch, block_len, d_state)

        solve_dtype = (
            torch.float32
            if directions.dtype in {torch.float16, torch.bfloat16}
            else directions.dtype
        )
        directions_solve = directions.to(solve_dtype)
        diag_solve = metric_diag.to(solve_dtype)
        u_solve = metric_u.to(solve_dtype)
        active_solve = active.to(solve_dtype)
        autocast_device = (
            directions.device.type
            if directions.device.type in {"cuda", "cpu"}
            else "cpu"
        )
        with torch.autocast(device_type=autocast_device, enabled=False):
            gv = diag_solve.unsqueeze(1) * directions_solve
            if u_solve.shape[-1] > 0:
                projected = torch.bmm(directions_solve, u_solve)
                gv = gv + torch.bmm(projected, u_solve.transpose(1, 2))
            gram = torch.bmm(directions_solve, gv.transpose(1, 2))
            gram = 0.5 * (gram + gram.transpose(1, 2))
            identity = torch.eye(
                directions.shape[1],
                device=directions.device,
                dtype=solve_dtype,
            ).unsqueeze(0)
            damping = max(float(self.config.metric_damping), 1e-6)
            regularized = gram + damping * identity

            if mode == "metric_subspace":
                rhs = active_solve.transpose(1, 2)
                solution, info = torch.linalg.solve_ex(regularized, rhs)
                if bool((info != 0).any()):
                    solution = torch.bmm(
                        torch.linalg.pinv(regularized, hermitian=True),
                        rhs,
                    )
                natural_coefficients = solution.transpose(1, 2)
                metric_first = torch.einsum(
                    "bln,bnd->bld",
                    natural_coefficients,
                    directions_solve,
                )
            else:
                orthonormal_directions = self._stable_metric_frame(
                    directions_solve,
                    regularized,
                    identity,
                    damping,
                )
                metric_first = torch.einsum(
                    "bln,bnd->bld",
                    active_solve,
                    orthonormal_directions,
                )

        metric_first = metric_first.to(raw.dtype)
        return (1.0 - strength) * raw + strength * metric_first
