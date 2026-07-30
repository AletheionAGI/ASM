from __future__ import annotations

import torch


class GeometricStepsMixin:
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
