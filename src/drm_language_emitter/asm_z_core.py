from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import DRMConfig


@dataclass(frozen=True)
class ASMZGeometry:
    potential: torch.Tensor
    gradient: torch.Tensor
    diagonal: torch.Tensor
    low_rank: torch.Tensor

    @property
    def metric(self) -> torch.Tensor:
        return torch.diag_embed(self.diagonal) + self.low_rank @ self.low_rank.transpose(-1, -2)


class ScalarPotential(nn.Module):
    """Learned input-conditioned scalar potential Phi(z, e)."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.regularization = float(config.asm_z_lambda)
        self.net = nn.Sequential(
            nn.Linear(config.d_state + config.d_token, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 1, bias=False),
        )

    def forward(self, state: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        learned = self.net(torch.cat((state, token), dim=-1)).squeeze(-1)
        return learned + 0.5 * self.regularization * state.square().sum(-1)


class InputConditionedSPDMetric(nn.Module):
    """Bounded diagonal SPD metric plus a Frobenius-bounded low-rank term."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.d_state = config.d_state
        self.rank = config.metric_rank
        self.d_min = float(config.asm_z_metric_d_min)
        self.d_max = float(config.asm_z_metric_d_max)
        self.u_bound = float(config.asm_z_metric_u_bound)
        output_size = self.d_state * (1 + self.rank)
        self.net = nn.Sequential(
            nn.Linear(config.d_state + config.d_token, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, output_size),
        )

    def forward(
        self, state: torch.Tensor, token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat((state, token), dim=-1))
        diagonal = self.d_min + (self.d_max - self.d_min) * torch.sigmoid(
            raw[:, : self.d_state]
        )
        if not self.rank:
            return diagonal, raw.new_zeros(raw.shape[0], self.d_state, 0)
        low_rank = raw[:, self.d_state :].reshape(-1, self.d_state, self.rank)
        norm = low_rank.norm(dim=(1, 2), keepdim=True)
        low_rank = self.u_bound * low_rank / torch.sqrt(self.u_bound**2 + norm.square())
        return diagonal, low_rank


def solve_spd_metric(
    diagonal: torch.Tensor, low_rank: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    """Woodbury solve for (diag(d) + U U^T) x = gradient."""
    solve_dtype = torch.float32 if diagonal.dtype in {torch.float16, torch.bfloat16} else diagonal.dtype
    diagonal_solve = diagonal.to(solve_dtype)
    low_rank_solve = low_rank.to(solve_dtype)
    gradient_solve = gradient.to(solve_dtype)
    inverse_diagonal_gradient = gradient_solve / diagonal_solve
    if not low_rank.shape[-1]:
        return inverse_diagonal_gradient.to(gradient.dtype)
    inverse_diagonal_u = low_rank_solve / diagonal_solve.unsqueeze(-1)
    system = low_rank_solve.transpose(-1, -2) @ inverse_diagonal_u
    identity = torch.eye(system.shape[-1], device=system.device, dtype=system.dtype)
    rhs = low_rank_solve.transpose(-1, -2) @ inverse_diagonal_gradient.unsqueeze(-1)
    correction = inverse_diagonal_u @ torch.linalg.solve(system + identity, rhs)
    return (inverse_diagonal_gradient - correction.squeeze(-1)).to(gradient.dtype)


class ASMZCore(nn.Module):
    """One exact deterministic natural-gradient step with constant eta."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.eta = float(config.asm_z_eta)
        self.potential = ScalarPotential(config)
        self.metric = InputConditionedSPDMetric(config)

    def geometry(self, state: torch.Tensor, token: torch.Tensor) -> ASMZGeometry:
        outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            differentiable_state = state
            if not outer_grad_enabled or not state.requires_grad:
                differentiable_state = state.detach().requires_grad_(True)
            potential = self.potential(differentiable_state, token)
            gradient = torch.autograd.grad(
                potential.sum(), differentiable_state,
                create_graph=self.training and outer_grad_enabled,
            )[0]
            diagonal, low_rank = self.metric(differentiable_state, token)
        if not outer_grad_enabled:
            return ASMZGeometry(potential.detach(), gradient.detach(), diagonal.detach(), low_rank.detach())
        return ASMZGeometry(potential, gradient, diagonal, low_rank)

    def forward(
        self, state: torch.Tensor, token: torch.Tensor
    ) -> tuple[torch.Tensor, ASMZGeometry]:
        geometry = self.geometry(state, token)
        natural_gradient = solve_spd_metric(
            geometry.diagonal, geometry.low_rank, geometry.gradient
        )
        return state - self.eta * natural_gradient, geometry
