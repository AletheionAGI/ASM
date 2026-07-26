from __future__ import annotations

from collections.abc import Callable

import torch


Transition = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sequential_rollout(transition: Transition, z0: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Evaluate z[t+1] = transition(z[t], inputs[:, t]) sequentially."""

    states = []
    z = z0
    for t in range(inputs.shape[1]):
        z = transition(z, inputs[:, t])
        states.append(z)
    return torch.stack(states, dim=1)


def trajectory_fixed_point(transition: Transition, z0: torch.Tensor, inputs: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
    """Apply the full-trajectory fixed-point map Phi(Z)."""

    previous = torch.cat([z0.unsqueeze(1), trajectory[:, :-1]], dim=1)
    batch, seq_len, d_state = previous.shape
    flat_previous = previous.reshape(batch * seq_len, d_state)
    flat_inputs = inputs.reshape(batch * seq_len, *inputs.shape[2:])
    flat_next = transition(flat_previous, flat_inputs)
    return flat_next.reshape(batch, seq_len, d_state)


def fixed_point_solve(
    transition: Transition,
    z0: torch.Tensor,
    inputs: torch.Tensor,
    *,
    iterations: int,
    relaxation: float = 1.0,
    initial_trajectory: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Solve a trajectory by repeated full-trajectory fixed-point updates."""

    if initial_trajectory is None:
        trajectory = z0.unsqueeze(1).expand(-1, inputs.shape[1], -1).clone()
    else:
        trajectory = initial_trajectory
    residuals: list[float] = []
    for _ in range(max(iterations, 0)):
        updated = trajectory_fixed_point(transition, z0, inputs, trajectory)
        residuals.append(float((updated - trajectory).norm().detach().cpu()))
        trajectory = trajectory + relaxation * (updated - trajectory)
    return trajectory, residuals


def cumulative_delta_warmstart(transition: Transition, z0: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Build a parallel warmstart by cumulatively summing local deltas from z0.

    This is exact only for additive state-independent dynamics. It is a cheap
    trajectory guess for mildly varying DRM transitions.
    """

    batch, seq_len = inputs.shape[:2]
    flat_z0 = z0.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch * seq_len, z0.shape[-1])
    flat_inputs = inputs.reshape(batch * seq_len, *inputs.shape[2:])
    local_next = transition(flat_z0, flat_inputs).reshape(batch, seq_len, z0.shape[-1])
    local_delta = local_next - z0.unsqueeze(1)
    return z0.unsqueeze(1) + torch.cumsum(local_delta, dim=1)


def anderson_solve(
    transition: Transition,
    z0: torch.Tensor,
    inputs: torch.Tensor,
    *,
    iterations: int,
    history_size: int = 5,
    ridge: float = 1e-4,
    relaxation: float = 1.0,
    initial_trajectory: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Non-causal Anderson acceleration for full-trajectory fixed-point iteration.

    This intentionally avoids explicit Jacobians. It is a prototype solver for
    tiny experiments, not yet an optimized DEER implementation. Because the
    Anderson coefficients are computed from the full trajectory residual, prefix
    states can depend on future inputs. Do not use this in autoregressive LM
    forward paths that require causal logits.
    """

    if initial_trajectory is None:
        trajectory = z0.unsqueeze(1).expand(-1, inputs.shape[1], -1).clone()
    else:
        trajectory = initial_trajectory

    iterates: list[torch.Tensor] = []
    images: list[torch.Tensor] = []
    residuals: list[float] = []
    for _ in range(max(iterations, 0)):
        image = trajectory_fixed_point(transition, z0, inputs, trajectory)
        residual = image - trajectory
        residuals.append(float(residual.norm().detach().cpu()))
        iterates.append(trajectory)
        images.append(image)
        if len(iterates) > history_size:
            iterates.pop(0)
            images.pop(0)
        if len(iterates) < 2:
            trajectory = trajectory + relaxation * residual
            continue

        flat_residuals = torch.stack([(f - x).reshape(-1) for x, f in zip(iterates, images)], dim=1).float()
        device_type = flat_residuals.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            gram = flat_residuals.T @ flat_residuals
            gram = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            ones = torch.ones(gram.shape[0], device=gram.device, dtype=gram.dtype)
            coeffs = torch.linalg.solve(gram, ones)
            coeffs = coeffs / coeffs.sum().clamp_min(1e-8)
        coeffs = coeffs.to(dtype=trajectory.dtype)
        accelerated = sum(coeff * image_i for coeff, image_i in zip(coeffs, images))
        trajectory = trajectory + relaxation * (accelerated - trajectory)
    return trajectory, residuals


def causal_anderson_solve(
    transition: Transition,
    z0: torch.Tensor,
    inputs: torch.Tensor,
    *,
    iterations: int,
    history_size: int = 5,
    ridge: float = 1e-4,
    relaxation: float = 1.0,
    initial_trajectory: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Causal Anderson acceleration using prefix Gram matrices.

    For each position t, the Anderson coefficients are fitted only from
    residuals at positions <= t. The prefix reductions are vectorized with
    cumsum over sequence length, and the small history systems are solved as a
    batched linear solve.
    """

    if initial_trajectory is None:
        trajectory = z0.unsqueeze(1).expand(-1, inputs.shape[1], -1).clone()
    else:
        trajectory = initial_trajectory

    iterates: list[torch.Tensor] = []
    images: list[torch.Tensor] = []
    residuals: list[float] = []
    for _ in range(max(iterations, 0)):
        image = trajectory_fixed_point(transition, z0, inputs, trajectory)
        residual = image - trajectory
        residuals.append(float(residual.norm().detach().cpu()))
        iterates.append(trajectory)
        images.append(image)
        if len(iterates) > history_size:
            iterates.pop(0)
            images.pop(0)
        if len(iterates) < 2:
            trajectory = trajectory + relaxation * residual
            continue

        residual_history = torch.stack([f - x for x, f in zip(iterates, images)], dim=-1)
        image_history = torch.stack(images, dim=-1)
        batch, seq_len, _d_state, history = residual_history.shape
        solve_dtype = torch.float32 if residual_history.dtype in {torch.float16, torch.bfloat16} else residual_history.dtype
        device_type = residual_history.device.type if residual_history.device.type in {"cuda", "cpu"} else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            residual_solve = residual_history.to(solve_dtype)
            gram_step = torch.einsum("btdh,btdk->bthk", residual_solve, residual_solve)
            gram_prefix = torch.cumsum(gram_step, dim=1)
            eye = torch.eye(history, device=trajectory.device, dtype=solve_dtype).view(1, 1, history, history)
            gram_prefix = gram_prefix + ridge * eye
            ones = torch.ones(batch, seq_len, history, 1, device=trajectory.device, dtype=solve_dtype)
            coeffs = torch.linalg.solve(
                gram_prefix.reshape(batch * seq_len, history, history),
                ones.reshape(batch * seq_len, history, 1),
            ).reshape(batch, seq_len, history)
            denom = coeffs.sum(dim=-1, keepdim=True)
            denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
            coeffs = coeffs / denom
        coeffs = coeffs.to(dtype=trajectory.dtype)
        accelerated = torch.einsum("bth,btdh->btd", coeffs, image_history)
        trajectory = trajectory + relaxation * (accelerated - trajectory)
    return trajectory, residuals
