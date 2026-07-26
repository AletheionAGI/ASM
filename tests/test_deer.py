import torch

from drm_language_emitter.deer import (
    anderson_solve,
    causal_anderson_solve,
    cumulative_delta_warmstart,
    fixed_point_solve,
    sequential_rollout,
)


def test_fixed_point_solve_matches_linear_sequential_rollout():
    torch.manual_seed(7)
    batch, seq_len, d_state = 2, 8, 4
    matrix = 0.2 * torch.eye(d_state)
    bias = torch.randn(d_state) * 0.01
    z0 = torch.randn(batch, d_state)
    inputs = torch.randn(batch, seq_len, d_state)

    def transition(z, x):
        return z @ matrix.T + 0.1 * x + bias

    expected = sequential_rollout(transition, z0, inputs)
    solved, residuals = fixed_point_solve(transition, z0, inputs, iterations=seq_len)
    assert residuals[-1] < residuals[0]
    assert torch.allclose(solved, expected, atol=1e-5)


def test_anderson_solve_reduces_nonlinear_residual():
    torch.manual_seed(11)
    batch, seq_len, d_state = 2, 10, 5
    matrix = 0.15 * torch.randn(d_state, d_state)
    z0 = torch.randn(batch, d_state) * 0.1
    inputs = torch.randn(batch, seq_len, d_state) * 0.1

    def transition(z, x):
        return torch.tanh(z @ matrix.T + x)

    _expected = sequential_rollout(transition, z0, inputs)
    solved, residuals = anderson_solve(transition, z0, inputs, iterations=8, history_size=4)
    final_image, _ = fixed_point_solve(transition, z0, inputs, iterations=1, initial_trajectory=solved)
    final_residual = (final_image - solved).norm()
    assert residuals[-1] < residuals[0]
    assert final_residual < residuals[0]


def test_cumulative_delta_warmstart_matches_additive_dynamics():
    torch.manual_seed(13)
    batch, seq_len, d_state = 2, 8, 4
    z0 = torch.randn(batch, d_state)
    inputs = torch.randn(batch, seq_len, d_state)

    def transition(z, x):
        return z + 0.1 * x

    expected = sequential_rollout(transition, z0, inputs)
    warmstart = cumulative_delta_warmstart(transition, z0, inputs)
    assert torch.allclose(warmstart, expected, atol=1e-6)


def test_global_anderson_can_couple_prefix_to_future_inputs():
    def transition(z, x):
        return torch.tanh(0.7 * z + x)

    torch.manual_seed(17)
    z0 = torch.zeros(1, 3)
    inputs = torch.randn(1, 8, 3)
    changed = inputs.clone()
    changed[:, 4:] += 3.0

    solved, _ = anderson_solve(transition, z0, inputs, iterations=4, history_size=4, ridge=1e-3)
    solved_changed, _ = anderson_solve(transition, z0, changed, iterations=4, history_size=4, ridge=1e-3)

    assert (solved[:, :4] - solved_changed[:, :4]).abs().max() > 1e-3


def test_causal_anderson_does_not_couple_prefix_to_future_inputs():
    def transition(z, x):
        return torch.tanh(0.7 * z + x)

    torch.manual_seed(19)
    z0 = torch.zeros(1, 3)
    inputs = torch.randn(1, 8, 3)
    changed = inputs.clone()
    changed[:, 4:] += 3.0

    solved, residuals = causal_anderson_solve(transition, z0, inputs, iterations=4, history_size=4, ridge=1e-3)
    solved_changed, _ = causal_anderson_solve(transition, z0, changed, iterations=4, history_size=4, ridge=1e-3)

    assert residuals[-1] < residuals[0]
    assert torch.allclose(solved[:, :4], solved_changed[:, :4], atol=1e-6)
