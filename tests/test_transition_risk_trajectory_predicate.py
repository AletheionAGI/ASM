import torch

from aletheion_state_models.benchmarks.transition_risk.dataset import make_worlds
from aletheion_state_models.benchmarks.transition_risk.trajectory_dataset import (
    rollout_trajectory_episode,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_predicate import (
    categorical_inverse_cdf,
    horizon_unsafe,
    physical_unsafe,
    unsafe_from_targets,
)


def test_physical_oracle_reproduces_branch_truth_and_horizons():
    config = make_worlds(3, 77, max_steps=14)
    for world in config:
        episode = rollout_trajectory_episode(world, 909)
        oracle = unsafe_from_targets(
            episode.targets, episode.trap_cells, episode.valid_mask, world.failure_delay
        )
        torch.testing.assert_close(oracle, episode.unsafe_truth)
        expected = torch.stack(
            [episode.unsafe_truth[:, :h].any(-1) for h in (1, 4, 8)], -1
        )
        torch.testing.assert_close(horizon_unsafe(oracle, episode.valid_mask), expected)


def test_predicate_collision_delayed_failure_and_post_terminal_ignore():
    agent = torch.tensor([[2, 7, 8, 4]])
    traps = torch.tensor([[[2, 3, 5]] * 4])
    hazard = torch.tensor([[0, 7, 0, 4]])
    low = torch.tensor([[0, 0, 3, 3]])
    recovery = torch.tensor([[0, 0, 0, 0]])
    terminal = torch.tensor([[0, 0, 1, 0]])
    result = physical_unsafe(agent, traps, hazard, low, recovery, terminal, 3)
    assert result.tolist() == [[True, True, True, False]]


def test_inverse_cdf_has_stable_boundaries_and_common_random_numbers():
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
    uniforms = torch.tensor([0.19, 0.50])
    assert categorical_inverse_cdf(probabilities, uniforms).tolist() == [0, 1]
    shared = torch.tensor([0.73, 0.73])
    assert categorical_inverse_cdf(probabilities, shared).tolist() == [2, 2]
