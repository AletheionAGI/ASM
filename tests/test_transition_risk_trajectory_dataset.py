import torch

from aletheion_state_models.benchmarks.transition_risk.dataset import make_worlds
from aletheion_state_models.benchmarks.transition_risk.trajectory_dataset import (
    collate_trajectory_episodes,
    rollout_trajectory_episode,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_types import (
    TARGET_CARDINALITIES,
)


def test_trajectory_episode_has_exact_physical_target_contract():
    episode = rollout_trajectory_episode(make_worlds(1, 51, max_steps=10)[0], 123)
    steps = episode.step_positions.numel()
    assert episode.plan_actions.shape == (steps, 8)
    assert episode.trap_cells.shape == (steps, 3)
    assert episode.valid_mask.shape == episode.unsafe_truth.shape == (steps, 8)
    assert episode.plan_actions.dtype == episode.trap_cells.dtype == torch.long
    assert episode.valid_mask.dtype == episode.unsafe_truth.dtype == torch.bool
    assert set(episode.targets.as_dict()) == set(TARGET_CARDINALITIES)
    assert not any(
        "hazard_probability" in name or "hazard_label" in name
        for name in episode.targets.as_dict()
    )
    assert all(
        value.shape == (steps, 8) for value in episode.targets.as_dict().values()
    )


def test_collate_pads_only_behavior_axes_and_keeps_alignment():
    config = make_worlds(1, 61, max_steps=10)[0]
    episodes = [rollout_trajectory_episode(config, seed) for seed in (2, 3)]
    batch = collate_trajectory_episodes(episodes)
    assert batch["input_ids"].shape[0] == 2
    assert batch["plan_actions"].shape[2] == 8
    assert batch["trap_cells"].shape[2] == 3
    assert batch["step_mask"].sum() == sum(x.step_positions.numel() for x in episodes)
    for row, episode in enumerate(episodes):
        steps = episode.step_positions.numel()
        torch.testing.assert_close(
            batch["unsafe_truth"][row, :steps], episode.unsafe_truth
        )
        torch.testing.assert_close(
            batch["targets"]["agent_cell"][row, :steps], episode.targets.agent_cell
        )
