import torch

from aletheion_state_models.benchmarks.transition_risk.dataset import make_worlds
from aletheion_state_models.benchmarks.transition_risk.trajectory_dataset import (
    rollout_trajectory_episode,
)


def test_behavior_suffix_perturbation_cannot_change_origin_plan_or_targets():
    config = make_worlds(1, 81, max_steps=12)[0]
    left = rollout_trajectory_episode(
        config, 404, behavior_actions=("R",) + ("U",) * 11
    )
    right = rollout_trajectory_episode(
        config, 404, behavior_actions=("R",) + ("RECOVER",) * 11
    )
    torch.testing.assert_close(left.plan_actions[0], right.plan_actions[0])
    torch.testing.assert_close(left.valid_mask[0], right.valid_mask[0])
    torch.testing.assert_close(left.unsafe_truth[0], right.unsafe_truth[0])
    for name in left.targets.as_dict():
        torch.testing.assert_close(
            left.targets.as_dict()[name][0], right.targets.as_dict()[name][0]
        )
    torch.testing.assert_close(left.input_ids[:4], right.input_ids[:4])
