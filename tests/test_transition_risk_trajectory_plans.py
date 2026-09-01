from aletheion_state_models.benchmarks.transition_risk.dataset import make_worlds
from aletheion_state_models.benchmarks.transition_risk.trajectory_plans import (
    frozen_action_plan,
    rollout_cloned_plan,
)
from world_model.hazard_world import HazardWorld


def test_frozen_plan_keeps_current_action_and_never_uses_stop():
    left = frozen_action_plan("BRAKE", "episode-a", 3)
    assert left == frozen_action_plan("BRAKE", "episode-a", 3)
    assert len(left) == 8 and left[0] == "BRAKE" and "STOP" not in left
    assert left[1:] != frozen_action_plan("BRAKE", "episode-b", 3)[1:]


def test_plan_rollout_clones_origin():
    world = HazardWorld(make_worlds(1, 91)[0])
    state = world.state
    result = rollout_cloned_plan(world, frozen_action_plan("R", "e", 0))
    assert result and world.state == state
