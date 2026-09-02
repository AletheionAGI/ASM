from dataclasses import replace

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_physical_targets import (
    audit_transition_target,
    encode_y_common,
    unsafe_predicate,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_types import (
    Y_COMMON_CARDINALITIES,
)
from world_model.hazard_world import HazardWorld
from world_model.hazard_world_types import HazardWorldConfig


def toy_config(**changes):
    base = HazardWorldConfig(
        world_id="toy-target",
        seed=12,
        traps=((0, 8), (8, 0), (8, 8)),
        moving_hazards=((7, 7),),
        hazard_velocities=((0, 0),),
        initial_agent=(4, 4),
        goal=(6, 6),
        forcing=0.0,
        sensor_noise=0.0,
    )
    return replace(base, **changes)


def test_common_target_has_exact_eleven_groups_and_excludes_hidden_mode():
    world = HazardWorld(toy_config())
    transition = world.step("BRAKE")
    target = encode_y_common(world.config, transition.state)
    assert len(target) == len(Y_COMMON_CARDINALITIES) == 11
    assert all(0 <= value < size for value, size in zip(target, Y_COMMON_CARDINALITIES))
    assert target[:3] == (8, 72, 80)
    assert target[5:7] == (1, 1)
    assert target[-1] == 0
    audit_transition_target(target, transition, world.config.failure_delay)


def test_predicate_detects_collision_and_delayed_failure_and_safe_terminal_suppresses():
    base = (8, 72, 80, 40, 70, 1, 1, 63, 0, 0, 0)
    assert unsafe_predicate((*base[:3], 8, *base[4:]), 3)
    delayed = (*base[:8], 3, 0, 0)
    assert unsafe_predicate(delayed, 3)
    assert not unsafe_predicate((*delayed[:-1], 1), 3)


def test_predicate_fails_closed_for_malformed_duplicate_or_wrong_delay():
    valid = (8, 72, 80, 40, 70, 1, 1, 63, 0, 0, 0)
    assert unsafe_predicate(valid[:-1], 3)
    assert unsafe_predicate((8, 8, *valid[2:]), 3)
    assert unsafe_predicate(valid, 2)
    assert unsafe_predicate((*valid[:-1], float("nan")), 3)


def test_audit_rejects_truth_mismatch():
    world = HazardWorld(toy_config())
    transition = world.step("BRAKE")
    collision_target = (8, 72, 80, 8, 70, 1, 1, 63, 0, 0, 0)
    with pytest.raises(ValueError, match="does not match"):
        audit_transition_target(collision_target, transition, 3)
