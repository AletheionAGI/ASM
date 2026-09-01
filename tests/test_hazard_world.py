from dataclasses import replace

import pytest

from world_model.hazard_world import (
    ACTIONS,
    HazardWorld,
    HazardWorldConfig,
    HazardWorldState,
    assert_no_world_leakage,
    config_from_json,
    is_safe,
    serialize_config,
    serialize_state,
    split_worlds,
    state_from_json,
    transition,
)


def config(**changes):
    base = HazardWorldConfig(
        world_id="world-a",
        seed=19,
        grid_size=7,
        goal=(6, 6),
        initial_agent=(1, 1),
        traps=((2, 1),),
        moving_hazards=((4, 4),),
        hazard_velocities=((0, 1),),
        sensor_noise=0.2,
    )
    return replace(base, **changes)


def test_seeded_dynamics_are_reproducible_and_all_actions_work():
    actions = ["R", "D", "BRAKE", "RECOVER", "L", "U", "STOP"]
    left, right = HazardWorld(config()), HazardWorld(config())
    assert set(actions) == set(ACTIONS)
    assert [left.step(a) for a in actions] == [right.step(a) for a in actions]


def test_trap_is_irreversible_unsafe_state():
    world = HazardWorld(config())
    item = world.step("D")
    assert item.unsafe and item.done and item.severity == 1.0
    assert not is_safe(world.config, item.state)
    with pytest.raises(RuntimeError):
        world.step("U")


def test_low_energy_has_delayed_failure_and_recover_window():
    cfg = config(
        initial_energy=0.11,
        failure_threshold=0.12,
        failure_delay=2,
        forcing=0,
        traps=(),
        moving_hazards=(),
        hazard_velocities=(),
    )
    initial = HazardWorldState((1, 1), (0, 0), (), (), 0.11)
    first = transition(cfg, initial, "BRAKE")
    assert not first.unsafe and first.state.low_energy_steps == 1
    recovered = transition(cfg, first.state, "RECOVER")
    assert recovered.state.energy > cfg.failure_threshold and not recovered.unsafe
    second = transition(cfg, first.state, "BRAKE")
    assert second.unsafe and second.done


def test_clone_interventions_share_exogenous_randomness():
    world = HazardWorld(config(traps=(), goal=(6, 0)))
    left, right = world.clone(), world.clone()
    for left_action, right_action in zip(["R", "D", "BRAKE"], ["L", "U", "RECOVER"]):
        l_item, r_item = left.step(left_action), right.step(right_action)
        assert l_item.state.hidden_mode == r_item.state.hidden_mode
        assert l_item.state.hazards == r_item.state.hazards
        assert l_item.state.hazard_velocities == r_item.state.hazard_velocities


def test_observation_is_partial_and_does_not_leak_latents():
    world = HazardWorld(config())
    observation = world.observe()
    assert not hasattr(observation, "hidden_mode")
    assert not hasattr(observation, "seed")
    assert not hasattr(observation, "low_energy_steps")
    assert observation.local_hazards == ()


def test_config_and_state_round_trip_serialization():
    world = HazardWorld(config())
    world.step("R")
    assert config_from_json(serialize_config(world.config)) == world.config
    assert state_from_json(serialize_state(world.state)) == world.state


def test_stable_complete_world_splits_have_no_leakage():
    worlds = [
        replace(config(), world_id=f"world-{index}", seed=index) for index in range(100)
    ]
    first = split_worlds(worlds)
    second = split_worlds(reversed(worlds))
    assert {name: {w.world_id for w in values} for name, values in first.items()} == {
        name: {w.world_id for w in values} for name, values in second.items()
    }
    assert_no_world_leakage(first)
    all_ids = [w.world_id for values in first.values() for w in values]
    assert len(all_ids) == len(set(all_ids)) == len(worlds)
    with pytest.raises(ValueError, match="duplicate"):
        split_worlds([worlds[0], worlds[0]])


def test_leakage_audit_rejects_same_world_in_two_splits():
    world = config()
    with pytest.raises(ValueError, match="occurs"):
        assert_no_world_leakage({"train": [world], "test_id": [world]})
