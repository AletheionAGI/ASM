from aletheion_state_models.benchmarks.transition_risk import rtg_cloning
from aletheion_state_models.benchmarks.transition_risk.rtg_cloning import (
    branch_fallback,
    materialize_origin_truth,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_dataset import (
    prepare_rtg_episode,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_types import NONSTOP_ACTIONS
from world_model.hazard_world import HazardWorld
from world_model.hazard_world_types import HazardWorldConfig


def toy_config():
    return HazardWorldConfig(
        world_id="toy-cloning",
        seed=13,
        max_steps=8,
        traps=((0, 8), (8, 0), (8, 8)),
        moving_hazards=((7, 7),),
        hazard_velocities=((0, 0),),
        initial_agent=(4, 4),
        goal=(6, 6),
        forcing=0.0,
        sensor_noise=0.0,
    )


def test_prepare_calls_no_candidate_branch_and_truth_materializes_later(monkeypatch):
    calls = []
    original = rtg_cloning.branch_candidate

    def tracked_branch(origin, action):
        calls.append(action)
        return original(origin, action)

    def forbidden_clone(self):
        raise AssertionError("prepare must not clone the physical origin")

    original_clone = HazardWorld.clone
    monkeypatch.setattr(rtg_cloning, "branch_candidate", tracked_branch)
    monkeypatch.setattr(HazardWorld, "clone", forbidden_clone)
    prepared = prepare_rtg_episode(
        toy_config(), split_id="toy", split_seed=901, episode_index=0
    )
    assert prepared and calls == []
    assert not hasattr(prepared[0], "truth")

    monkeypatch.setattr(HazardWorld, "clone", original_clone)
    materialized = materialize_origin_truth(prepared[0])
    assert calls == list(NONSTOP_ACTIONS)
    assert len(materialized.truth.candidates) == 6
    assert len(materialized.truth.persistence_target) == 11


def test_truth_uses_six_independent_clones_and_preserves_snapshot(monkeypatch):
    prepared = prepare_rtg_episode(
        toy_config(), split_id="toy", split_seed=902, episode_index=0
    )[0]
    clone_states = []
    original_clone = HazardWorld.clone

    def tracked_clone(self):
        clone = original_clone(self)
        clone_states.append(clone)
        return clone

    monkeypatch.setattr(HazardWorld, "clone", tracked_clone)
    materialized = materialize_origin_truth(prepared)
    assert len(clone_states) == 6
    assert len({id(item) for item in clone_states}) == 6
    assert prepared.snapshot.state.step == prepared.metadata.t
    assert all(
        item.transition.state.step == prepared.snapshot.state.step + 1
        for item in materialized.truth.candidates
    )


def test_fallback_is_a_one_step_brake_clone_without_origin_mutation():
    world = HazardWorld(toy_config())
    initial = world.state
    fallback = branch_fallback(world)
    assert world.state == initial
    assert fallback.transition.state.step == initial.step + 1
    direct = world.clone().step("BRAKE")
    assert fallback.transition == direct
