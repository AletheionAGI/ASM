import hashlib

from aletheion_state_models.benchmarks.transition_risk.rtg_policy import (
    BEHAVIOR_ACTIONS,
    choose_behavior_action,
    keyed_index,
    keyed_uniform,
)
from world_model.hazard_world_types import HazardObservation


def observation(energy):
    return HazardObservation((1, 1), (0, 0), energy, (2, 2), (), (), (), 3)


def test_keyed_uniform_matches_canonical_sha256_key():
    key = "ATTR-RTG-POLICY-V1|123|toy-episode|3|recover"
    integer = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    assert keyed_uniform(123, "toy-episode", 3, "recover") == (integer + 0.5) / 2**64
    assert 0.0 < keyed_uniform(123, "toy-episode", 3, "recover") < 1.0


def test_policy_is_call_order_independent_and_never_stops():
    expected = choose_behavior_action(observation(0.8), 123, "toy-episode", 3)
    for step in range(20):
        choose_behavior_action(observation(0.1), 999, "other", step)
    assert choose_behavior_action(observation(0.8), 123, "toy-episode", 3) == expected
    assert expected in BEHAVIOR_ACTIONS and expected != "STOP"


def test_action_index_uses_separate_deterministic_channel():
    left = keyed_index(7, "toy", 2, "action", len(BEHAVIOR_ACTIONS))
    right = keyed_index(7, "toy", 2, "action", len(BEHAVIOR_ACTIONS))
    assert left == right and 0 <= left < len(BEHAVIOR_ACTIONS)
