"""Synthetic NON-OFFICIAL goldens for the V1 data/H8 fast path."""

import json
from dataclasses import replace
from pathlib import Path

from attr_rtg_rcmz.adapters import hazard_world, model_process_input
from attr_rtg_rcmz.constants import CANDIDATES
from attr_rtg_rcmz.data_contracts import OriginKey
from attr_rtg_rcmz.h8 import h8_all_candidates, h8_truth
from attr_rtg_rcmz.manifests import batch_manifest, candidate_manifest, split_manifest
from world_model.hazard_world_types import HazardWorldConfig, HazardWorldState

GOLDEN = Path(__file__).parent / "goldens/attr_rtg_rcmz_v1_synthetic.json"


def synthetic_config(**changes):
    base = HazardWorldConfig(
        world_id="SYNTHETIC-NON-OFFICIAL-0",
        seed=7,
        grid_size=5,
        max_steps=8,
        goal=(4, 4),
        initial_agent=(1, 1),
        traps=(),
        moving_hazards=(),
        hazard_velocities=(),
        sensor_noise=0.0,
        forcing=0.0,
    )
    return replace(base, **changes)


def test_manifest_goldens_are_deterministic_and_non_official():
    origins = (
        OriginKey("train", "synthetic-b", 0, 1),
        OriginKey("train", "synthetic-a", 0, 1),
    )
    actual = {
        "split": split_manifest().to_dict(),
        "batch": batch_manifest(origins, 29, batch_size=2, updates=3).to_dict(),
        "candidate": candidate_manifest(origins).to_dict(),
    }
    assert actual == json.loads(GOLDEN.read_text())
    assert all("NON-OFFICIAL" in item["payload"]["notice"] for item in actual.values())
    assert actual["candidate"]["payload"]["rows"][0]["candidates"] == list(CANDIDATES)


def test_exact_four_field_model_message_and_fixed_candidate_axis():
    frames = [[index, 0, 1, 2] for index in range(6)]
    payload = model_process_input([[1, 2, 3]], [frames], [[False] * 6], [3])
    assert tuple(payload.as_dict()) == (
        "history_bytes",
        "candidate4s",
        "masks",
        "logical_lengths",
    )
    assert not any(
        key in payload.as_dict() for key in ("arm", "seed", "truth", "world")
    )


def test_h8_candidate_then_brake_and_unsafe_is_inclusive():
    cfg = synthetic_config(traps=((2, 1),))
    result = h8_truth(hazard_world(cfg), "D")
    assert result.valid and result.unsafe is True and result.transitions == 1
    assert result.terminal


def test_safe_terminal_is_absorbing_and_labelled_zero():
    cfg = synthetic_config(max_steps=1)
    result = h8_truth(hazard_world(cfg), "BRAKE")
    assert result.valid and result.unsafe is False
    assert result.terminal and result.transitions == 1


def test_every_candidate_uses_one_independent_origin_clone():
    cfg = synthetic_config(max_steps=8)
    world = hazard_world(cfg)
    before = world.state
    truths = h8_all_candidates(world)
    assert tuple(item.candidate for item in truths) == CANDIDATES
    assert all(item.valid and item.transitions == 8 for item in truths)
    assert world.state == before


def test_invalid_clone_truth_fails_closed_without_action_remap():
    class Broken:
        state = HazardWorldState((0, 0), (0, 0), (), (), 1.0)

        def clone(self):
            return self

    result = h8_truth(Broken(), "RECOVER")
    assert result.candidate == "RECOVER" and not result.valid
    assert result.unsafe is None and result.transitions == 0


def test_registered_origin_adapter_preserves_byte_frame_semantics_without_truth():
    from aletheion_state_models.benchmarks.transition_risk.dataset import encode_frame
    from attr_rtg_rcmz.official_data import generate_registered_origins, materialize

    data = generate_registered_origins(miniature=True)
    assert tuple(data) == (
        "train",
        "validation",
        "calibration",
        "test_id",
        "test_shift",
        "test_ood",
    )
    origin = data["train"][0]
    assert origin.origin == 1 and len(origin.history) == 4
    observation = origin.world.observe()
    assert origin.candidate4s == tuple(
        encode_frame(observation, action, origin.world.config.grid_size)
        for action in CANDIDATES
    )
    batch = materialize(data["train"], [0], "cpu")
    assert set(batch["message"]) == {
        "history_bytes",
        "candidate4s",
        "masks",
        "logical_lengths",
    }
    assert batch["message"]["logical_lengths"].tolist() == [4]
    assert not any(
        name in batch["message"] for name in ("truth", "labels", "world", "seed")
    )


def test_privileged_h8_cache_reuses_truth_but_never_enters_message():
    from attr_rtg_rcmz.official_data import (
        TruthCache,
        generate_registered_origins,
        materialize,
        truths_after_forward,
    )

    origins = generate_registered_origins(miniature=True)["train"]
    cache = TruthCache()
    batch = materialize(origins, [0], "cpu")
    first = truths_after_forward(batch["origins"], "cpu", cache)
    second = truths_after_forward(batch["origins"], "cpu", cache)
    assert first.equal(second) and len(cache._rows) == 1
    assert set(batch["message"]) == {
        "history_bytes",
        "candidate4s",
        "masks",
        "logical_lengths",
    }
