import json
from dataclasses import replace

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_allowed_manifest import (
    ALLOWED_SPLITS,
    AllowedSplitData,
    _assert_pairwise_disjoint,
    _audit_disjointness,
    _episode_payload,
    _origin_payload,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_dataset import (
    prepare_rtg_episode,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_training_data import (
    BehavioralEpisode,
)
from world_model.hazard_world_types import HazardWorldConfig


def test_allowed_manifest_freezes_input_only_payloads_and_disjoint_ids():
    episode_payload = _episode_payload(BehavioralEpisode("toy", (1, 2, 3, 4)))
    assert ALLOWED_SPLITS == ("train", "validation", "calibration")
    assert episode_payload["logical_length"] == 4
    assert episode_payload["offsets"] == [0, 4]
    assert episode_payload["input_ids"][:5] == [1, 2, 3, 4, 0]
    assert episode_payload["targets"][:5] == [2, 3, 4, -100, -100]
    assert episode_payload["target_mask"][:5] == [True, True, True, False, False]
    assert len(episode_payload["input_ids"]) == 64

    config = HazardWorldConfig(
        world_id="toy-world",
        seed=7,
        max_steps=2,
        traps=((1, 1), (2, 2), (3, 3)),
    )
    prepared = prepare_rtg_episode(
        config, split_id="train", split_seed=11, episode_index=0
    )
    origin_payload = _origin_payload(prepared[0])
    serialized = json.dumps(origin_payload, sort_keys=True)
    for forbidden in ("truth", "unsafe", "persistence", "failure_delay"):
        assert forbidden not in serialized
    assert len(origin_payload["config_sha256"]) == 64
    assert len(origin_payload["state_sha256"]) == 64
    assert len(origin_payload["origin_input_sha256"]) == 64
    assert all(
        len(candidate["candidate_input_sha256"]) == 64
        for candidate in origin_payload["candidates"]
    )

    _assert_pairwise_disjoint(({"train"}, {"validation"}), "episode_ids")
    with pytest.raises(ValueError, match="not disjoint"):
        _assert_pairwise_disjoint(({"same"}, {"same"}), "episode_ids")


def test_content_only_hashes_reject_same_content_under_different_ids():
    base_config = HazardWorldConfig(
        world_id="world-a", seed=7, max_steps=2,
        traps=((1, 1), (2, 2), (3, 3)),
    )
    original = prepare_rtg_episode(
        base_config, split_id="train", split_seed=11, episode_index=0
    )[0]
    groups = []
    for index, split in enumerate(ALLOWED_SPLITS):
        config = replace(base_config, world_id=f"world-{index}", seed=100 + index)
        metadata = replace(
            original.metadata, split_id=split, world_id=config.world_id,
            episode_id=f"episode-{index}",
        )
        snapshot = replace(original.snapshot, config=config)
        prepared = replace(original, metadata=metadata, snapshot=snapshot)
        episode = BehavioralEpisode(f"episode-{index}", (1, 2, 3, 4))
        groups.append(AllowedSplitData(split, (config,), (episode,), (prepared,)))
    assert len({_episode_payload(item.episodes[0])["episode_sha256"] for item in groups}) == 1
    payloads = [_origin_payload(item.origins[0]) for item in groups]
    assert len({row["config_sha256"] for row in payloads}) == 1
    assert len({row["origin_input_sha256"] for row in payloads}) == 1
    assert len({row["candidates"][0]["candidate_input_sha256"] for row in payloads}) == 1
    with pytest.raises(ValueError, match="not disjoint"):
        _audit_disjointness(tuple(groups))
