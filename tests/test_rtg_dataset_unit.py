import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_cloning import (
    materialize_origin_truth,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_dataset import (
    prepare_rtg_episode,
    prepare_rtg_input_origins,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_types import (
    NONSTOP_ACTIONS,
    OriginInput,
    PreparedRtgOrigin,
)
from world_model.hazard_world_types import HazardWorldConfig


def toy_config(world_id="toy-dataset", max_steps=5):
    return HazardWorldConfig(
        world_id=world_id,
        seed=14,
        max_steps=max_steps,
        traps=((0, 8), (8, 0), (8, 8)),
        moving_hazards=((7, 7),),
        hazard_velocities=((0, 0),),
        initial_agent=(4, 4),
        goal=(6, 6),
        forcing=0.0,
        sensor_noise=0.0,
    )


def test_prepare_excludes_t0_and_contains_no_truth_or_labels():
    origins = prepare_rtg_episode(
        toy_config(), split_id="toy", split_seed=991, episode_index=0
    )
    assert origins and all(isinstance(item, PreparedRtgOrigin) for item in origins)
    assert origins[0].metadata.t == 1
    for origin in origins:
        assert len(origin.inputs.history) == origin.metadata.t * 4
        assert len(origin.inputs.candidates) == 6
        assert len(origin.inputs.candidates) == len(NONSTOP_ACTIONS)
        assert not hasattr(origin, "truth")
        assert not hasattr(origin.inputs.candidates[0], "unsafe")
        assert origin.snapshot.state.step == origin.metadata.t
        assert not origin.snapshot.state.terminal


def test_prepare_is_reproducible_without_materializing_truth():
    kwargs = {"split_id": "toy", "split_seed": 992, "episode_index": 2}
    left = prepare_rtg_episode(toy_config(), **kwargs)
    right = prepare_rtg_episode(toy_config(), **kwargs)
    assert [item.metadata for item in left] == [item.metadata for item in right]
    assert [
        [candidate.frame for candidate in item.inputs.candidates] for item in left
    ] == [[candidate.frame for candidate in item.inputs.candidates] for item in right]
    assert [item.snapshot.state for item in left] == [
        item.snapshot.state for item in right
    ]


def test_second_phase_adds_persistence_and_six_truths_only_on_request():
    prepared = prepare_rtg_episode(
        toy_config(), split_id="toy", split_seed=995, episode_index=0
    )[0]
    complete = materialize_origin_truth(prepared)
    assert complete.metadata == prepared.metadata
    assert complete.inputs is prepared.inputs
    assert len(complete.truth.persistence_target) == 11
    assert len(complete.truth.candidates) == 6
    assert all(len(item.target) == 11 for item in complete.truth.candidates)


def test_prepare_input_origins_sorts_and_preserves_six_candidate_cluster():
    origins = prepare_rtg_input_origins(
        [toy_config("toy-z"), toy_config("toy-a")],
        episodes_per_world=2,
        split_id="toy",
        split_seed=993,
    )
    keys = [
        (item.metadata.world_id, item.metadata.episode_id, item.metadata.t)
        for item in origins
    ]
    assert keys == sorted(keys)
    assert all(len(item.inputs.candidates) == 6 for item in origins)
    assert {item.metadata.world_id for item in origins} == {"toy-a", "toy-z"}


def test_origin_input_rejects_candidate_reordering():
    origin = prepare_rtg_episode(
        toy_config(), split_id="toy", split_seed=994, episode_index=0
    )[0]
    with pytest.raises(ValueError, match="action order"):
        OriginInput(
            history=origin.inputs.history,
            candidates=tuple(reversed(origin.inputs.candidates)),
        )
