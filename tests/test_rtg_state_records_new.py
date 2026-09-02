import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk.rtg_cloning import (
    materialize_origin_truth,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_dataset import (
    prepare_rtg_episode,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_state_export import (
    CausalStateExporter,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_state_records import (
    CandidateStateInputRecord,
    attach_candidate_truths,
    export_candidate_state_inputs,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_types import (
    CandidateInput,
    CandidateTruth,
    OriginInput,
    OriginMetadata,
    OriginTruth,
    RtgOrigin,
)
from world_model.hazard_world_types import HazardWorldConfig


class ToyExporter(CausalStateExporter):
    representation_dim = 28

    def _forward_states(self, input_ids):
        totals = input_ids.float().cumsum(1).unsqueeze(-1)
        return totals.expand(*input_ids.shape, 28)


def test_export_is_causal_and_candidate_aligned():
    candidates = tuple(
        CandidateInput((0, index, 0, 0), torch.zeros(32)) for index in range(6)
    )
    truths = tuple(
        CandidateTruth(tuple([0] * 11), False, None) for _ in range(6)
    )
    materialized = RtgOrigin(
        OriginMetadata("toy", "w", "e", 1),
        OriginInput((9, 8, 7, 6), candidates),
        OriginTruth(tuple([0] * 11), truths, 3),
    )
    records = attach_candidate_truths(
        tuple(
            CandidateStateInputRecord(
                "toy",
                "w",
                "e",
                1,
                index,
                torch.full((28,), 30.0),
                torch.full((28,), 30.0 + index),
                torch.zeros(32),
            )
            for index in range(6)
        ),
        (materialized,),
    )
    assert len(records) == 6
    assert records[0].pre_state[0].item() == 30.0
    assert records[5].next_state[0].item() == 35.0
    assert [record.action_index for record in records] == list(range(6))


def _toy_config() -> HazardWorldConfig:
    return HazardWorldConfig(
        world_id="toy-state-records",
        seed=31,
        max_steps=6,
        traps=((0, 8), (8, 0), (8, 8)),
        moving_hazards=((7, 7),),
        hazard_velocities=((0, 0),),
        initial_agent=(4, 4),
        goal=(6, 6),
        forcing=0.0,
        sensor_noise=0.0,
    )


def test_prepared_export_has_no_truth_and_attach_joins_by_identity():
    prepared = prepare_rtg_episode(
        _toy_config(), split_id="toy", split_seed=1201, episode_index=0
    )[:2]
    inputs = export_candidate_state_inputs(
        prepared, ToyExporter(nn.Identity()), torch.eye(28)
    )

    assert len(inputs) == 12
    assert all(isinstance(item, CandidateStateInputRecord) for item in inputs)
    assert all(not hasattr(item, "physical_target") for item in inputs)

    materialized = tuple(materialize_origin_truth(item) for item in reversed(prepared))
    records = attach_candidate_truths(inputs, materialized)
    truth_by_origin = {
        (
            origin.metadata.split_id,
            origin.metadata.world_id,
            origin.metadata.episode_id,
            origin.metadata.t,
        ): origin.truth
        for origin in materialized
    }
    for record in records:
        truth = truth_by_origin[
            (record.split_id, record.world_id, record.episode_id, record.t)
        ]
        assert record.physical_target == truth.candidates[record.action_index].target
        assert record.persistence_target == truth.persistence_target
