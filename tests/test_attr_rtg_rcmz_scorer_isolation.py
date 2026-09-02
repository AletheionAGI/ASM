"""Negative IPC and broker-order tests; SYNTHETIC NON-OFFICIAL only."""

import pytest
import torch

from attr_rtg_rcmz.constants import ARMS
from attr_rtg_rcmz.evaluation_broker import ScorerRefs, freeze_all_arms_then_join_truth
from attr_rtg_rcmz.official_data import TruthCache, generate_registered_origins
from attr_rtg_rcmz.scorer import (
    MESSAGE_KEYS,
    ScorerRequest,
    ScorerResponse,
    serialize_message,
)


def four_fields():
    return {
        "history_bytes": torch.zeros((1, 256), dtype=torch.uint8),
        "candidate4s": torch.zeros((1, 6, 4), dtype=torch.float32),
        "masks": torch.ones((1, 6), dtype=torch.bool),
        "logical_lengths": torch.ones(1, dtype=torch.int64),
    }


def test_scorer_ipc_keys_are_exact_and_serialized_tensors_are_detached():
    fields = four_fields()
    encoded = serialize_message(fields)
    assert frozenset(encoded) == MESSAGE_KEYS
    assert all(
        value.device.type == "cpu" and not value.requires_grad
        for value in encoded.values()
    )
    ScorerRequest("R", "config.yaml", "checkpoint.pt", encoded)
    with pytest.raises(ValueError, match="exactly"):
        ScorerRequest("R", "config.yaml", "checkpoint.pt", {**encoded, "truth": None})


def test_world_origin_and_truth_objects_are_rejected_from_scorer_fields():
    origin = generate_registered_origins(miniature=True)["train"][0]
    fields = four_fields()
    fields["history_bytes"] = origin
    with pytest.raises(TypeError, match="forbidden"):
        ScorerRequest("R", "config.yaml", "checkpoint.pt", fields)
    with pytest.raises(TypeError, match="tensors"):
        serialize_message(fields)


def test_broker_freezes_all_four_arm_scores_before_truth(monkeypatch):
    import attr_rtg_rcmz.evaluation_broker as broker

    origins = generate_registered_origins(miniature=True)["calibration"][:1]
    events = []

    def fake_score(requests, *, device):
        events.append(requests[0].arm)
        return (ScorerResponse(((0.0,) * 6,), ((0.0,) * 24,), ((0.0,),)),)

    def fake_truth(selected, device, cache):
        events.append("truth")
        return torch.zeros((len(selected), 6), dtype=torch.float32)

    monkeypatch.setattr(broker, "score_in_clean_process", fake_score)
    monkeypatch.setattr(broker, "truths_after_forward", fake_truth)
    refs = {arm: ScorerRefs(f"{arm}.yaml", f"{arm}.pt") for arm in ARMS}
    result = freeze_all_arms_then_join_truth(
        origins, refs, device="cpu", batch_size=1, truth_cache=TruthCache()
    )
    assert events == ["R", "CM", "Z", "T", "truth"]
    assert tuple(item.arm for item in result.scores) == ARMS
    assert result.identities == (origins[0].identity,)
