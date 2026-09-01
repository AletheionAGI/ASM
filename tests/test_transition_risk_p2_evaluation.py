from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk.dataset import HazardEpisode
from aletheion_state_models.benchmarks.transition_risk.model_adapters import (
    ASMModelAdapter,
)
from aletheion_state_models.benchmarks.transition_risk.model_heads import (
    TransitionRiskHeads,
)
from aletheion_state_models.benchmarks.transition_risk.p2_evaluation import (
    compute_aggregate_metrics,
    evaluate_episodes,
    read_episode_records_jsonl,
    write_episode_records_jsonl,
)


class _PositionASM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_state=2)
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, *, return_states, collect_diagnostics):
        assert return_states and not collect_diagnostics
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).float()
        states = torch.stack((positions, positions + 1), -1)
        return {"states": states.expand(input_ids.shape[0], -1, -1) + self.anchor}


def _episode(episode_id="episode-1") -> HazardEpisode:
    return HazardEpisode(
        episode_id=episode_id,
        world_id="world-1",
        input_ids=torch.tensor([8, 7, 6, 5, 4, 3, 2, 1]),
        step_positions=torch.tensor([3, 7]),
        next_states=torch.zeros(2, 2),
        hazard_labels=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        severity=torch.tensor([0.25, 0.75]),
        time_to_hazard=torch.tensor([2.0, 1.0]),
        unsafe=torch.tensor([False, True]),
        actions=("L", "R"),
    )


def _system():
    adapter = ASMModelAdapter(_PositionASM())
    heads = TransitionRiskHeads(2, state_dim=2, horizons=(1, 8))
    for parameter in heads.parameters():
        nn.init.zeros_(parameter)
    return adapter, heads


def test_evaluation_preserves_episode_alignment_and_all_outputs() -> None:
    adapter, heads = _system()
    evaluation = evaluate_episodes(adapter, heads, [_episode()], device="cpu")

    record = evaluation.records[0]
    assert record.episode_id == "episode-1"
    assert record.world_id == "world-1"
    assert record.actions == ("L", "R")
    assert record.horizons == (1, 8)
    assert record.hazard_labels == ((0, 1), (1, 0))
    assert record.hazard_probabilities == ((0.5, 0.5), (0.5, 0.5))
    assert len(record.next_state_nll) == 2
    # Zero common heads imply N(0, I), so each two-dimensional NLL is log(2 pi).
    assert record.next_state_nll == pytest.approx((1.837877, 1.837877))
    assert record.severity_predictions == pytest.approx((0.693147, 0.693147))
    assert record.time_to_hazard_predictions == pytest.approx((0.693147, 0.693147))
    assert evaluation.metrics["hazard"]["1"]["brier"] == pytest.approx(0.25)
    assert evaluation.metrics["steps"] == 2


def test_jsonl_round_trip_is_atomic_and_strict(tmp_path) -> None:
    adapter, heads = _system()
    records = evaluate_episodes(
        adapter, heads, [_episode(), _episode("episode-2")]
    ).records
    path = tmp_path / "nested" / "predictions.jsonl"

    assert write_episode_records_jsonl(path, records) == path
    assert read_episode_records_jsonl(path) == records
    assert len(path.read_text().splitlines()) == 2
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_invalid_records_fail_before_replacing_existing_jsonl(tmp_path) -> None:
    adapter, heads = _system()
    valid = evaluate_episodes(adapter, heads, [_episode()]).records[0]
    invalid = replace(valid, next_state_nll=(float("nan"),) * 2)
    path = tmp_path / "predictions.jsonl"
    path.write_text("keep me")

    with pytest.raises(ValueError, match="finite"):
        write_episode_records_jsonl(path, [invalid])
    assert path.read_text() == "keep me"

    broken = replace(valid, actions=("L",))
    with pytest.raises(ValueError, match="align"):
        compute_aggregate_metrics([broken])


def test_reader_rejects_misaligned_or_non_finite_json(tmp_path) -> None:
    adapter, heads = _system()
    record = evaluate_episodes(adapter, heads, [_episode()]).records[0].to_dict()
    record["hazard_probabilities"] = [[0.2], [0.5, 0.5]]
    path = tmp_path / "bad.jsonl"
    import json

    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="horizons"):
        read_episode_records_jsonl(path)

    path.write_text("{}\n")
    with pytest.raises(ValueError, match="invalid episode"):
        read_episode_records_jsonl(path)
