from types import SimpleNamespace

import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk import (
    rtg_pipeline_train as pipeline,
)


def data():
    return tuple(
        SimpleNamespace(name=name, origins=(SimpleNamespace(split_id=name),))
        for name in ("train", "validation", "calibration")
    )


def models():
    return {
        pipeline._arm_name(kind, seed): nn.Identity()
        for kind, seed in pipeline.ARMS
    }


def record(split):
    return SimpleNamespace(
        split_id=split, world_id=f"{split}-world", episode_id=f"{split}-episode",
        t=1, action_index=0, pre_state=torch.zeros(28),
        next_state=torch.ones(28), fixed_frame=torch.zeros(32),
    )


def test_all_ten_by_three_exports_finish_before_first_truth(monkeypatch, tmp_path):
    events = []

    def export(origins, exporter, projection):
        events.append("export")
        return (record(origins[0].split_id),)

    def materialize(origin):
        assert events[:30] == ["export"] * 30
        assert "truth" not in events[:30]
        assert len(tuple(tmp_path.glob("*_seed*/state_inputs_*.pt"))) == 30
        events.append("truth")
        return origin

    monkeypatch.setattr(pipeline, "export_candidate_state_inputs", export)
    monkeypatch.setattr(pipeline, "materialize_origin_truth", materialize)
    exports, input_paths, truths = pipeline._causal_export_then_truth(
        data(), models(), tmp_path
    )
    assert len(exports) == 10
    assert len(input_paths) == 30
    assert set(truths) == {"train", "validation", "calibration"}
    assert events == ["export"] * 30 + ["truth"] * 3


def test_export_interruption_materializes_nothing_and_writes_nothing(monkeypatch, tmp_path):
    calls = {"export": 0, "truth": 0}

    def interrupt(origins, exporter, projection):
        calls["export"] += 1
        if calls["export"] == 20:
            raise RuntimeError("export interrupted")
        return (record(origins[0].split_id),)

    def forbidden(origin):
        calls["truth"] += 1

    monkeypatch.setattr(pipeline, "export_candidate_state_inputs", interrupt)
    monkeypatch.setattr(pipeline, "materialize_origin_truth", forbidden)
    try:
        pipeline._causal_export_then_truth(data(), models(), tmp_path)
    except RuntimeError:
        pass
    assert calls == {"export": 20, "truth": 0}
    assert list(tmp_path.iterdir()) == []


def test_input_ledger_schema_has_no_privileged_fields(tmp_path):
    item = SimpleNamespace(
        split_id="train", world_id="w", episode_id="e", t=1, action_index=0,
        pre_state=torch.zeros(28), next_state=torch.ones(28), fixed_frame=torch.zeros(32),
    )
    path = pipeline._save_input_records(tmp_path / "inputs.pt", (item,))
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"schema", "identity", "pre_state", "next_state", "fixed_frame"}
    assert not {"truth", "unsafe", "persistence_target", "failure_delay"}.intersection(payload)
