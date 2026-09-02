from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from aletheion_state_models.benchmarks.transition_risk import (
    rtg_registered_evaluation as registered,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_calibration import (
    RtgCalibration,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_evaluator import (
    EvaluationBatch,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_normalization import (
    StateNormalization,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_seal import (
    PipelineSealPaths,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_state_records import (
    CandidateStateRecord,
)


def _paths() -> PipelineSealPaths:
    return PipelineSealPaths(
        ".", {}, "generator", "evaluator", {}, {}, {}, {}, {}, {}, {}
    )


def _normalization() -> StateNormalization:
    zero = torch.zeros(28, dtype=torch.float32)
    one = torch.ones(28, dtype=torch.float32)
    return StateNormalization(zero, one, zero, one)


def test_private_injection_exercises_complete_registered_matrix_without_test_worlds(
    monkeypatch,
):
    claimed = False

    def claim_once(*_args):
        nonlocal claimed
        if claimed:
            raise PermissionError("registered evaluation was already claimed")
        claimed = True

    monkeypatch.setattr(registered, "claim_evaluation_authority", claim_once)
    events = []

    def load_arm(root, paths, kind, seed):
        events.append(("load", kind, seed))
        calibration = RtgCalibration(1.0, 0.5)
        return registered._Arm(
            SimpleNamespace(),
            torch.eye(28),
            _normalization(),
            torch.nn.Identity(),
            torch.nn.Identity(),
            torch.nn.Identity(),
            calibration,
            calibration,
        )

    def prepare_origins(worlds, *, episodes_per_world, split_id, split_seed):
        assert episodes_per_world == 4
        events.append(("origins", split_id, split_seed))
        return (split_id,)

    def export_inputs(origins, exporter, projection):
        split_id = origins[0]
        events.append(("export", split_id))
        return (split_id,)

    def materialize_truth(origin):
        events.append(("truth", origin))
        return origin

    def attach_truths(inputs, origins):
        split_id = inputs[0]
        values = []
        for action_index in range(6):
            values.append(
                CandidateStateRecord(
                    split_id,
                    f"{split_id}-world",
                    f"{split_id}-episode",
                    1,
                    action_index,
                    torch.zeros(28),
                    torch.ones(28),
                    torch.zeros(32),
                    (0,) * 11,
                    (0,) * 11,
                    action_index == 4,
                    3,
                )
            )
        return tuple(values)

    def evaluate_g(rows, g, d, calibration):
        assert len(rows) == 6
        assert all(row["brake_unsafe"] is True for row in rows)
        return EvaluationBatch(tuple(rows), {"finite": 1.0})

    def evaluate_c(rows, c, calibration):
        return EvaluationBatch(tuple(rows), {"finite": 1.0})

    dependencies = registered._Dependencies(
        load_arm,
        prepare_origins,
        export_inputs,
        materialize_truth,
        attach_truths,
        evaluate_g,
        evaluate_c,
        lambda result, evidence: {"gates": {"toy": True}},
    )
    worlds = {
        name: (SimpleNamespace(world_id=f"{name}-world"),)
        for name in ("test_id", "test_shift", "test_ood")
    }
    arguments = (".", _paths(), worlds, object(), object(), "0" * 64)
    result = registered._evaluate_registered_test(
        *arguments, dependencies=dependencies, strict_registry=False
    )
    completed_events = tuple(events)
    with pytest.raises(PermissionError, match="already claimed"):
        registered._evaluate_registered_test(
            *arguments, dependencies=dependencies, strict_registry=False
        )
    assert tuple(events) == completed_events

    assert [event[0] for event in events[:10]] == ["load"] * 10
    assert [event[0] for event in events[10:13]] == ["origins"] * 3
    assert [event[0] for event in events[13:43]] == ["export"] * 30
    assert [event[0] for event in events[43:46]] == ["truth"] * 3
    assert set(result["splits"]) == {"test_id", "test_shift", "test_ood"}
    for split in result["splits"].values():
        assert len(split["systems"]) == 10
        assert set(next(iter(split["systems"].values()))) == {"G", "C"}


def test_public_api_has_no_callback_surface():
    assert tuple(inspect.signature(registered.evaluate_registered_test).parameters) == (
        "root",
        "paths",
        "split_worlds",
        "capability",
        "evidence",
        "seal_sha256",
    )


def test_export_failure_materializes_no_truth(monkeypatch):
    monkeypatch.setattr(registered, "claim_evaluation_authority", lambda *args: None)
    events = []
    calibration = RtgCalibration(1.0, 0.5)
    arm = registered._Arm(
        SimpleNamespace(),
        torch.eye(28),
        _normalization(),
        torch.nn.Identity(),
        torch.nn.Identity(),
        torch.nn.Identity(),
        calibration,
        calibration,
    )

    def fail_export(origins, exporter, projection):
        events.append("export")
        if len(events) == 7:
            raise RuntimeError("synthetic export failure")
        return origins

    dependencies = registered._Dependencies(
        lambda *args: arm,
        lambda worlds, **kwargs: (kwargs["split_id"],),
        fail_export,
        lambda origin: events.append("truth") or origin,
        lambda inputs, origins: (),
        lambda *args: EvaluationBatch((), {}),
        lambda *args: EvaluationBatch((), {}),
        lambda result, evidence: {"gates": {}},
    )
    worlds = {
        name: (SimpleNamespace(world_id=f"{name}-world"),)
        for name in ("test_id", "test_shift", "test_ood")
    }
    try:
        registered._evaluate_registered_test(
            ".",
            _paths(),
            worlds,
            object(),
            object(),
            "0" * 64,
            dependencies=dependencies,
            strict_registry=False,
        )
    except RuntimeError as error:
        assert str(error) == "synthetic export failure"
    else:
        raise AssertionError("export failure must propagate")
    assert "truth" not in events


def test_private_evaluator_checks_authority_before_loading_or_export(monkeypatch):
    events = []

    def reject(*args):
        events.append("authority")
        raise PermissionError("missing evidence")

    monkeypatch.setattr(registered, "claim_evaluation_authority", reject)
    dependencies = registered._Dependencies(
        lambda *args: events.append("load"),
        lambda *args, **kwargs: events.append("rollout"),
        lambda *args: events.append("export"),
        lambda *args: events.append("truth"),
        lambda *args: (),
        lambda *args: EvaluationBatch((), {}),
        lambda *args: EvaluationBatch((), {}),
        lambda *args: {"gates": {}},
    )
    with pytest.raises(PermissionError, match="missing evidence"):
        registered._evaluate_registered_test(
            ".",
            _paths(),
            {},
            object(),
            object(),
            "0" * 64,
            dependencies=dependencies,
            strict_registry=False,
        )
    assert events == ["authority"]
