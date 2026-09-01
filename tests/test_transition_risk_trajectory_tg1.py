from __future__ import annotations

import json
from dataclasses import replace

import pytest

from aletheion_state_models.benchmarks.transition_risk.trajectory_evaluation import (
    PhysicalTrajectorySample,
    TrajectoryIdentity,
    TrajectoryRecord,
    evaluate_physical_trajectories,
    read_records_jsonl,
    validate_record,
    write_records_jsonl,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_plots import (
    render_trajectory_grounded,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_statistics import (
    build_summary,
    fit_fpr_thresholds,
    paired_hierarchical_bootstrap,
    quality_by_horizon,
    simple_lead,
    threshold_metrics,
)


def _identity(arm="candidate", split="test_id", episode="e0", anchor=0, seed=29):
    return TrajectoryIdentity(seed, arm, split, "w0", episode, anchor)


def _sample(hit=(), offset=0.0):
    return PhysicalTrajectorySample(
        tuple(index in hit for index in range(8)),
        tuple(
            {"x": offset + index / 10, "energy": -0.1 + offset} for index in range(8)
        ),
    )


def _record(
    *,
    arm="candidate",
    split="test_id",
    episode="e0",
    anchor=0,
    truth=(False,) * 8,
    risks=(0.1, 0.2, 0.3),
    nll=1.0,
    seed=29,
):
    return TrajectoryRecord(
        _identity(arm, split, episode, anchor, seed),
        dict(zip((1, 4, 8), risks)),
        tuple(truth),
        {
            h: {
                "x": float(nll + h / 100),
                "energy": float(nll),
                "joint": float(2 * nll + h / 100),
            }
            for h in (1, 4, 8)
        },
        4,
    )


def test_physical_evaluator_derives_prefix_risk_and_mean_nll():
    samples = (_sample(()), _sample((3,), 1), _sample((7,), 2), _sample((0,), 3))
    record = evaluate_physical_trajectories(
        _identity(), (False,) * 7 + (True,), lambda _identity: samples
    )
    assert record.risk_by_horizon == {1: 0.25, 4: 0.5, 8: 0.75}
    assert record.trajectory_nll[1]["x"] == pytest.approx(1.5)
    assert record.trajectory_nll[4]["x"] == pytest.approx(1.65)
    assert record.trajectory_nll[8]["energy"] == pytest.approx(1.4)
    assert (
        "HazardHead"
        not in __import__(
            "aletheion_state_models.benchmarks.transition_risk.trajectory_evaluation",
            fromlist=["dummy"],
        ).__dict__
    )


@pytest.mark.parametrize(
    "change",
    [
        {"physical_sample_count": True},
        {"unsafe_truth": (False,) * 7},
        {"risk_by_horizon": {1: 0.4, 4: 0.3, 8: 0.8}},
        {"risk_by_horizon": {1: 0.1, 4: float("nan"), 8: 0.8}},
        {"trajectory_nll": {1: {"x": 1.0}, 4: {"y": 1.0}, 8: {"x": 1.0}}},
    ],
)
def test_record_validation_is_strict(change):
    with pytest.raises(ValueError):
        validate_record(replace(_record(), **change))


def test_jsonl_round_trip_atomic_failure_and_duplicate_reader(tmp_path):
    path = tmp_path / "records.jsonl"
    original = _record()
    write_records_jsonl(path, [original])
    assert read_records_jsonl(path) == (original,)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_records_jsonl(path, [replace(original, physical_sample_count=0)])
    assert path.read_bytes() == before
    payload = original.to_dict()
    payload["unexpected"] = 1
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        read_records_jsonl(path)
    line = json.dumps(original.to_dict())
    path.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_records_jsonl(path)


def _metric_rows(arm="candidate", split="test_id"):
    return [
        _record(
            arm=arm,
            split=split,
            episode=f"e{i}",
            truth=((i % 2 == 0),) + (False,) * 7,
            risks=(score, score, score),
            nll=score,
        )
        for i, score in enumerate((0.9, 0.8, 0.7, 0.1))
    ]


def test_quality_thresholds_and_validation_constraint():
    validation = _metric_rows(split="validation")
    thresholds = fit_fpr_thresholds(validation)
    assert thresholds == {1: 0.9, 4: 0.9, 8: 0.9}
    result = threshold_metrics(_metric_rows(), thresholds)
    assert result["H8"]["fpr"] == 0
    assert result["H8"]["recall"] == 0.5
    quality = quality_by_horizon(_metric_rows())
    assert quality["H1"]["prevalence"] == 0.5
    assert set(quality["H8"]) >= {"auprc", "brier", "event_logloss", "trajectory_nll"}
    with pytest.raises(ValueError, match="validation"):
        fit_fpr_thresholds(_metric_rows())


def test_simple_lead_groups_anchors_by_episode():
    truth0 = (False, False, False, True) + (False,) * 4
    rows = [
        _record(episode="event", anchor=0, truth=truth0, risks=(0.2, 0.2, 0.8)),
        _record(
            episode="event",
            anchor=1,
            truth=(False, False, True) + (False,) * 5,
            risks=(0.2, 0.2, 0.9),
        ),
        _record(episode="safe", anchor=0, risks=(0.9, 0.9, 0.9)),
    ]
    lead = simple_lead(rows, 0.75)
    assert lead == {
        "event_episodes": 1,
        "detected_episodes": 1,
        "detection_rate": 1.0,
        "mean_lead_steps": 3.0,
    }


def test_paired_bootstrap_is_deterministic_and_reports_nll():
    left, right = [], []
    for seed in (29, 43, 71, 89, 107):
        for episode in ("e0", "e1"):
            for anchor in (0, 1):
                truth = (episode == "e0",) + (False,) * 7
                left.append(
                    _record(
                        seed=seed,
                        episode=episode,
                        anchor=anchor,
                        truth=truth,
                        risks=(0.9, 0.9, 0.9),
                        nll=0.5,
                    )
                )
                right.append(
                    _record(
                        seed=seed,
                        arm="reference",
                        episode=episode,
                        anchor=anchor,
                        truth=truth,
                        risks=(0.5, 0.5, 0.5),
                        nll=1.0,
                    )
                )
    one = paired_hierarchical_bootstrap(left, right, replicates=20, seed=7)
    two = paired_hierarchical_bootstrap(left, right, replicates=20, seed=7)
    assert one == two
    assert one["replicates"] == 20
    assert one["trajectory_nll"]["H8"]["x"]["delta"] == pytest.approx(-0.5)
    broken = list(right)
    broken[0] = replace(broken[0], unsafe_truth=(False,) * 8)
    with pytest.raises(ValueError, match="truth"):
        paired_hierarchical_bootstrap(left, broken, replicates=2)


def test_render_complete_separate_asset_set(tmp_path):
    rows = _metric_rows()
    summary = build_summary(rows, thresholds={1: 0.9, 4: 0.9, 8: 0.9})
    reference = [
        replace(row, identity=replace(row.identity, arm="reference")) for row in rows
    ]
    summary["paired_deltas"] = paired_hierarchical_bootstrap(
        rows, reference, replicates=2
    )
    summary["splits"] = {
        split: {
            "asm_x_base": build_summary(rows, thresholds={1: 0.9, 4: 0.9, 8: 0.9}),
            "transformer_base": build_summary(
                reference, thresholds={1: 0.9, 4: 0.9, 8: 0.9}
            ),
        }
        for split in ("test_id", "test_shift", "test_ood")
    }
    summary["gates"] = {"TG0_integrity": True}
    paths = render_trajectory_grounded(summary, tmp_path / "trajectory")
    names = {path.name for path in paths}
    assert names == {
        "trajectory_grounded_anticipation.png",
        "trajectory_grounded_anticipation.svg",
        "quality_by_horizon.png",
        "quality_by_horizon.svg",
        "paired_deltas.png",
        "paired_deltas.svg",
        "summary.json",
        "index.html",
    }
    dashboard = (tmp_path / "trajectory" / "index.html").read_text()
    assert "physical trajectories" in dashboard
    assert "No HazardHead" in dashboard
