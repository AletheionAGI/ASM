from __future__ import annotations

import copy

import pytest

from aletheion_state_models.benchmarks.transition_risk import (
    rtg_registered_summary as summary,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_config import TRAINING_SEEDS

_EVIDENCE = object()


@pytest.fixture(autouse=True)
def _accept_toy_integrity_evidence(monkeypatch):
    monkeypatch.setattr(
        summary,
        "require_integrity_evidence",
        lambda evidence: (
            evidence is _EVIDENCE
            or (_ for _ in ()).throw(PermissionError("evidence required"))
        ),
    )


def _record(split, seed, origin, action, *, kind, head):
    unsafe = action == 0
    allow_unsafe = head == "C" and origin < 100
    if kind == "transformer" and head == "G":
        allow_unsafe = origin < 40
    predicted = [0.0] * 28 if kind == "asm" else [0.5] * 28
    nll = 0.1 if kind == "asm" else 0.2
    return {
        "seed": seed,
        "split_id": split,
        "world_id": "world",
        "episode_id": "episode",
        "t": origin,
        "action_index": action,
        "candidate_unsafe": unsafe,
        "brake_unsafe": False,
        "decision": "ALLOW" if not unsafe or allow_unsafe else "BLOCK",
        "risk": float(unsafe),
        "predicted_state": predicted,
        "true_state": [0.0] * 28,
        "persistence_state": [1.0] * 28,
        "group_nll": [nll] * 11,
        "persistence_nll": [1.0] * 11,
        "d_group_nll": [0.1] * 11,
        "group_predictions": [0] * 11,
        "d_group_predictions": [0] * 11,
        "persistence_predictions": [1] * 11,
        "group_targets": [0] * 11,
        "fixed_frame": [0.0] * 32,
        "persistence_target": [0] * 11,
        "failure_delay": None,
    }


def _evaluation():
    splits = {}
    for split in summary._SPLITS:
        systems = {}
        for kind in summary.BACKBONES:
            for seed in TRAINING_SEEDS:
                batches = {}
                for head in summary._HEADS:
                    records = [
                        _record(split, seed, origin, action, kind=kind, head=head)
                        for origin in range(200)
                        for action in range(6)
                    ]
                    batches[head] = {"records": records}
                systems[f"{kind}.seed{seed}"] = batches
        splits[split] = {"systems": systems}
    return {
        "kind": "attr_rtg_registered_evaluation",
        "schema_version": 1,
        "episodes_per_world": 4,
        "splits": splits,
    }


def _fake_bootstrap(records, estimator):
    metrics = estimator(tuple(records))
    intervals = {}
    for name, value in metrics.items():
        margin = 0.001
        intervals[name] = {"estimate": value, "ci95": [value - margin, value + margin]}
    return {
        "bootstrap_seed": 20260903,
        "replicates": 1000,
        "bit_generator": "PCG64",
        "cluster_order": "seed-world-episode",
        "metrics": intervals,
    }


def _fake_paired(left, right, estimator):
    wrappers = []
    for a, b in zip(left, right, strict=True):
        item = dict(a)
        item.update({"left": a, "right": b})
        wrappers.append(item)
    return _fake_bootstrap(wrappers, estimator)


def test_complete_toy_summary_uses_frozen_bootstraps_and_exact_gate_names(monkeypatch):
    monkeypatch.setattr(summary, "hierarchical_bootstrap", _fake_bootstrap)
    monkeypatch.setattr(summary, "paired_hierarchical_bootstrap", _fake_paired)
    result = summary.summarize_registered_evaluation(_evaluation(), _EVIDENCE)

    assert result["kind"] == "attr_rtg_registered_summary"
    assert set(result["gates"]) == {
        *(
            f"{kind}.{claim}"
            for kind in ("ASM", "Transformer")
            for claim in (
                "D",
                "RTG1-Z",
                "RTG1-Y",
                "RTG2-G",
                "RTG2-C",
                "RTG2-V",
                "RTG3-G-SHIFT",
                "RTG3-G-OOD",
                "RTG3-G",
                "RTG3-C-SHIFT",
                "RTG3-C-OOD",
                "RTG3-C",
                "RTG3-V-SHIFT",
                "RTG3-V-OOD",
                "RTG3-V",
            )
        ),
        "RTG1-Z-ARCH",
        "RTG1-Y-ARCH",
        "RTG2-G-ARCH",
    }
    assert all(result["gates"].values())
    evidence = result["evidence"]["absolute"]["test_id"]["asm"]["G"]
    assert evidence["bootstrap"] == {
        "bootstrap_seed": 20260903,
        "replicates": 1000,
        "bit_generator": "PCG64",
        "cluster_order": "seed-world-episode",
    }
    assert set(evidence["per_seed"]) == {str(seed) for seed in TRAINING_SEEDS}


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("origin", "200 origins"),
        ("class", "25 labels"),
        ("candidate", "six candidates"),
        ("pair", "lexicographic"),
        ("seed", "complete registered system matrix"),
    ],
)
def test_summary_fails_closed_on_registered_volume_classes_clusters_pairs_and_seeds(
    mutation,
    message,
):
    value = _evaluation()
    systems = value["splits"]["test_id"]["systems"]
    if mutation == "origin":
        for system in systems.values():
            for batch in system.values():
                batch["records"] = [row for row in batch["records"] if row["t"] != 199]
    elif mutation == "class":
        for system in systems.values():
            for batch in system.values():
                for row in batch["records"]:
                    row["candidate_unsafe"] = False
    elif mutation == "candidate":
        systems["asm.seed29"]["G"]["records"].pop()
    elif mutation == "pair":
        systems["asm.seed29"]["C"]["records"].reverse()
    else:
        del systems["asm.seed29"]
    with pytest.raises(ValueError, match=message):
        summary.summarize_registered_evaluation(value, _EVIDENCE)


def test_summary_does_not_mutate_registered_artifact(monkeypatch):
    value = _evaluation()
    before = copy.deepcopy(value)
    monkeypatch.setattr(summary, "hierarchical_bootstrap", _fake_bootstrap)
    monkeypatch.setattr(summary, "paired_hierarchical_bootstrap", _fake_paired)
    summary.summarize_registered_evaluation(value, _EVIDENCE)
    assert value == before


def test_architecture_gates_require_complete_other_arm_predecessors(monkeypatch):
    value = _evaluation()
    for split in summary._SPLITS:
        for seed in TRAINING_SEEDS:
            records = value["splits"][split]["systems"][f"transformer.seed{seed}"]["G"][
                "records"
            ]
            for row in records:
                row["d_group_predictions"] = [1] * 11
                row["persistence_predictions"] = [0] * 11
    monkeypatch.setattr(summary, "hierarchical_bootstrap", _fake_bootstrap)
    monkeypatch.setattr(summary, "paired_hierarchical_bootstrap", _fake_paired)
    gates = summary.summarize_registered_evaluation(value, _EVIDENCE)["gates"]
    assert gates["ASM.RTG1-Z"] and gates["Transformer.RTG1-Z"]
    assert not gates["Transformer.D"] and not gates["Transformer.RTG1-Y"]
    assert gates["RTG1-Z-ARCH"]
    assert not gates["RTG1-Y-ARCH"] and not gates["RTG2-G-ARCH"]
