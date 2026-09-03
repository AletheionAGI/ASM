from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from attr_rtg_rcmz.constants import ARMS, CANDIDATES, TRAINING_SEEDS
from attr_rtg_rcmz.models import build_adapter
from attr_rtg_rcmz.official_training import _authorized_config
from attr_rtg_rcmz.recovery import (
    RecoveredCheckpoint,
    training_arms,
    validate_recovery_manifest,
    write_recovery_record,
)


def _wrong_tensor_shape(payload):
    name = next(iter(payload["model"]))
    tensor = payload["model"][name]
    payload["model"][name] = tensor.reshape(-1)[:1]


def _manifest(tmp_path: Path, *, mutation=None) -> Path:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    entries = []
    for arm in ARMS:
        config = _authorized_config(arm, 29)
        model = build_adapter(config)
        payload = {
            "update": 2000,
            "config": config.__dict__,
            "model": model.state_dict(),
            "optimizer": {},
        }
        if mutation is not None and arm == "R":
            mutation(payload)
        target = checkpoints / f"seed-29_{arm}_update-2000.ckpt"
        torch.save(payload, target)
        entries.append(
            {
                "arm": arm,
                "seed": 29,
                "update": 2000,
                "path": str(target.relative_to(tmp_path)),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    path = tmp_path / "trusted-recovery.json"
    path.write_text(json.dumps({"schema_version": 1, "checkpoints": entries}))
    return path


def test_complete_seed29_is_validated_recorded_and_not_retrained(tmp_path):
    manifest = _manifest(tmp_path)
    recovered = validate_recovery_manifest(manifest)
    assert set(recovered) == {29}
    assert training_arms(29, recovered) == ()
    assert training_arms(43, recovered) == ARMS
    record = write_recovery_record(tmp_path / "continued", manifest, recovered)
    document = json.loads(record.read_text())
    assert document["status"] == "VALIDATED"
    assert len(document["reused"]) == 4
    assert [item["seed"] for item in document["pending"]] == [43, 71, 89, 107]


def test_recovery_rejects_digest_mismatch_before_loading(tmp_path):
    manifest = _manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["checkpoints"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="digest differs"):
        validate_recovery_manifest(manifest)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda payload: payload.__setitem__("update", 1999), "update/config differs"),
        (
            lambda payload: payload["config"].__setitem__("hidden_size", 999),
            "update/config differs",
        ),
        (
            lambda payload: payload["model"].pop(next(iter(payload["model"]))),
            "model keys differ",
        ),
        (_wrong_tensor_shape, "tensor shape/dtype differs"),
    ],
)
def test_recovery_rejects_update_config_and_shape_contracts(
    tmp_path, mutation, message
):
    manifest = _manifest(tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match=message):
        validate_recovery_manifest(manifest)


def test_recovery_rejects_partial_seed_group(tmp_path):
    manifest = _manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["checkpoints"].pop()
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="complete R,CM,Z,T"):
        validate_recovery_manifest(manifest)


def test_all_twenty_recovered_checkpoints_are_score_only(monkeypatch, tmp_path):
    from attr_rtg_rcmz import (
        checkpoint,
        official_isolated,
        official_stats,
        official_training,
    )

    recovered = {}
    original_bytes = {}
    for seed in TRAINING_SEEDS:
        recovered[seed] = {}
        for arm in ARMS:
            path = tmp_path / "recovered" / f"seed-{seed}_{arm}.ckpt"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"terminal:{seed}:{arm}".encode())
            original_bytes[path] = path.read_bytes()
            recovered[seed][arm] = RecoveredCheckpoint(
                arm, seed, path, hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def forbidden(*args, **kwargs):
        raise AssertionError("score-only recovery attempted training/checkpoint write")

    scored = []

    def fake_broker(origins, refs, **kwargs):
        split = origins[0]
        scored.append(
            (split, tuple(refs), tuple(ref.checkpoint_ref for ref in refs.values()))
        )
        score = tuple(0.0 for _ in CANDIDATES)
        truth = tuple(0.0 for _ in CANDIDATES)
        batches = (SimpleNamespace(logits=(score,)),)
        frozen = tuple(SimpleNamespace(arm=arm, batches=batches) for arm in ARMS)
        return SimpleNamespace(
            scores=frozen,
            truth=(truth,),
            identities=((split, 0, 0, 0),),
        )

    monkeypatch.setattr(official_training, "_train", forbidden)
    monkeypatch.setattr(checkpoint, "write_exact_checkpoint", forbidden)
    monkeypatch.setattr(official_isolated, "training_order", lambda origins, seed: ())
    monkeypatch.setattr(
        official_isolated, "freeze_all_arms_then_join_truth", fake_broker
    )
    monkeypatch.setattr(official_stats, "calibrate", lambda records: (1.0, 0.5))
    monkeypatch.setattr(
        official_stats,
        "summarize",
        lambda records, temperature, tau, **metadata: {"status": "VALID", **metadata},
    )

    splits = ("train", "calibration", "test_id", "test_shift", "test_ood")
    data = {name: (name,) for name in splits}
    rows = official_isolated.train_and_score_isolated(
        data,
        tmp_path / "new-output",
        lambda event: None,
        updates=2_000,
        batch_size=64,
        device="cpu",
        recovered=recovered,
    )

    assert len(rows) == len(TRAINING_SEEDS) * len(ARMS) * 3
    assert {(row["seed"], row["arm"], row["regime"]) for row in rows} == {
        (seed, arm, regime)
        for seed in TRAINING_SEEDS
        for arm in ARMS
        for regime in ("ID", "shift", "OOD")
    }
    assert [split for split, _, _ in scored] == [
        split
        for _ in TRAINING_SEEDS
        for split in ("calibration", "test_id", "test_shift", "test_ood")
    ]
    assert all(arms == ARMS for _, arms, _ in scored)
    assert all(path.read_bytes() == payload for path, payload in original_bytes.items())
    assert not (tmp_path / "new-output" / "checkpoints").exists()


def test_cli_recovery_flags_are_paired_and_official_only(tmp_path):
    from attr_rtg_rcmz.cli import main

    manifest = tmp_path / "manifest.json"
    assert (
        main(
            [
                "--dry-run",
                "--output-dir",
                str(tmp_path / "one"),
                "--recover-completed",
                "--recovery-manifest",
                str(manifest),
            ]
        )
        == 2
    )
    assert (
        main(
            ["--official", "--output-dir", str(tmp_path / "two"), "--recover-completed"]
        )
        == 2
    )
