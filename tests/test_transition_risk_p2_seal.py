import json
from dataclasses import asdict

import pytest

from aletheion_state_models.benchmarks.transition_risk.dataset import make_worlds
from aletheion_state_models.benchmarks.transition_risk.p2_seal import (
    MODEL_ARMS,
    TRAINING_SEEDS,
    canonical_json,
    canonical_sha256,
    create_p2_seal,
    default_p2_specs,
    open_p2_seal,
    read_p2_seal,
    write_p2_seal,
)


def _checkpoints(tmp_path):
    result = {}
    for arm in MODEL_ARMS:
        for seed in TRAINING_SEEDS:
            path = tmp_path / f"{arm}-{seed}.pt"
            path.write_bytes(f"checkpoint:{arm}:{seed}".encode())
            result[(arm, seed)] = path
    return result


def test_registered_specs_are_deterministic_and_contain_no_generated_data():
    specs = default_p2_specs()
    assert specs == default_p2_specs()
    assert [spec.test_id for spec in specs] == ["test_id", "test_shift", "test_ood"]
    assert all(spec.world_count == 32 and spec.episodes_per_world == 4 for spec in specs)
    assert "episodes" not in asdict(specs[0])
    assert "labels" not in asdict(specs[0])
    assert (specs[1].sensor_noise, specs[1].forcing) == (0.12, 0.22)
    assert (specs[2].failure_delay, specs[2].recovery_window) == (1, 2)


def test_canonical_sha_is_independent_of_mapping_order():
    left = {"b": [2, 3], "a": 1}
    right = {"a": 1, "b": [2, 3]}
    assert canonical_json(left) == b'{"a":1,"b":[2,3]}'
    assert canonical_sha256(left) == canonical_sha256(right)


def test_seal_round_trip_is_byte_deterministic(tmp_path):
    checkpoints = _checkpoints(tmp_path)
    seal = create_p2_seal(checkpoints)
    first = write_p2_seal(seal, tmp_path / "first.json")
    loaded = read_p2_seal(first)
    second = write_p2_seal(loaded, tmp_path / "second.json")
    assert loaded == seal
    assert loaded.sha256 == seal.sha256
    assert first.read_bytes() == second.read_bytes()
    assert open_p2_seal(loaded, checkpoints) == default_p2_specs()


def test_open_fails_closed_for_missing_or_changed_checkpoint(tmp_path):
    checkpoints = _checkpoints(tmp_path)
    seal = create_p2_seal(checkpoints)
    missing = dict(checkpoints)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="exact six-arm"):
        open_p2_seal(seal, missing)
    changed = dict(checkpoints)
    changed[(MODEL_ARMS[0], TRAINING_SEEDS[0])].write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        open_p2_seal(seal, changed)


def test_read_rejects_tampered_seal(tmp_path):
    path = write_p2_seal(create_p2_seal(_checkpoints(tmp_path)), tmp_path / "seal.json")
    document = json.loads(path.read_text())
    document["payload"]["splits"][0]["seed"] += 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        read_p2_seal(path)


def test_shift_families_change_dynamics_without_changing_baseline():
    baseline = make_worlds(2, 17, max_steps=16)
    assert baseline == make_worlds(2, 17, dynamic_family="baseline", max_steps=16)
    shift = make_worlds(1, 17, dynamic_family="shift")[0]
    ood = make_worlds(1, 17, dynamic_family="ood")[0]
    assert (shift.sensor_noise, shift.forcing, shift.failure_delay) == (0.12, 0.22, 3)
    assert (ood.sensor_noise, ood.forcing, ood.failure_delay, ood.recovery_window) == (
        0.18, 0.16, 1, 2
    )
    with pytest.raises(ValueError, match="unknown"):
        make_worlds(1, 17, dynamic_family="unregistered")
