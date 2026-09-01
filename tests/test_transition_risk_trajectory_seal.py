import json
from dataclasses import asdict

import pytest
import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk.trajectory_checkpoint import (
    atomic_write_json,
    checkpoint_metadata,
    save_terminal_checkpoint,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_manifests import (
    ARMS,
    HORIZON,
    OPTIMIZER_SEEDS,
    K,
    default_trajectory_protocol,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_seal import (
    create_trajectory_preseal,
    create_trajectory_seal,
    open_trajectory_seal,
    read_trajectory_preseal,
    read_trajectory_seal,
    write_trajectory_preseal,
    write_trajectory_seal,
)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 3)


def _files(tmp_path):
    code = tmp_path / "runner.py"
    data = tmp_path / "common.json"
    code.write_text("registered code")
    data.write_text("fixed common data")
    return {"runner.py": code}, {"common.json": data}


def _checkpoints(tmp_path):
    paths = {}
    for arm in ARMS:
        for seed in OPTIMIZER_SEEDS:
            path = tmp_path / f"{arm}-{seed}.pt"
            save_terminal_checkpoint(
                path, Adapter(), nn.Linear(3, 1), checkpoint_metadata(arm, seed)
            )
            paths[(arm, seed)] = path
    return paths


def _sealed(tmp_path):
    code, data = _files(tmp_path)
    preseal = create_trajectory_preseal(code, data)
    preseal_path = write_trajectory_preseal(preseal, tmp_path / "preseal.json")
    checkpoints = _checkpoints(tmp_path)
    seal = create_trajectory_seal(read_trajectory_preseal(preseal_path), checkpoints)
    seal_path = write_trajectory_seal(seal, tmp_path / "seal.json")
    return code, data, checkpoints, preseal_path, seal_path


def test_protocol_is_exact_and_specs_do_not_contain_generated_test_data():
    protocol = default_trajectory_protocol()
    assert protocol.arms == ("asm_x_base", "transformer_base")
    assert protocol.optimizer_seeds == (29, 43, 71, 89, 107)
    assert (HORIZON, K) == (8, 256)
    assert [
        (s.name, s.seed, s.world_count, s.episodes_per_world) for s in protocol.splits
    ] == [
        ("train", 310001, 64, 4),
        ("validation", 320001, 16, 4),
        ("test_id", 330001, 32, 4),
        ("test_shift", 330002, 32, 4),
        ("test_ood", 330003, 32, 4),
    ]
    payload = asdict(protocol)
    split_keys = set().union(*(split.keys() for split in payload["splits"]))
    assert not {"labels", "observations", "worlds", "episodes"} & split_keys
    assert protocol.leakage_control.optimizer_seed_used_for_data is False
    assert (
        protocol.gates.delta_auprc_min,
        protocol.gates.delta_auprc_lower_ci_min,
    ) == (0.03, 0.0)
    assert protocol.gates.brier_delta_max == 0.01
    assert (protocol.bootstrap_seed, protocol.bootstrap_replicates) == (20260901, 1000)


def test_preseal_and_final_seal_are_canonical_and_open_once(tmp_path):
    code, data, checkpoints, preseal_path, seal_path = _sealed(tmp_path)
    assert read_trajectory_preseal(preseal_path).sha256
    assert len(read_trajectory_seal(seal_path).checkpoints) == 10
    specs = open_trajectory_seal(seal_path, checkpoints, code, data)
    assert [item.name for item in specs] == ["test_id", "test_shift", "test_ood"]
    with pytest.raises(FileExistsError):
        open_trajectory_seal(seal_path, checkpoints, code, data)


def test_missing_extra_and_checkpoint_tamper_fail_without_opening(tmp_path):
    code, data, checkpoints, _, seal_path = _sealed(tmp_path)
    missing = dict(checkpoints)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="exact two-arm"):
        open_trajectory_seal(seal_path, missing, code, data)
    changed = dict(checkpoints)
    victim = changed[(ARMS[0], OPTIMIZER_SEEDS[0])]
    victim.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        open_trajectory_seal(seal_path, changed, code, data)
    assert not seal_path.with_suffix(".json.opened").exists()


def test_manifest_and_seal_tamper_wrong_schema_and_nonfinite_rejected(tmp_path):
    code, data, checkpoints, _, seal_path = _sealed(tmp_path)
    document = json.loads(seal_path.read_text())
    document["payload"]["schema_version"] = 2
    seal_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="SHA256"):
        open_trajectory_seal(seal_path, checkpoints, code, data)
    with pytest.raises(ValueError, match="non-finite"):
        atomic_write_json(tmp_path / "nan.json", {"metric": float("nan")})


def test_nonfinite_checkpoint_is_rejected_before_sealing(tmp_path):
    code, data = _files(tmp_path)
    checkpoints = _checkpoints(tmp_path)
    victim = checkpoints[(ARMS[0], OPTIMIZER_SEEDS[0])]
    payload = torch.load(victim, map_location="cpu", weights_only=True)
    payload["model_state"]["weight"][0, 0] = float("inf")
    torch.save(payload, victim)
    with pytest.raises(ValueError, match="non-finite"):
        create_trajectory_seal(create_trajectory_preseal(code, data), checkpoints)


def test_atomic_json_refuses_overwrite(tmp_path):
    path = atomic_write_json(tmp_path / "artifact.json", {"a": 1})
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}
