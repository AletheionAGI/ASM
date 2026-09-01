import json
import shutil
from pathlib import Path

import torch

from aletheion_state_models.benchmarks.transition_risk.dataset import (
    collate_episodes,
    gather_step_representations,
    make_episodes,
    make_worlds,
)
from aletheion_state_models.benchmarks.transition_risk.p2_risk_mass_models import (
    build_risk_mass_arm,
    verify_parameter_and_initialization_parity,
)
from aletheion_state_models.benchmarks.transition_risk.p2_risk_mass_runner import (
    ensure_extension_manifest,
)
from aletheion_state_models.benchmarks.transition_risk.training import attr_loss

ROOT = Path(__file__).resolve().parents[1]


def test_native_risk_mass_preserves_parameters_and_initialization():
    audit = verify_parameter_and_initialization_parity(ROOT)
    assert audit["baseline_parameters"] == audit["variant_parameters"] == 219_610
    assert audit["identical_initial_tensors"] is True


def test_attr_objective_backpropagates_into_native_risk_mass():
    adapter, heads, _ = build_risk_mass_arm(ROOT, seed=29, updates=1_000)
    episodes = make_episodes(make_worlds(1, 29, max_steps=4), 1, 29)
    batch = collate_episodes(episodes)
    representations = adapter(batch["input_ids"])
    steps = gather_step_representations(representations, batch["step_positions"])
    loss, _ = attr_loss(heads(steps), batch)
    loss.backward()
    gradients = [parameter.grad for parameter in adapter.model.risk.parameters()]
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


def test_extension_manifest_is_frozen_before_training(tmp_path):
    original = ROOT / "docs/benchmarks/asm_transformer_transition_risk/p2"
    output = tmp_path / "p2"
    output.mkdir()
    for name in ("test_spec_preseal.json", "dataset_seal.json", "test_open_event.json"):
        shutil.copy2(original / name, output / name)
    run_root = ROOT / "runs/attr_p2"
    document = ensure_extension_manifest(ROOT, output, run_root)
    assert (
        document["payload"]["confirmatory_status"]
        == "posthoc_exploratory_not_registered_p2_arm"
    )
    path = output / "risk_mass_extension_pretrain_manifest.json"
    changed = json.loads(path.read_text())
    changed["payload"]["updates_per_arm"] = 999
    path.write_text(json.dumps(changed))
    try:
        ensure_extension_manifest(ROOT, output, run_root)
    except ValueError as error:
        assert "changed after freezing" in str(error)
    else:
        raise AssertionError("tampered extension manifest must fail closed")
