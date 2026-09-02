import pytest
import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk.rtg_checkpoint import (
    load_terminal_checkpoint,
    save_terminal_checkpoint,
    write_terminal_metadata,
)


def test_terminal_checkpoint_is_atomic_strict_finite_and_no_overwrite(tmp_path):
    model = nn.Linear(2, 1)
    path = tmp_path / "terminal.pt"
    save_terminal_checkpoint(path, model, kind="toy-G", training_seed=13, terminal_update=2, metadata={"split": "toy"})
    restored = load_terminal_checkpoint(path, nn.Linear(2, 1), expected_kind="toy-G", expected_seed=13, expected_update=2, expected_metadata={"split": "toy"})
    assert not restored.training
    with pytest.raises(FileExistsError):
        save_terminal_checkpoint(path, model, kind="toy-G", training_seed=13, terminal_update=2)
    metadata = write_terminal_metadata(tmp_path / "metadata.json", {"finite": 1.0})
    assert metadata.read_text() == '{"finite":1.0}\n'
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        save_terminal_checkpoint(tmp_path / "bad.pt", model, kind="toy", training_seed=13)
