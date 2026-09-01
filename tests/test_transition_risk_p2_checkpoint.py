import torch
from torch import nn
from aletheion_state_models.benchmarks.transition_risk.p2_checkpoint import (
    file_sha256,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(3, 4)


def test_terminal_checkpoint_round_trip_and_hash(tmp_path):
    torch.manual_seed(3)
    adapter = Adapter()
    heads = nn.Linear(4, 2)
    path = tmp_path / "terminal.pt"
    record = save_terminal_checkpoint(
        path, adapter, heads, {"seed": 29, "test_opened": False}
    )
    assert record["sha256"] == file_sha256(path)
    expected_model = {
        name: value.clone() for name, value in adapter.model.state_dict().items()
    }
    with torch.no_grad():
        for parameter in adapter.model.parameters():
            parameter.zero_()
    metadata = load_terminal_checkpoint(path, adapter, heads, device="cpu")
    assert metadata == {"seed": 29, "test_opened": False}
    for name, value in adapter.model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name])
