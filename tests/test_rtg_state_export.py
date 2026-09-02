import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk.rtg_state_export import (
    ASMStateExporter,
    TransformerReadoutExporter,
)


class ToyASM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, ids, **options):
        self.calls.append((ids.clone(), options, torch.is_grad_enabled(), self.training))
        return {"states": ids.unsqueeze(-1).expand(-1, -1, 28).float() + self.anchor}


class ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, ids, return_hidden_states=False):
        assert return_hidden_states
        return {"hidden_states": ids.unsqueeze(-1).expand(-1, -1, 32).float() + self.anchor}


def test_asm_export_recomputes_pre_and_post_in_eval_without_grad():
    model = ToyASM()
    exporter = ASMStateExporter(model)
    history = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    frame = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    pre, post = exporter.export_transition(history, frame)
    assert (pre.shape, post.shape) == ((1, 28), (1, 28))
    assert [call[0].shape[1] for call in model.calls] == [4, 8]
    assert all(call[1]["global_step"] == 1_000 for call in model.calls)
    assert all(not call[2] and not call[3] for call in model.calls)


def test_transformer_export_returns_final_post_norm_readout():
    readout = TransformerReadoutExporter(ToyTransformer()).export(torch.tensor([[1, 2]], dtype=torch.long))
    assert readout.shape == (1, 32)
    assert torch.all(readout == 2)
