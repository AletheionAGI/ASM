from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk.model_adapters import (
    ASMModelAdapter,
    TransformerModelAdapter,
)
from aletheion_state_models.benchmarks.transition_risk.model_heads import (
    TransitionRiskHeads,
)
from transformer.tiny_transformer import TinyTransformerConfig, TinyTransformerLM


def _tiny_transformer() -> TinyTransformerLM:
    torch.manual_seed(7)
    return TinyTransformerLM(
        TinyTransformerConfig(
            vocab_size=23,
            d_model=12,
            n_heads=3,
            n_layers=2,
            hidden_size=20,
            max_seq_len=12,
            dropout=0.0,
        )
    ).eval()


def test_transformer_hidden_states_are_opt_in_and_backward_compatible() -> None:
    model = _tiny_transformer()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    legacy = model(input_ids)
    exposed = model(input_ids, return_hidden_states=True)

    assert set(legacy) == {"logits"}
    assert exposed["hidden_states"].shape == (1, 4, model.config.d_model)
    torch.testing.assert_close(legacy["logits"], exposed["logits"])
    torch.testing.assert_close(
        model.lm_head(exposed["hidden_states"]), exposed["logits"]
    )


def test_transformer_hidden_states_are_causal() -> None:
    model = _tiny_transformer()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    changed_suffix = torch.tensor([[1, 2, 3, 9, 10]])

    first_states = model(first, return_hidden_states=True)["hidden_states"]
    changed_states = model(changed_suffix, return_hidden_states=True)["hidden_states"]

    torch.testing.assert_close(
        first_states[:, :3], changed_states[:, :3], atol=1e-6, rtol=1e-6
    )


class _FakeASM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_state=12)
        self.embedding = nn.Embedding(23, 12)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_states: bool,
        collect_diagnostics: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_states
        assert not collect_diagnostics
        return {"states": self.embedding(input_ids)}


def test_backbone_adapters_have_the_same_tensor_contract() -> None:
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    adapters = [
        ASMModelAdapter(_FakeASM()),
        TransformerModelAdapter(_tiny_transformer()),
    ]

    for adapter in adapters:
        representations = adapter(input_ids)
        assert representations.shape == (2, 3, 12)
        assert adapter.representation_dim == 12


def test_common_heads_cover_next_state_hazard_and_severity_on_cpu() -> None:
    representations = torch.randn(2, 5, 12, requires_grad=True)
    heads = TransitionRiskHeads(12, state_dim=6, horizons=(1, 4, 8, 16))

    predictions = heads(representations)

    assert predictions["next_state"]["mean"].shape == (2, 5, 6)
    assert predictions["next_state"]["log_scale"].shape == (2, 5, 6)
    assert predictions["hazard_logits"].shape == (2, 5, 4)
    assert predictions["severity"]["severity"].shape == (2, 5)
    assert predictions["severity"]["time_to_hazard"].shape == (2, 5)
    assert torch.all(predictions["severity"]["severity"] >= 0)
    assert all(parameter.device.type == "cpu" for parameter in heads.parameters())

    loss = sum(
        tensor.float().mean()
        for tensor in (
            predictions["next_state"]["mean"],
            predictions["hazard_logits"],
            predictions["severity"]["severity"],
        )
    )
    loss.backward()
    assert representations.grad is not None


class _GlobalStepASM(_FakeASM):
    def forward(self, input_ids, *, return_states, collect_diagnostics, global_step):
        assert global_step == 37
        return super().forward(
            input_ids,
            return_states=return_states,
            collect_diagnostics=collect_diagnostics,
        )


def test_asm_adapter_forwards_registered_global_step():
    adapter = ASMModelAdapter(_GlobalStepASM(), global_step=37)
    states = adapter(torch.tensor([[1, 2, 3]]))
    assert states.shape == (1, 3, 12)
