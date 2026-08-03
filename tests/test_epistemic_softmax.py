import pytest
import torch

from aletheion_state_models.variants import build_compact_epistemic_memory
from drm_language_emitter import DRMConfig
from drm_language_emitter.fast_weight_memory import FastWeightMemory
from drm_language_emitter.utils import EpistemicConfidenceGate, EpistemicSoftmax


def config(**overrides) -> DRMConfig:
    values = dict(
        vocab_size=101,
        d_token=8,
        d_state=12,
        n_directions=4,
        metric_rank=2,
        hidden_size=16,
        sequence_mode="directional_block_cumsum",
        directional_cumsum_step_mode="velocity",
        directional_cumsum_block_size=4,
        compact_streaming_inference=True,
        use_direction_field=False,
        addressable_memory=True,
        addressable_memory_backend="fast_weight",
        addressable_memory_dim=6,
        fast_weight_durable_memory=True,
        epistemic_memory_gating=True,
        epistemic_gate_hidden_dim=8,
        epistemic_gate_num_layers=1,
    )
    values.update(overrides)
    return DRMConfig(**values)


def test_epistemic_softmax_is_normalized_and_differentiable():
    module = EpistemicSoftmax(6, gate_hidden_dim=8, gate_num_layers=1)
    logits = torch.randn(3, 11, requires_grad=True)
    context = torch.randn(3, 6, requires_grad=True)
    probabilities, uncertainty, _, _ = module(logits, context)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.all((uncertainty >= 0) & (uncertainty <= 1))
    probabilities[:, 0].sum().backward()
    assert logits.grad is not None
    assert context.grad is not None


def test_confidence_gate_starts_near_requested_confidence():
    module = EpistemicConfidenceGate(5, hidden_dim=7, num_layers=1, initial_confidence=0.8)
    confidence, uncertainty, _, _ = module(torch.randn(4, 5))
    assert torch.allclose(confidence, torch.full_like(confidence, 0.8), atol=1e-5)
    assert torch.allclose(confidence + uncertainty, torch.ones_like(confidence))


def test_asm_cm_e_gates_fast_weight_reads_and_writes_with_gradients():
    module = FastWeightMemory(config())
    torch.nn.init.normal_(module.read_output.weight, std=0.1)
    states = torch.randn(2, 4, 12, requires_grad=True)
    tokens = torch.randn(2, 4, 8, requires_grad=True)
    output, _, diagnostics = module.forward_sequence(states, tokens)
    for name in (
        "epistemic_read_confidence",
        "epistemic_read_uncertainty",
        "epistemic_write_confidence",
        "epistemic_write_uncertainty",
    ):
        assert diagnostics[name].shape == (2, 4)
        assert torch.isfinite(diagnostics[name]).all()
    output.sum().backward()
    assert module.epistemic_read_gate.local_evidence.network[0].weight.grad is not None


def test_asm_cm_e_variant_preserves_bounded_streaming_state():
    model = build_compact_epistemic_memory(config(epistemic_memory_gating=False)).eval()
    logits, state = model.prefill(torch.randint(0, 101, (1, 4)))
    assert logits.shape == (1, 4, 101)
    assert model.config.epistemic_memory_gating is True
    for _ in range(8):
        _, state = model.decode_step(torch.randint(0, 101, (1, 1)), state)
    assert state.addressable_memory.matrix.shape == (1, 6, 6)
    assert state.input_ids.numel() == 0


def test_epistemic_gating_rejects_non_fast_weight_configuration():
    with pytest.raises(ValueError, match="requires the fast_weight"):
        config(addressable_memory_backend="slots")
