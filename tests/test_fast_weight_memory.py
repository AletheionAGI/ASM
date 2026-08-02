import torch

from aletheion_state_models.core import FastWeightMemory
from aletheion_state_models.variants import build_compact_fast_weight
from drm_language_emitter import DRMConfig


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
    )
    values.update(overrides)
    return DRMConfig(**values)


def test_fast_weight_memory_has_bounded_state_and_gradients():
    module = FastWeightMemory(config())
    states = torch.randn(2, 5, 12, requires_grad=True)
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    output, final, diagnostics = module.forward_sequence(states, tokens)
    assert output.shape == states.shape
    assert final.matrix.shape == (2, 6, 6)
    assert final.consolidated.shape == (2, 6, 6)
    assert final.previous_token.shape == (2, 8)
    assert diagnostics["read_gate"].shape == (2, 5)
    output.sum().backward()
    assert module.key.weight.grad is not None
    assert module.read_output.weight.grad is not None


def test_fast_weight_disabled_write_preserves_matrix():
    module = FastWeightMemory(config(addressable_memory_write_enabled=False))
    initial = module.initial_state(2, torch.device("cpu"), torch.float32)
    _, final, diagnostics = module.step(torch.randn(2, 12), torch.randn(2, 8), initial)
    assert torch.equal(final.matrix, initial.matrix)
    assert torch.equal(diagnostics["write_gate"], torch.zeros(2))


def test_durable_fast_weight_keeps_fp32_consolidated_state():
    module = FastWeightMemory(config(
        fast_weight_durable_memory=True,
        fast_weight_state_fp32=True,
        fast_weight_compute_fp32=True,
        fast_weight_hard_write_threshold=0.5,
    ))
    initial = module.initial_state(2, torch.device("cpu"), torch.bfloat16)
    assert initial.matrix.dtype == torch.float32
    assert initial.consolidated.dtype == torch.float32
    assert initial.previous_token.dtype == torch.float32
    _, final, diagnostics = module.step(
        torch.randn(2, 12), torch.randn(2, 8), initial
    )
    assert final.consolidated.shape == (2, 6, 6)
    assert torch.isfinite(final.consolidated).all()
    assert "consolidation_gate" in diagnostics


def test_durable_fast_weight_fp32_compute_preserves_caller_dtype():
    module = FastWeightMemory(config(
        fast_weight_durable_memory=True,
        fast_weight_state_fp32=True,
        fast_weight_compute_fp32=True,
    )).to(dtype=torch.float32)
    initial = module.initial_state(2, torch.device("cpu"), torch.bfloat16)
    output, final, _ = module.step(
        torch.randn(2, 12, dtype=torch.bfloat16),
        torch.randn(2, 8, dtype=torch.bfloat16),
        initial,
    )
    assert output.dtype == torch.bfloat16
    assert final.matrix.dtype == torch.float32
    assert final.consolidated.dtype == torch.float32
    assert final.previous_token.dtype == torch.float32


def test_compact_fast_weight_variant_streams_with_bounded_cache():
    model = build_compact_fast_weight(config(addressable_memory=False)).eval()
    logits, state = model.prefill(torch.randint(0, 101, (1, 4)))
    assert logits.shape == (1, 4, 101)
    for _ in range(12):
        _, state = model.decode_step(torch.randint(0, 101, (1, 1)), state)
    assert state.addressable_memory is not None
    assert state.addressable_memory.matrix.shape == (1, 6, 6)
    assert state.input_ids.numel() == 0
