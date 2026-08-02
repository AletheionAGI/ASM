import torch

from aletheion_state_models.core import AddressableMemory
from drm_language_emitter import DRMConfig


def config(**overrides) -> DRMConfig:
    values = dict(
        vocab_size=17,
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
        addressable_memory_slots=4,
        addressable_memory_dim=6,
    )
    values.update(overrides)
    return DRMConfig(**values)


def test_addressable_memory_shapes_probabilities_and_gradients():
    memory = AddressableMemory(config())
    states = torch.randn(2, 5, 12, requires_grad=True)
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    output, final, diagnostics = memory.forward_sequence(states, tokens)
    assert output.shape == states.shape
    assert final.keys.shape == (2, 4, 6)
    assert final.values.shape == (2, 4, 6)
    assert torch.isfinite(output).all()
    assert final.usage.min() >= 0 and final.usage.max() <= 1
    assert final.age.min() >= 0 and final.age.max() <= 1
    output.sum().backward()
    assert memory.query.weight.grad is not None
    assert memory.read_output.weight.grad is not None


def test_disabled_write_preserves_memory_state():
    module = AddressableMemory(config(addressable_memory_write_enabled=False))
    initial = module.initial_state(2, torch.device("cpu"), torch.float32)
    _, final, diagnostics = module.step(torch.randn(2, 12), torch.randn(2, 8), initial)
    assert torch.equal(final.keys, initial.keys)
    assert torch.equal(final.values, initial.values)
    assert torch.equal(final.usage, initial.usage)
    assert torch.equal(diagnostics["write_gate"], torch.zeros(2))


def test_disabled_read_does_not_change_state():
    module = AddressableMemory(config(addressable_memory_read_enabled=False))
    state = torch.randn(2, 12)
    initial = module.initial_state(2, torch.device("cpu"), torch.float32)
    output, _, diagnostics = module.step(state, torch.randn(2, 8), initial)
    assert torch.equal(output, state)
    assert torch.equal(diagnostics["read_gate"], torch.zeros(2))
