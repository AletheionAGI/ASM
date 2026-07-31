import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model_components import SelectiveStateMemory


def test_parallel_selective_memory_matches_sequential_recurrence():
    config = DRMConfig(
        d_token=4,
        d_state=4,
        hidden_size=8,
        selective_memory=True,
        selective_memory_hidden_size=6,
        selective_memory_scale=1.0,
        bounded_state=False,
    )
    memory = SelectiveStateMemory(config)
    z_start = torch.randn(2, 4)
    states = torch.randn(2, 5, 4)
    tokens = torch.randn(2, 5, 4)

    previous = torch.cat([z_start.unsqueeze(1), states[:, :-1]], dim=1)
    hidden = torch.nn.functional.silu(
        memory.input_proj(torch.cat([previous, tokens], dim=-1))
    )
    forget = torch.sigmoid(memory.forget_head(hidden) + config.selective_memory_forget_bias)
    write = torch.sigmoid(memory.write_head(hidden))
    candidate = torch.tanh(memory.candidate_head(hidden))
    current = z_start
    expected = []
    for index in range(states.shape[1]):
        current = forget[:, index] * current + write[:, index] * candidate[:, index]
        expected.append(current)
    expected_states = states + torch.stack(expected, dim=1)

    actual = memory(z_start, states, tokens)
    assert torch.allclose(actual, expected_states, atol=1e-5, rtol=1e-5)


def test_selective_memory_long_sequence_is_finite_with_small_forget_gates():
    batch, seq_len, width = 2, 512, 8
    forget = torch.full((batch, seq_len, width), 1e-4)
    update = torch.randn(batch, seq_len, width)
    initial = torch.randn(batch, width)

    actual = SelectiveStateMemory._affine_scan(forget, update, initial)

    current = initial
    expected = []
    for index in range(seq_len):
        current = forget[:, index] * current + update[:, index]
        expected.append(current)
    expected = torch.stack(expected, dim=1)

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
