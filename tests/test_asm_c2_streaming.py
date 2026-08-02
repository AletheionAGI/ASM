from dataclasses import fields, is_dataclass

import torch

from aletheion_state_models.variants import build_compact_addressable
from drm_language_emitter import DRMConfig


def config() -> DRMConfig:
    return DRMConfig(
        vocab_size=17,
        d_token=8,
        d_state=12,
        n_directions=4,
        metric_rank=2,
        hidden_size=16,
        sequence_mode="directional_block_cumsum",
        directional_cumsum_step_mode="velocity",
        directional_cumsum_block_size=4,
        directional_local_mixer="causal_conv",
        directional_local_mixer_hidden_size=12,
        directional_local_mixer_kernel_size=3,
        token_state_residual=True,
        selective_memory=True,
        selective_memory_hidden_size=12,
        bounded_state=False,
        use_direction_field=False,
        addressable_memory_dim=6,
    )


def tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if is_dataclass(value):
        return sum(tensor_bytes(getattr(value, item.name)) for item in fields(value))
    return 0


def test_asm_c2_full_forward_is_prefix_causal():
    torch.manual_seed(3)
    model = build_compact_addressable(config(), slots=4).eval()
    torch.nn.init.normal_(model.addressable_memory.read_output.weight, std=0.1)
    left = torch.randint(0, 17, (2, 11))
    right = left.clone()
    right[:, 7:] = torch.randint(0, 17, right[:, 7:].shape)
    a = model(left, collect_diagnostics=False)["logits"]
    b = model(right, collect_diagnostics=False)["logits"]
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-6, rtol=1e-6)


def test_asm_c2_incremental_matches_full_forward_and_cache_is_bounded():
    torch.manual_seed(4)
    model = build_compact_addressable(config(), slots=4).eval()
    torch.nn.init.normal_(model.addressable_memory.read_output.weight, std=0.1)
    tokens = torch.randint(0, 17, (2, 17))
    expected = model(tokens, collect_diagnostics=False)["logits"]
    first, state = model.prefill(tokens[:, :3])
    rows = [first]
    boundary_sizes = []
    for position in range(3, tokens.shape[1]):
        logits, state = model.decode_step(tokens[:, position], state)
        rows.append(logits[:, None])
        assert state.input_ids.numel() == 0
        if state.sequence_length % state.block_size == 0:
            boundary_sizes.append(tensor_bytes(state))
    assert torch.allclose(torch.cat(rows, dim=1), expected, atol=1e-6, rtol=1e-6)
    assert len(set(boundary_sizes)) == 1
    assert state.addressable_memory is not None
    assert state.addressable_memory.keys.shape == (2, 4, 6)


def test_asm_c2_state_resume_reproduces_continuation():
    model = build_compact_addressable(config(), slots=4).eval()
    torch.nn.init.normal_(model.addressable_memory.read_output.weight, std=0.1)
    tokens = torch.randint(0, 17, (1, 12))
    _, state = model.prefill(tokens[:, :5])
    saved = state
    first = []
    second = []
    for position in range(5, 12):
        logits, state = model.decode_step(tokens[:, position], state)
        first.append(logits)
        logits, saved = model.decode_step(tokens[:, position], saved)
        second.append(logits)
    assert torch.equal(torch.stack(first), torch.stack(second))
