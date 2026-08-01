import torch

from drm_language_emitter import DRMConfig, DRMEmitterModel
from drm_language_emitter.generation import generate
from aletheion_state_models.variants import (
    build_causal_memory,
    build_direct_state,
    build_explicit_drm,
    build_metric_subspace,
    build_metric_frame,
    build_relational_state,
    build_selective_state,
)


def test_generation_extends_sequence():
    config = DRMConfig(vocab_size=11, d_token=8, d_state=12, n_directions=4, metric_rank=2, hidden_size=16, top_k=5)
    model = DRMEmitterModel(config)
    x = torch.randint(0, 11, (1, 4))
    out = generate(model, x, max_new_tokens=3)
    assert out.shape == (1, 7)
    assert out.max() < 11


def _block_config() -> DRMConfig:
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
    )


def test_prefill_and_decode_match_full_forward():
    torch.manual_seed(7)
    tokens = torch.randint(0, 17, (2, 7))
    for builder in (
        build_explicit_drm,
        build_metric_subspace,
        build_metric_frame,
        build_relational_state,
        build_direct_state,
        build_selective_state,
        build_causal_memory,
    ):
        model = builder(_block_config()).eval()
        expected = model(tokens, collect_diagnostics=False)["logits"]
        state = model.init_inference_state(tokens.shape[0], tokens.device)
        first_logits, state = model.prefill(tokens[:, :3], state)
        observed = [first_logits]
        for position in range(3, tokens.shape[1]):
            step_logits, state = model.decode_step(tokens[:, position], state)
            observed.append(step_logits.unsqueeze(1))
        actual = torch.cat(observed, dim=1)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_generation_supports_all_public_asm_variants():
    prompt = torch.randint(0, 17, (2, 5))
    for builder in (
        build_explicit_drm,
        build_metric_subspace,
        build_metric_frame,
        build_relational_state,
        build_direct_state,
        build_selective_state,
        build_causal_memory,
    ):
        model = builder(_block_config())
        output = generate(model, prompt, max_new_tokens=2, top_k=1)
        assert output.shape == (2, 7)
        assert output.max() < 17
