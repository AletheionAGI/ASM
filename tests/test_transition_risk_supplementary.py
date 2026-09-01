import torch
from aletheion_state_models.benchmarks.transition_risk.supplementary import (
    _build_arm,
    _build_backbone,
)


def _counts(model):
    return sum(parameter.numel() for parameter in model.parameters()), sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def test_cm_and_vr_s_are_total_parameter_matched_but_not_trainable_matched():
    cm, _, _ = _build_backbone("asm_cm_durable", 17)
    full, full_rank, _ = _build_backbone("asm_vr_s_full64", 17)
    fixed, fixed_rank, _ = _build_backbone("asm_vr_s_fixed32", 17)
    assert _counts(cm) == (274058, 274058)
    assert _counts(full) == _counts(fixed) == (274135, 269975)
    assert abs(_counts(cm)[0] / _counts(full)[0] - 1) < 0.001
    assert (full_rank, fixed_rank) == (64, 32)


def test_fixed32_head_input_has_no_inactive_state_payload():
    adapter, heads, metadata = _build_arm("asm_vr_s_fixed32", 17, 1000)
    input_ids = torch.randint(0, 256, (2, 64))
    states = adapter(input_ids)
    assert metadata["logical_rank"] == 32
    torch.testing.assert_close(states[..., 32:], torch.zeros_like(states[..., 32:]))
    assert heads(states)["hazard_logits"].shape == (2, 64, 4)
