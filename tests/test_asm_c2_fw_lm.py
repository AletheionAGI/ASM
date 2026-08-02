import torch

from drm_language_emitter import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from scripts.train_asm_c2_fw_lm import distillation_loss, optimizer_groups


def tiny_model() -> DRMEmitterModel:
    return DRMEmitterModel(DRMConfig(
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
        fast_weight_state_fp32=True,
        fast_weight_compute_fp32=True,
    ))


def test_distillation_loss_is_zero_for_identical_logits():
    logits = torch.randn(2, 3, 7)
    loss = distillation_loss(logits, logits, 2.0)
    assert abs(float(loss)) < 1e-5


def test_optimizer_groups_separate_fast_weight_memory():
    groups = optimizer_groups(tiny_model(), 1e-5, 1e-4, 0.01)
    assert [group["group_name"] for group in groups] == [
        "language_backbone",
        "fast_weight_memory_and_gates",
    ]
    assert [group["lr"] for group in groups] == [1e-5, 1e-4]
    assert not set(map(id, groups[0]["params"])) & set(map(id, groups[1]["params"]))
