from pathlib import Path

import pytest
import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from scripts.evaluate_asm_r_checkpoint import checkpoint_audit
from scripts.evaluate_asm_r_mqar_curve import parse_milestones
from scripts.run_mqar_architecture_comparison import load_asm_r_without_memory


def asm_r_config() -> DRMConfig:
    return DRMConfig(
        vocab_size=128,
        d_token=16,
        d_state=16,
        n_directions=4,
        metric_rank=2,
        hidden_size=32,
        max_seq_len=32,
        use_direction_field=False,
        selective_memory=True,
        selective_memory_hidden_size=8,
        sequence_mode="directional_block_cumsum",
        directional_cumsum_step_mode="velocity",
        directional_cumsum_block_size=8,
    ).validated_copy()


def test_checkpoint_audit_accepts_finite_asm_r(tmp_path: Path) -> None:
    checkpoint = tmp_path / "asm_r.pt"
    torch.save(DRMEmitterModel(asm_r_config()).state_dict_with_config(), checkpoint)
    result = checkpoint_audit(checkpoint)
    assert result["is_asm_r"] is True
    assert result["invalid_values"] == 0
    assert result["parameter_count"] > 0


def test_checkpoint_audit_rejects_nonfinite_weights(tmp_path: Path) -> None:
    model = DRMEmitterModel(asm_r_config())
    payload = model.state_dict_with_config()
    first = next(value for value in payload["model"].values() if value.is_floating_point())
    first.reshape(-1)[0] = float("nan")
    checkpoint = tmp_path / "invalid.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="non-finite"):
        checkpoint_audit(checkpoint)


def test_checkpoint_audit_rejects_non_asm_r(tmp_path: Path) -> None:
    config = asm_r_config()
    config.use_direction_field = True
    checkpoint = tmp_path / "asm_x.pt"
    torch.save(DRMEmitterModel(config.validated_copy()).state_dict_with_config(), checkpoint)
    with pytest.raises(ValueError, match="not the promoted ASM-R"):
        checkpoint_audit(checkpoint)


def test_parse_mqar_milestones_sorts_and_deduplicates() -> None:
    assert parse_milestones("500,200,500,1000") == [200, 500, 1000]


def test_parse_mqar_milestones_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        parse_milestones("0,200")


def test_no_memory_control_preserves_compatible_pretrained_weights(tmp_path: Path) -> None:
    model = DRMEmitterModel(asm_r_config())
    checkpoint = tmp_path / "asm_r.pt"
    torch.save(model.state_dict_with_config(), checkpoint)
    control, removed = load_asm_r_without_memory(checkpoint)
    assert control.config.selective_memory is False
    assert removed
    assert all(key.startswith("selective_memory.") for key in removed)
    assert torch.equal(control.token_embedding.embedding.weight, model.token_embedding.embedding.weight)
