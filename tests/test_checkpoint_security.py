from pathlib import Path

import torch

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel


def test_load_model_accepts_weights_only_checkpoint(tmp_path: Path) -> None:
    config = DRMConfig(
        vocab_size=17,
        d_token=8,
        d_state=12,
        n_directions=4,
        metric_rank=2,
        hidden_size=16,
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(DRMEmitterModel(config).state_dict_with_config(), checkpoint)

    loaded = load_model(checkpoint)

    assert loaded.config == config
    assert not loaded.training
