from pathlib import Path

import torch
import pytest

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


def test_load_model_migrates_legacy_schema_one_checkpoint(tmp_path: Path) -> None:
    config = DRMConfig(vocab_size=16, d_token=8, d_state=8, hidden_size=8)
    model = DRMEmitterModel(config)
    legacy_config = config.to_dict()
    legacy_config.pop("schema_version")
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"config": legacy_config, "model": model.state_dict()}, checkpoint)
    loaded = load_model(checkpoint)
    assert loaded.config.schema_version == 1
    assert loaded.config.vocab_size == 16


def test_load_model_rejects_future_checkpoint_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "future.pt"
    torch.save({"schema_version": 999, "config": {}, "model": {}}, checkpoint)
    with pytest.raises(ValueError, match="newer than supported"):
        load_model(checkpoint)
