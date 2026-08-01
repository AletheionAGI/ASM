from __future__ import annotations

from pathlib import Path

import torch

from .config import DRMConfig
from .model import DRMEmitterModel


def load_model(checkpoint: str | Path) -> DRMEmitterModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    missing = {"config", "model"} - set(payload)
    if missing:
        raise ValueError(f"checkpoint missing required key(s): {', '.join(sorted(missing))}")
    if not isinstance(payload["config"], dict):
        raise ValueError("checkpoint 'config' must be a dictionary")
    if not isinstance(payload["model"], dict):
        raise ValueError("checkpoint 'model' must be a state_dict dictionary")
    schema_version = payload.get(
        "schema_version",
        payload["config"].get("schema_version", 1),
    )
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError("checkpoint schema_version must be a positive integer")
    if schema_version > 2:
        raise ValueError(
            f"checkpoint schema_version {schema_version} is newer than supported version 2"
        )
    config_data = dict(payload["config"])
    config_data.setdefault("schema_version", schema_version)
    config = DRMConfig.from_dict(config_data)
    model = DRMEmitterModel(config)
    model.load_state_dict(payload["model"])
    model.eval()
    return model
