"""Atomic, finite, strict terminal checkpoints and metadata for ATTR-RTG."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _validate_metadata(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint metadata contains non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_metadata(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_metadata(item)
        return
    raise TypeError("checkpoint metadata must contain only JSON values")


def _cpu_finite_state(module: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().clone()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"non-finite checkpoint tensor: {name}")
        state[name] = tensor
    return state


def _atomic_create(path: str | Path, writer) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_terminal_checkpoint(
    path: str | Path,
    module: nn.Module,
    *,
    kind: str,
    training_seed: int,
    terminal_update: int = 1_000,
    metadata: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Create, never overwrite, one terminal checkpoint without optimizer state."""
    if not kind or type(training_seed) is not int or terminal_update < 1:
        raise ValueError("checkpoint identity is invalid")
    details = dict(metadata or {})
    _validate_metadata(details)
    payload: dict[str, Any] = {
        "kind": kind,
        "training_seed": training_seed,
        "terminal_update": terminal_update,
        "metadata": details,
        "model": _cpu_finite_state(module),
    }
    if config is not None:
        payload["config"] = dict(config)
        _validate_metadata(payload["config"])
    return _atomic_create(path, lambda stream: torch.save(payload, stream))


def load_terminal_checkpoint(
    path: str | Path,
    module: nn.Module,
    *,
    expected_kind: str,
    expected_seed: int,
    expected_update: int = 1_000,
    expected_metadata: Mapping[str, Any] | None = None,
    expected_config: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Validate exact identity/metadata/config and load state strictly."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"kind", "training_seed", "terminal_update", "metadata", "model"}
    expected_keys = required | ({"config"} if expected_config is not None else set())
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("terminal checkpoint fields differ from expected schema")
    expected = (expected_kind, expected_seed, expected_update, dict(expected_metadata or {}))
    actual = (payload["kind"], payload["training_seed"], payload["terminal_update"], payload["metadata"])
    if actual != expected or (expected_config is not None and payload["config"] != dict(expected_config)):
        raise ValueError("terminal checkpoint identity or metadata differs")
    _validate_metadata(payload["metadata"])
    if not isinstance(payload["model"], dict):
        raise TypeError("terminal model state is malformed")
    for value in payload["model"].values():
        if not isinstance(value, torch.Tensor) or (value.is_floating_point() and not torch.isfinite(value).all()):
            raise FloatingPointError("terminal model state is malformed or non-finite")
    module.load_state_dict(payload["model"], strict=True)
    module.eval()
    return module


def write_terminal_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Create canonical finite JSON metadata atomically without overwrite."""
    values = dict(metadata)
    _validate_metadata(values)
    encoded = (json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return _atomic_create(path, lambda stream: stream.write(encoded))
