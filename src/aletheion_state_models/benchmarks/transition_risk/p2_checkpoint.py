"""Atomic terminal checkpoints for ATTR P2 model and common heads."""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def save_terminal_checkpoint(
    path: str | Path,
    adapter: torch.nn.Module,
    heads: torch.nn.Module,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Persist one terminal checkpoint without optimizer or test information."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state": _cpu_state(adapter.model),
        "heads_state": _cpu_state(heads),
        "metadata": metadata,
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": path.as_posix(), "sha256": file_sha256(path), "metadata": metadata}


def load_terminal_checkpoint(
    path: str | Path,
    adapter: torch.nn.Module,
    heads: torch.nn.Module,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if set(payload) != {"model_state", "heads_state", "metadata"}:
        raise ValueError("invalid ATTR P2 terminal checkpoint payload")
    adapter.model.load_state_dict(payload["model_state"], strict=True)
    heads.load_state_dict(payload["heads_state"], strict=True)
    adapter.to(device)
    heads.to(device)
    return payload["metadata"]


__all__ = ["file_sha256", "load_terminal_checkpoint", "save_terminal_checkpoint"]
