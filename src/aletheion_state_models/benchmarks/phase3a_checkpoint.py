"""Atomic local artifacts for Phase 3A training runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path

import torch


def atomic_torch_save(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def write_result(path: str | Path, result: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result) if is_dataclass(result) else result
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = ["atomic_torch_save", "write_result"]
