"""Single-update exact checkpoint persistence."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def write_exact_checkpoint(
    root: Path,
    arm: str,
    seed: int,
    update: int,
    payload: bytes,
    *,
    synthetic: bool = False,
) -> tuple[Path, str]:
    if update != 2_000 and not synthetic:
        raise ValueError("only update-2,000 checkpoints are permitted")
    if not payload:
        raise ValueError("checkpoint payload is empty")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"seed-{seed}_{arm}_update-{update}.ckpt"
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path, hashlib.sha256(payload).hexdigest()
