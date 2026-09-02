"""Stable locked official backend entry points used by the terminal CLI."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

Progress = Callable[[dict[str, object]], None]
_LOCKED = "LOCAL PROTOCOL LOCK"


def run_official(
    output_dir: Path, progress_callback: Progress | None, lock: Mapping[str, object]
) -> list[dict[str, object]]:
    """Run the registered CUDA pipeline after the CLI supplies a verified lock."""
    _require_lock(lock)
    from .policy import prepare_deterministic_environment

    prepare_deterministic_environment()
    callback = progress_callback or (lambda event: None)
    callback({"phase": "generating-registered-worlds"})
    from .official_data import generate_registered_origins
    from .official_training import train_and_score

    data = generate_registered_origins(miniature=False, lock=dict(lock))
    callback({"phase": "training", "total_updates": 2_000})
    metric_rows = train_and_score(data, Path(output_dir), callback, lock=dict(lock))
    rows = metric_rows + _contrast_rows(metric_rows)
    from .official_contrasts import strip_sufficient

    strip_sufficient(metric_rows)
    _write_rows(Path(output_dir), rows, official=True, lock_sha256=str(lock["sha256"]))
    callback({"phase": "completed", "rows": len(rows)})
    return rows


def run_smoke_official(
    output_dir: Path, progress_callback: Progress | None = None
) -> list[dict[str, object]]:
    """Run a tiny labeled synthetic integration path; it is never official."""
    callback = progress_callback or (lambda event: None)
    callback({"phase": "synthetic-smoke-worlds"})
    from .official_data import generate_registered_origins
    from .official_training import train_and_score

    data = generate_registered_origins(miniature=True)
    rows = train_and_score(
        data, Path(output_dir), callback, updates=2, batch_size=2, miniature=True
    )
    from .official_contrasts import strip_sufficient

    strip_sufficient(rows)
    for row in rows:
        row["status"] = "SYNTHETIC"
        row["notice"] = "SYNTHETIC NON-OFFICIAL — NOT AN OFFICIAL WORLD OR RESULT"
    _write_rows(Path(output_dir), rows, official=False, lock_sha256=None)
    callback({"phase": "synthetic-smoke-completed", "rows": len(rows)})
    return rows


def _require_lock(lock: Mapping[str, object]) -> None:
    digest = lock.get("sha256")
    if lock.get("verified") is not True or lock.get("state") != _LOCKED:
        raise PermissionError(
            "official backend requires a CLI-verified LOCAL PROTOCOL LOCK"
        )
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PermissionError("verified lock requires a lowercase SHA-256")
    from .lock_guard import verify_runtime_lock

    verify_runtime_lock(lock)


def _contrast_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    from .official_contrasts import contrast_rows

    return contrast_rows(metric_rows)


def _write_rows(
    output_dir: Path,
    rows: list[dict[str, object]],
    *,
    official: bool,
    lock_sha256: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        "official_rows.json" if official else "smoke_official_rows.json"
    )
    if path.exists():
        raise FileExistsError(f"result rows already exist: {path}")
    payload = {"official": official, "lock_sha256": lock_sha256, "rows": rows}
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
