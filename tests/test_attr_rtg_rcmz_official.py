from __future__ import annotations

import json
from pathlib import Path

import pytest

from attr_rtg_rcmz.official import run_official, run_smoke_official


def test_official_fails_before_world_generation_without_verified_lock(tmp_path: Path):
    events = []
    with pytest.raises(PermissionError, match="CLI-verified"):
        run_official(tmp_path, events.append, {"verified": False})
    assert events == []
    assert list(tmp_path.iterdir()) == []


def test_smoke_official_runs_miniature_synthetic_pipeline(tmp_path: Path):
    events = []
    rows = run_smoke_official(tmp_path, events.append)
    assert len(rows) == 3
    assert {row["regime"] for row in rows} == {"ID", "shift", "OOD"}
    assert all(row["status"] == "SYNTHETIC" for row in rows)
    payload = json.loads((tmp_path / "smoke_official_rows.json").read_text())
    assert payload["official"] is False
    assert events[-1]["phase"] == "synthetic-smoke-completed"
