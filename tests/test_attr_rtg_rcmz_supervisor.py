from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from attr_rtg_rcmz.official_supervisor import supervise


def test_supervisor_streams_and_completes(tmp_path: Path):
    stream = io.StringIO()
    receipt = tmp_path / "receipt.json"
    code = supervise(
        [sys.executable, "-c", "print('live-line', flush=True)"],
        receipt,
        timeout_seconds=2,
        stream=stream,
    )
    assert code == 0
    assert "live-line" in stream.getvalue()
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "COMPLETED"
    assert (
        payload["start_monotonic"]
        <= payload["finish_monotonic"]
        <= payload["deadline_monotonic"]
    )
    assert (
        payload["elapsed_seconds"]
        == payload["finish_monotonic"] - payload["start_monotonic"]
    )
    assert payload["returncode"] == 0


def test_supervisor_terminates_and_tombstones_timeout(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    code = supervise(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        receipt,
        timeout_seconds=0.05,
        kill_grace_seconds=0.05,
        stream=io.StringIO(),
    )
    assert code == 124
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "TIMEOUT"
    assert payload["finish_monotonic"] >= payload["deadline_monotonic"]
    assert payload["returncode"] is not None
