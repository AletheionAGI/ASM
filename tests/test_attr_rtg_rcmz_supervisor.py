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


def test_official_command_forwards_recovery_manifest(tmp_path: Path):
    from attr_rtg_rcmz.official_supervisor import official_command

    recovery = tmp_path / "official" / "recovery_input_v1.json"
    command = official_command(
        tmp_path / "official", tmp_path / "lock.json", "a" * 64, recovery
    )
    assert command[-3:] == ["--recover-completed", "--recovery-manifest", str(recovery)]


def test_recovery_archives_receipts_but_preserves_checkpoints(tmp_path: Path):
    from attr_rtg_rcmz.recovery import archive_previous_run

    output = tmp_path / "official"
    checkpoint = output / "checkpoints" / "seed-29_R_update-2000.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"kept")
    (output / "TOMBSTONE.json").write_text("old")
    (output / "status.json").write_text("old-status")
    receipt = output / "supervisor.json"
    receipt.write_text("old-supervisor")
    archived = archive_previous_run(output, (receipt,))
    assert archived is not None
    assert checkpoint.read_bytes() == b"kept"
    assert (archived / "TOMBSTONE.json").read_text() == "old"
    assert (archived / "status.json").read_text() == "old-status"
    assert (archived / "supervisor.json").read_text() == "old-supervisor"
