"""Prearmed monotonic supervisor for the locked official CLI process."""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO

TIMEOUT_SECONDS = 20 * 60 * 60
KILL_GRACE_SECONDS = 10.0


def supervise(
    command: list[str],
    receipt_path: Path,
    *,
    timeout_seconds: float = TIMEOUT_SECONDS,
    kill_grace_seconds: float = KILL_GRACE_SECONDS,
    stream: TextIO | None = None,
) -> int:
    """Prearm, stream child output, and write COMPLETED/CRASH/TIMEOUT only."""
    if timeout_seconds <= 0 or kill_grace_seconds < 0:
        raise ValueError("timeouts must be positive")
    target = stream or sys.stdout
    started = time.monotonic()
    deadline = started + timeout_seconds
    _write_status(
        receipt_path,
        {
            "status": "ARMED",
            "clock": "monotonic",
            "timeout_seconds": timeout_seconds,
            "start_monotonic": started,
            "deadline_monotonic": deadline,
        },
        exclusive=True,
    )
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(
        target=_read_lines, args=(child.stdout, lines), daemon=True
    )
    reader.start()
    timed_out = False
    while child.poll() is None:
        _drain(lines, target)
        if time.monotonic() - started >= timeout_seconds:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=kill_grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            break
        time.sleep(0.02)
    reader.join(timeout=1)
    _drain(lines, target)
    if timed_out:
        status = "TIMEOUT"
    elif child.returncode == 0:
        status = "COMPLETED"
    else:
        status = "CRASH"
    finished = time.monotonic()
    _write_status(
        receipt_path,
        {
            "status": status,
            "returncode": child.returncode,
            "start_monotonic": started,
            "deadline_monotonic": deadline,
            "finish_monotonic": finished,
            "elapsed_seconds": finished - started,
        },
        exclusive=False,
    )
    return int(child.returncode or 0) if not timed_out else 124


def official_command(
    output_dir: Path,
    lock_file: Path,
    lock_sha256: str,
    recovery_manifest: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "attr_rtg_rcmz.cli",
        "--official",
        "--output-dir",
        str(output_dir),
        "--lock-file",
        str(lock_file),
        "--lock-sha256",
        lock_sha256,
    ]
    if recovery_manifest is not None:
        command.extend(
            ["--recover-completed", "--recovery-manifest", str(recovery_manifest)]
        )
    return command


def _read_lines(pipe: TextIO | None, lines: queue.Queue[str | None]) -> None:
    if pipe is not None:
        for line in pipe:
            lines.put(line)
    lines.put(None)


def _drain(lines: queue.Queue[str | None], stream: TextIO) -> None:
    while True:
        try:
            line = lines.get_nowait()
        except queue.Empty:
            return
        if line is not None:
            stream.write(line)
            stream.flush()


def _write_status(path: Path, payload: dict[str, object], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"supervisor receipt already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    mode = "x" if exclusive else "w"
    with temporary.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="20-hour official ATTR supervisor")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path)
    args = parser.parse_args()
    if args.recovery_manifest is not None:
        from .recovery import archive_previous_run

        if (
            args.recovery_manifest.resolve()
            == (args.output_dir / "recovery_manifest.json").resolve()
        ):
            parser.error(
                "recovery input cannot be the generated recovery_manifest.json"
            )
        archive_previous_run(args.output_dir, (args.receipt,))
    command = official_command(
        args.output_dir, args.lock_file, args.lock_sha256, args.recovery_manifest
    )
    return supervise(command, args.receipt)


if __name__ == "__main__":
    raise SystemExit(main())
