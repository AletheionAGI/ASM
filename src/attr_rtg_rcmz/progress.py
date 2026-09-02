"""Durable terminal progress reporting with atomic machine-readable state."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Self, TextIO


class ProgressReporter:
    """Write human, JSON, and log heartbeats for one sequential run."""

    def __init__(
        self, output_dir: Path, *, interval: float = 10.0, stream: TextIO | None = None
    ) -> None:
        if not 0 < interval <= 10:
            raise ValueError("heartbeat interval must be in (0, 10] seconds")
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.stream = stream
        self.started = time.monotonic()
        self._state: dict[str, Any] = {
            "phase": "starting",
            "seed": None,
            "arm": None,
            "update": 0,
            "total_updates": None,
            "vram_bytes": None,
            "output_paths": [],
        }
        self._lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "run.log"
        self.status_path = self.output_dir / "status.json"

    def __enter__(self) -> Self:
        self.publish()
        self._thread = threading.Thread(
            target=self._heartbeat, name="attr-heartbeat", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc is not None:
            self.update(phase="failed", error=str(exc))
        self.close()

    def update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)
        self.publish()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
        self.publish()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        elapsed = max(0.0, time.monotonic() - self.started)
        total, current = state.get("total_updates"), state.get("update", 0)
        eta = (
            elapsed * (total - current) / current
            if total and current and current < total
            else (0.0 if total and current >= total else None)
        )
        state.update(
            elapsed_seconds=round(elapsed, 3),
            eta_seconds=None if eta is None else round(eta, 3),
            heartbeat_unix=time.time(),
        )
        return state

    def publish(self) -> None:
        with self._publish_lock:
            state = self.snapshot()
            line = self._format(state)
            if self.stream is not None:
                print(line, file=self.stream, flush=True)
            self._append_log(line)
            self._atomic_json(state)

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            self.publish()

    def _append_log(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _atomic_json(self, state: dict[str, Any]) -> None:
        temporary = self.status_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.status_path)

    @staticmethod
    def _format(state: dict[str, Any]) -> str:
        def shown(value: Any) -> str:
            return "-" if value is None else str(value)

        return (
            f"phase={shown(state.get('phase'))} seed={shown(state.get('seed'))} "
            f"arm={shown(state.get('arm'))} update={shown(state.get('update'))}/"
            f"{shown(state.get('total_updates'))} elapsed={state['elapsed_seconds']:.1f}s "
            f"ETA={shown(state.get('eta_seconds'))}s VRAM={shown(state.get('vram_bytes'))} "
            f"outputs={','.join(map(str, state.get('output_paths', []))) or '-'}"
        )
