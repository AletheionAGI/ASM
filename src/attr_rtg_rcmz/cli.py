"""Command-line fast path for safe synthetic ATTR-RTG-RCMZ runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .progress import ProgressReporter
from .rendering import render_summary

SEEDS = (29, 43, 71, 89, 107)
ARMS = ("R", "CM", "Z", "T")


class LockVerificationError(ValueError):
    """The supplied local protocol lock is absent or unverifiable."""


def verify_official_lock(path: Path, expected_sha256: str) -> str:
    """Verify only the canonical receipt against the generated source anchor."""
    from .lock_guard import LockGuardError, verify_canonical_lock

    try:
        verified = verify_canonical_lock(path, expected_sha256)
    except LockGuardError as error:
        raise LockVerificationError(str(error)) from error
    return str(verified["sha256"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATTR-RTG-RCMZ terminal runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="render deterministic synthetic data only",
    )
    mode.add_argument(
        "--official", action="store_true", help="request locked official execution"
    )
    mode.add_argument("--smoke-official", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--lock-sha256")
    parser.add_argument(
        "--recover-completed",
        action="store_true",
        help="reuse complete terminal seed groups declared by --recovery-manifest",
    )
    parser.add_argument(
        "--recovery-manifest",
        type=Path,
        help="trusted manifest containing checkpoint paths and expected SHA-256 values",
    )
    return parser


def synthetic_rows() -> list[dict[str, object]]:
    """Return labeled fixture values; these are not benchmark results."""
    rows = []
    for seed_index, seed in enumerate(SEEDS):
        for arm_index, arm in enumerate(ARMS):
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "regime": "synthetic",
                    "h8_nll": round(0.40 + arm_index * 0.025 + seed_index * 0.003, 6),
                    "status": "SYNTHETIC",
                }
            )
    return rows


def run_dry(
    output_dir: Path, heartbeat_seconds: float, stream: object = None
) -> list[Path]:
    rows = synthetic_rows()
    target_stream = sys.stdout if stream is None else stream
    with ProgressReporter(
        output_dir, interval=heartbeat_seconds, stream=target_stream
    ) as progress:
        progress.update(phase="synthetic", total_updates=2000, vram_bytes=0)
        for index, row in enumerate(rows, 1):
            progress.update(seed=row["seed"], arm=row["arm"], update=index * 100)
        paths = render_summary(rows, output_dir)
        relative = [str(path.relative_to(output_dir)) for path in paths]
        progress.update(
            phase="completed", seed=None, arm=None, update=2000, output_paths=relative
        )
    return paths


def run_engine(
    output_dir: Path,
    heartbeat_seconds: float,
    *,
    lock: dict[str, object] | None,
    smoke: bool = False,
    recovery_manifest: Path | None = None,
    stream: object = None,
) -> list[Path]:
    """Run the official engine behind progress and render its scalar rows."""
    from .official import run_official, run_smoke_official

    target_stream = sys.stdout if stream is None else stream
    with ProgressReporter(
        output_dir, interval=heartbeat_seconds, stream=target_stream
    ) as progress:
        progress.update(
            phase="official-smoke" if smoke else "official", total_updates=2000
        )
        callback = lambda event: progress.update(**event)
        rows = (
            run_smoke_official(output_dir, callback)
            if smoke
            else run_official(
                output_dir,
                callback,
                lock=lock or {},
                recovery_manifest=recovery_manifest,
            )
        )
        paths = render_summary(rows, output_dir, synthetic=smoke)
        for name in ("official_rows.json", "smoke_official_rows.json"):
            engine_rows = output_dir / name
            if engine_rows.exists():
                paths.append(engine_rows)
        relative = [str(path.relative_to(output_dir)) for path in paths]
        final_update = progress.snapshot().get(
            "total_updates"
        ) or progress.snapshot().get("update")
        progress.update(
            phase="completed",
            seed=None,
            arm=None,
            update=final_update,
            output_paths=relative,
        )
    return paths


def write_tombstone(output_dir: Path, error: BaseException) -> Path:
    """Atomically record a failed engine run without selective results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "TOMBSTONE.json"
    temporary = output_dir / ".TOMBSTONE.json.tmp"
    temporary.write_text(
        json.dumps({"status": "INVALID", "error": str(error)}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.recover_completed) != bool(args.recovery_manifest):
        print(
            "recovery requires both --recover-completed and --recovery-manifest",
            file=sys.stderr,
        )
        return 2
    if args.recover_completed and not args.official:
        print("checkpoint recovery is available only in official mode", file=sys.stderr)
        return 2
    if args.official:
        if args.lock_file is None or args.lock_sha256 is None:
            print(
                "official mode requires --lock-file and --lock-sha256", file=sys.stderr
            )
            return 2
        try:
            digest = verify_official_lock(args.lock_file, args.lock_sha256)
        except (OSError, UnicodeError, LockVerificationError) as error:
            print(f"official mode blocked: {error}", file=sys.stderr)
            return 2
        lock = {"verified": True, "state": "LOCAL PROTOCOL LOCK", "sha256": digest}
        try:
            if args.recovery_manifest is not None:
                from .recovery import archive_previous_run

                if (
                    args.recovery_manifest.resolve()
                    == (args.output_dir / "recovery_manifest.json").resolve()
                ):
                    raise ValueError(
                        "recovery input cannot be the generated recovery_manifest.json"
                    )
                archive_previous_run(args.output_dir)
            engine_options: dict[str, object] = {"lock": lock}
            if args.recovery_manifest is not None:
                engine_options["recovery_manifest"] = args.recovery_manifest
            run_engine(args.output_dir, args.heartbeat_seconds, **engine_options)
        except (
            ImportError,
            MemoryError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            write_tombstone(args.output_dir, error)
            print(f"official run failed: {error}", file=sys.stderr)
            return 1
        return 0
    try:
        if args.smoke_official:
            run_engine(args.output_dir, args.heartbeat_seconds, lock=None, smoke=True)
        else:
            run_dry(args.output_dir, args.heartbeat_seconds)
    except (
        ImportError,
        MemoryError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        write_tombstone(args.output_dir, error)
        print(f"run failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
