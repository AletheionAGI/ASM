from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterator

from drm_language_emitter.utils import save_json


def manifest_shards(manifest_path: Path, split: str | None = None) -> list[Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.resolve().parent
    paths: list[Path] = []
    for shard in payload.get("shards", []):
        if split is not None and shard.get("split") != split:
            continue
        path = (root / shard["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"shard path escapes manifest directory: {shard['path']!r}")
        paths.append(path)
    if not paths:
        raise ValueError(f"no matching shards in {manifest_path}")
    return paths


def stream_blocks(paths: list[Path], block_size: int, stride: int) -> Iterator[tuple[int, bytes]]:
    buffer = bytearray()
    absolute_offset = 0
    next_offset = 0
    for path in paths:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                buffer.extend(chunk)
                while next_offset + block_size <= absolute_offset + len(buffer):
                    local = next_offset - absolute_offset
                    yield next_offset, bytes(buffer[local : local + block_size])
                    next_offset += stride
                discard = max(0, next_offset - absolute_offset)
                if discard:
                    del buffer[:discard]
                    absolute_offset += discard


def audit_manifests(
    manifests: dict[str, Path],
    block_size: int = 512,
    stride: int = 256,
    split: str | None = None,
    database: Path | None = None,
    sample_limit: int = 20,
) -> dict[str, object]:
    if block_size <= 0 or stride <= 0 or stride > block_size:
        raise ValueError("require block_size > 0 and 0 < stride <= block_size")
    labels = list(manifests)
    if len(labels) != len(set(labels)) or len(labels) < 2:
        raise ValueError("at least two uniquely named manifests are required")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if database is None:
        temporary = tempfile.TemporaryDirectory(prefix="drm-contamination-")
        database = Path(temporary.name) / "hashes.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)

    counts = {label: 0 for label in labels}
    overlaps = {f"{a}__{b}": 0 for i, a in enumerate(labels) for b in labels[i + 1 :]}
    samples: list[dict[str, object]] = []
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE blocks (digest BLOB, dataset TEXT, offset INTEGER, PRIMARY KEY (digest, dataset))")
        for label, manifest in manifests.items():
            paths = manifest_shards(manifest, split)
            for offset, block in stream_blocks(paths, block_size, stride):
                digest = hashlib.sha256(block).digest()
                counts[label] += 1
                prior = connection.execute(
                    "SELECT dataset, offset FROM blocks WHERE digest = ? AND dataset != ?",
                    (digest, label),
                ).fetchall()
                for other, other_offset in prior:
                    key = "__".join(sorted((label, str(other)), key=labels.index))
                    overlaps[key] += 1
                    if len(samples) < sample_limit:
                        samples.append(
                            {
                                "datasets": [other, label],
                                "offsets": [other_offset, offset],
                                "sha256": digest.hex(),
                            }
                        )
                connection.execute(
                    "INSERT OR IGNORE INTO blocks(digest, dataset, offset) VALUES (?, ?, ?)",
                    (digest, label, offset),
                )
            connection.commit()
    finally:
        connection.close()
        if temporary is not None:
            temporary.cleanup()

    return {
        "passed": not any(overlaps.values()),
        "block_size": block_size,
        "stride": stride,
        "split_filter": split,
        "manifests": {label: str(path) for label, path in manifests.items()},
        "blocks_scanned": counts,
        "overlaps": overlaps,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact block overlap across token-shard manifests.")
    parser.add_argument("--manifest", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--split", default=None, help="Optional shard split filter, e.g. train, val, or test.")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = {name: Path(path) for name, path in args.manifest}
    result = audit_manifests(
        manifests,
        args.block_size,
        args.stride,
        args.split,
        args.database,
        args.sample_limit,
    )
    save_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
