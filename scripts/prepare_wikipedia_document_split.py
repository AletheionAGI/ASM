from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from drm_language_emitter.utils import save_json
try:
    from scripts.prepare_independent_benchmark import normalize_document, sha256_file
except ModuleNotFoundError:
    from prepare_independent_benchmark import normalize_document, sha256_file


class ShardWriter:
    def __init__(self, root: Path, split: str, shard_bytes: int) -> None:
        self.root = root
        self.split = split
        self.shard_bytes = shard_bytes
        self.index = 0
        self.handle = None
        self.path: Path | None = None
        self.size = 0
        self.total = 0
        self.corpus_digest = hashlib.sha256()
        self.shard_digest = hashlib.sha256()
        self.shards: list[dict[str, Any]] = []

    def _open(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{self.split}_{self.index:06d}.bin"
        self.handle = self.path.open("wb")

    def _close(self) -> None:
        if self.handle is None or self.path is None:
            return
        self.handle.close()
        if self.size:
            self.shards.append(
                {
                    "split": self.split,
                    "path": self.path.name,
                    "bytes": self.size,
                    "sha256": self.shard_digest.hexdigest(),
                }
            )
        else:
            self.path.unlink(missing_ok=True)
        self.handle = None

    def write(self, data: bytes) -> None:
        self.corpus_digest.update(data)
        self.total += len(data)
        cursor = 0
        while cursor < len(data):
            if self.handle is None:
                self._open()
            room = self.shard_bytes - self.size
            chunk = data[cursor : cursor + room]
            self.handle.write(chunk)
            self.shard_digest.update(chunk)
            self.size += len(chunk)
            cursor += len(chunk)
            if self.size == self.shard_bytes:
                self._close()
                self.index += 1
                self.size = 0
                self.shard_digest = hashlib.sha256()

    def finish(self, document_count: int, source: Path) -> dict[str, Any]:
        self._close()
        manifest = {
            "format": "drm-language-emitter-token-shards",
            "version": 1,
            "tokenizer_type": "byte",
            "dtype": "uint8",
            "split": self.split,
            "total_tokens": self.total,
            "document_count": document_count,
            "shard_bytes": self.shard_bytes,
            "corpus_sha256": self.corpus_digest.hexdigest(),
            "source": str(source),
            "shards": self.shards,
        }
        save_json(self.root / "manifest.json", manifest)
        return manifest


def iter_blank_line_documents(path: Path) -> Iterator[str]:
    pieces: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pieces.append(line)
            elif pieces:
                yield "".join(pieces)
                pieces.clear()
    if pieces:
        yield "".join(pieces)


def prepare_wikipedia_document_split(
    source: str | Path,
    output_root: str | Path,
    validation_fraction: float = 0.001,
    shard_bytes: int = 100_000_000,
    min_doc_chars: int = 200,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    source = Path(source)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_root} is not empty; pass overwrite=True")
    if overwrite:
        for split in ("train", "validation"):
            root = output_root / split
            if root.exists():
                for path in root.glob("*"):
                    if path.is_file():
                        path.unlink()

    writers = {
        split: ShardWriter(output_root / split, split, shard_bytes)
        for split in ("train", "validation")
    }
    counts = {"train": 0, "validation": 0, "short_rejected": 0, "duplicates_removed": 0}
    seen: set[str] = set()
    threshold = int(validation_fraction * (1 << 64))
    for raw_document in iter_blank_line_documents(source):
        document = normalize_document(raw_document)
        if len(document) < min_doc_chars:
            counts["short_rejected"] += 1
            continue
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
        if digest in seen:
            counts["duplicates_removed"] += 1
            continue
        seen.add(digest)
        split = "validation" if int(digest[:16], 16) < threshold else "train"
        writers[split].write(document.encode("utf-8") + b"\n\n")
        counts[split] += 1

    manifests = {
        split: writer.finish(counts[split], source)
        for split, writer in writers.items()
    }
    if not counts["train"] or not counts["validation"]:
        raise ValueError("deterministic partition produced an empty split")
    provenance = {
        "format": "drm-wikipedia-document-split",
        "version": 1,
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "assignment": "first 64 bits of SHA-256(normalized document) compared with fraction threshold",
        "validation_fraction": validation_fraction,
        "min_doc_chars": min_doc_chars,
        "counts": counts,
        "manifests": {
            split: {
                "path": str(output_root / split / "manifest.json"),
                "tokens": manifest["total_tokens"],
                "corpus_sha256": manifest["corpus_sha256"],
            }
            for split, manifest in manifests.items()
        },
    }
    save_json(output_root / "provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic document-level Wikipedia train/validation shards.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/benchmark_125m_wikipedia"))
    parser.add_argument("--validation-fraction", type=float, default=0.001)
    parser.add_argument("--shard-bytes", type=int, default=100_000_000)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = prepare_wikipedia_document_split(
        args.source,
        args.output_root,
        args.validation_fraction,
        args.shard_bytes,
        args.min_doc_chars,
        args.overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
