from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from drm_language_emitter.utils import save_json


SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_inputs(inputs: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            paths.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        else:
            paths.append(path)
    if not paths:
        raise ValueError("input set contains no files")
    return paths


def normalize_document(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split()).strip()


def iter_documents(path: Path, text_field: str = "text") -> Iterator[str]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                text = payload.get(text_field) if isinstance(payload, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"{path}:{line_number} has no string field {text_field!r}")
                yield text
        return
    text = path.read_text(encoding="utf-8")
    pieces = text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
    yield from (piece for piece in pieces if piece.strip())


def write_split_manifest(
    documents: list[str],
    output_dir: Path,
    split: str,
    shard_bytes: int,
) -> dict[str, Any]:
    if not documents:
        raise ValueError(f"split {split!r} has no documents after filtering and deduplication")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "documents.txt"
    corpus_path.write_text("\n\n".join(documents) + "\n", encoding="utf-8", newline="\n")
    raw = corpus_path.read_bytes()
    shards: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(raw), shard_bytes)):
        chunk = raw[start : start + shard_bytes]
        shard_path = output_dir / f"{split}_{index:06d}.bin"
        shard_path.write_bytes(chunk)
        shards.append(
            {
                "split": split,
                "path": shard_path.name,
                "bytes": len(chunk),
                "sha256": hashlib.sha256(chunk).hexdigest(),
            }
        )
    manifest = {
        "format": "drm-language-emitter-token-shards",
        "version": 1,
        "tokenizer_type": "byte",
        "dtype": "uint8",
        "split": split,
        "total_tokens": len(raw),
        "document_count": len(documents),
        "shard_bytes": shard_bytes,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "sources": [
            {
                "path": corpus_path.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
        "shards": shards,
    }
    save_json(output_dir / "manifest.json", manifest)
    return manifest


def prepare_independent_benchmark(
    inputs: dict[str, list[str | Path]],
    output_root: str | Path,
    shard_bytes: int = 100_000_000,
    min_doc_chars: int = 200,
    text_field: str = "text",
    metadata: dict[str, str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if shard_bytes <= 0 or min_doc_chars < 1:
        raise ValueError("shard_bytes and min_doc_chars must be positive")
    requested_splits = [split for split in SPLITS if split in inputs]
    if not requested_splits or set(inputs) - set(SPLITS):
        raise ValueError(f"inputs must use one or more of: {', '.join(SPLITS)}")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_root} is not empty; pass overwrite=True")

    files = {split: expand_inputs(inputs[split]) for split in requested_splits}
    accepted: dict[str, list[str]] = {split: [] for split in requested_splits}
    seen: dict[str, tuple[str, str]] = {}
    duplicate_counts = {split: 0 for split in requested_splits}
    rejected_short = {split: 0 for split in requested_splits}
    document_records: list[dict[str, object]] = []

    for split in requested_splits:
        for path in files[split]:
            for document_index, raw_text in enumerate(iter_documents(path, text_field)):
                text = normalize_document(raw_text)
                if len(text) < min_doc_chars:
                    rejected_short[split] += 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                prior = seen.get(digest)
                if prior is not None:
                    duplicate_counts[split] += 1
                    document_records.append(
                        {
                            "sha256": digest,
                            "requested_split": split,
                            "status": "duplicate",
                            "duplicate_of_split": prior[0],
                            "source": str(path),
                            "source_document_index": document_index,
                        }
                    )
                    continue
                seen[digest] = (split, str(path))
                accepted[split].append(text)
                document_records.append(
                    {
                        "sha256": digest,
                        "requested_split": split,
                        "status": "accepted",
                        "chars": len(text),
                        "source": str(path),
                        "source_document_index": document_index,
                    }
                )

    if overwrite:
        output_root.mkdir(parents=True, exist_ok=True)
        for split in requested_splits:
            split_root = output_root / split
            if split_root.exists():
                for path in split_root.glob("*"):
                    if path.is_file():
                        path.unlink()

    manifests = {
        split: write_split_manifest(accepted[split], output_root / split, split, shard_bytes)
        for split in requested_splits
    }
    provenance = {
        "format": "drm-independent-benchmark-provenance",
        "version": 1,
        "split_precedence": requested_splits,
        "normalization": "NUL to space; Unicode preserved; whitespace collapsed",
        "deduplication": "exact SHA-256 of normalized UTF-8 document",
        "min_doc_chars": min_doc_chars,
        "metadata": metadata or {},
        "inputs": {
            split: [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in files[split]
            ]
            for split in requested_splits
        },
        "accepted_documents": {split: len(accepted[split]) for split in requested_splits},
        "duplicates_removed": duplicate_counts,
        "short_documents_rejected": rejected_short,
        "manifests": {
            split: {
                "path": str(output_root / split / "manifest.json"),
                "corpus_sha256": manifests[split]["corpus_sha256"],
                "tokens": manifests[split]["total_tokens"],
            }
            for split in requested_splits
        },
    }
    save_json(output_root / "provenance.json", provenance)
    records_path = output_root / "document_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in document_records),
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare document-disjoint train/validation/test byte shards.")
    parser.add_argument("--train-input", nargs="+")
    parser.add_argument("--validation-input", nargs="+")
    parser.add_argument("--test-input", nargs="+")
    parser.add_argument("--output-root", type=Path, default=Path("data/benchmark_125m"))
    parser.add_argument("--shard-bytes", type=int, default=100_000_000)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--source-name", default="")
    parser.add_argument("--source-version", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    inputs = {
        split: values
        for split, values in (
            ("train", args.train_input),
            ("validation", args.validation_input),
            ("test", args.test_input),
        )
        if values
    }
    provenance = prepare_independent_benchmark(
        inputs,
        args.output_root,
        args.shard_bytes,
        args.min_doc_chars,
        args.text_field,
        {
            "source_name": args.source_name,
            "source_version": args.source_version,
            "source_url": args.source_url,
            "license": args.license,
        },
        args.overwrite,
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
