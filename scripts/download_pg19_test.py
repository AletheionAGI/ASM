from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from drm_language_emitter.utils import save_json


BUCKET = "deepmind-gutenberg"
API_ROOT = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}"
DOWNLOAD_ROOT = f"https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o"


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def list_test_objects() -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"prefix": "test/", "maxResults": 1000})
    payload = fetch_json(f"{API_ROOT}/o?{query}")
    objects = [item for item in payload.get("items", []) if item.get("name", "").endswith(".txt")]
    if len(objects) != 100:
        raise ValueError(f"expected 100 PG-19 test books, found {len(objects)}")
    return sorted(objects, key=lambda item: item["name"])


def object_matches(item: dict[str, Any], path: Path) -> bool:
    if not path.exists() or path.stat().st_size != int(item["size"]):
        return False
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest() == base64.b64decode(item["md5Hash"])


def download_object(item: dict[str, Any], destination: Path, timeout: int = 30, retries: int = 3) -> None:
    name = str(item["name"])
    url = f"{DOWNLOAD_ROOT}/{urllib.parse.quote(name, safe='')}?alt=media"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        digest = hashlib.md5(usedforsecurity=False)
        size = 0
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != int(item["size"]) or digest.digest() != base64.b64decode(item["md5Hash"]):
                raise ValueError(f"integrity check failed for {name}")
            partial.replace(destination)
            return
        except (OSError, TimeoutError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(attempt)


def metadata_by_id(metadata_path: Path) -> dict[str, dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        book_id = row.get("book_id", "").strip()
        if book_id:
            result[book_id] = row
    return result


def download_pg19_test(output_root: str | Path, overwrite: bool = False) -> dict[str, Any]:
    output_root = Path(output_root)
    books_root = output_root / "books"
    objects = list_test_objects()

    metadata_path = output_root / "metadata.csv"
    license_path = output_root / "LICENSE"
    for object_name, destination in (("metadata.csv", metadata_path), ("LICENSE", license_path)):
        item = fetch_json(f"{API_ROOT}/o/{urllib.parse.quote(object_name, safe='')}")
        if overwrite or not object_matches(item, destination):
            download_object(item, destination)

    metadata = metadata_by_id(metadata_path)
    records: list[dict[str, Any]] = []
    downloaded = 0
    for item in objects:
        book_id = Path(item["name"]).stem
        destination = books_root / f"{book_id}.txt"
        if overwrite or not object_matches(item, destination):
            download_object(item, destination)
            downloaded += 1
        print(f"[{len(records) + 1:03d}/{len(objects):03d}] verified {destination.name}", flush=True)
        text = destination.read_text(encoding="utf-8")
        row = metadata.get(book_id, {})
        records.append(
            {
                "book_id": book_id,
                "book_title": row.get("short_book_title", ""),
                "publication_date": row.get("publication_date", ""),
                "book_link": row.get("book_link", ""),
                "text": text,
                "source_object": item["name"],
                "source_size": int(item["size"]),
                "source_md5_base64": item["md5Hash"],
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )

    jsonl_path = output_root / "pg19_test.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "dataset": "PG-19",
        "provider": "Google DeepMind",
        "source_bucket": f"gs://{BUCKET}/test/",
        "source_repository": "https://github.com/google-deepmind/pg19",
        "license_file": str(license_path),
        "split": "test",
        "books": len(records),
        "bytes": sum(record["source_size"] for record in records),
        "downloaded_this_run": downloaded,
        "jsonl": str(jsonl_path),
        "jsonl_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
    }
    save_json(output_root / "download_provenance.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the official PG-19 test split.")
    parser.add_argument("--output-root", type=Path, default=Path("data/external/pg19"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download_pg19_test(args.output_root, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
