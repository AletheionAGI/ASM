from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from drm_language_emitter.data import MemmapTokenDataset, split_lm_ids
from scripts.tokenize_corpus_to_uint8 import tokenize_corpus_to_uint8


def test_tokenize_corpus_to_uint8_writes_manifest_and_shards(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    source.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")

    manifest = tokenize_corpus_to_uint8(
        inputs=[source],
        output_dir=tmp_path / "tokens",
        shard_bytes=10,
        val_bytes=6,
    )

    assert manifest["tokenizer_type"] == "byte"
    assert manifest["dtype"] == "uint8"
    assert manifest["train_tokens"] == 20
    assert manifest["val_tokens"] == 6
    assert len([shard for shard in manifest["shards"] if shard["split"] == "train"]) == 2
    assert len([shard for shard in manifest["shards"] if shard["split"] == "val"]) == 1


def test_memmap_token_dataset_returns_shifted_lm_batch(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    source.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    tokenize_corpus_to_uint8(inputs=[source], output_dir=tmp_path / "tokens", shard_bytes=8, val_bytes=2)

    dataset = MemmapTokenDataset(tmp_path / "tokens" / "manifest.json", split="train")
    x, y = dataset.make_batch(
        batch_size=4,
        seq_len=5,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )

    assert x.shape == (4, 5)
    assert y.shape == (4, 5)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_memmap_token_dataset_reads_windows_across_shards(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    source.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    tokenize_corpus_to_uint8(inputs=[source], output_dir=tmp_path / "tokens", shard_bytes=8, val_bytes=2)

    with MemmapTokenDataset(tmp_path / "tokens" / "manifest.json", split="train") as dataset:
        x, y = dataset.window(start=6, seq_len=5)

    assert x.tolist() == list(b"ghijk")
    assert y.tolist() == list(b"hijkl")


def test_split_lm_ids_produces_disjoint_partitions() -> None:
    ids = list(range(100))
    train_ids, val_ids = split_lm_ids(ids, 0.1)
    assert train_ids == list(range(90))
    assert val_ids == list(range(90, 100))
    assert set(train_ids).isdisjoint(val_ids)


def test_memmap_token_dataset_rejects_shard_outside_manifest_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"abc")
    root = tmp_path / "tokens"
    root.mkdir()
    manifest = {
        "dtype": "uint8",
        "shards": [
            {
                "split": "train",
                "path": "../outside.bin",
                "bytes": 3,
                "sha256": "0" * 64,
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes manifest directory"):
        MemmapTokenDataset(path)


def test_memmap_token_dataset_detects_corrupted_shard(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    source.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    manifest = tokenize_corpus_to_uint8(
        inputs=[source],
        output_dir=tmp_path / "tokens",
        shard_bytes=20,
        val_bytes=6,
    )
    train_shard = next(item for item in manifest["shards"] if item["split"] == "train")
    shard = tmp_path / "tokens" / train_shard["path"]
    shard.write_bytes(b"X" + shard.read_bytes()[1:])

    with pytest.raises(ValueError, match="sha256 mismatch"):
        MemmapTokenDataset(tmp_path / "tokens" / "manifest.json", split="train")
