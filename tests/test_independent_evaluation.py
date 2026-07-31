from pathlib import Path

import pytest
import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.data import MemmapTokenDataset
from drm_language_emitter.model import DRMEmitterModel
from scripts.audit_dataset_contamination import audit_manifests
from scripts.evaluate_frozen_test import evaluate_sequential
from scripts.prepare_independent_benchmark import prepare_independent_benchmark
from scripts.prepare_wikipedia_document_split import prepare_wikipedia_document_split
from scripts.tokenize_corpus_to_uint8 import tokenize_corpus_to_uint8
from scripts.train_gpt2_memmap import next_token_ce


def make_manifest(tmp_path: Path, name: str, content: str) -> Path:
    source = tmp_path / f"{name}.txt"
    source.write_text(content, encoding="utf-8")
    root = tmp_path / name
    tokenize_corpus_to_uint8([source], root, shard_bytes=32, val_bytes=8)
    return root / "manifest.json"


def test_contamination_audit_accepts_disjoint_corpora(tmp_path: Path) -> None:
    train = make_manifest(tmp_path, "train", "A" * 96)
    test = make_manifest(tmp_path, "test", "B" * 96)
    result = audit_manifests({"train": train, "test": test}, block_size=16, stride=8)
    assert result["passed"] is True
    assert result["overlaps"]["train__test"] == 0


def test_contamination_audit_detects_shared_blocks(tmp_path: Path) -> None:
    shared = "unique-shared-block-" * 8
    train = make_manifest(tmp_path, "train", "prefix-a" + shared)
    test = make_manifest(tmp_path, "test", "prefix-b" + shared)
    result = audit_manifests({"train": train, "test": test}, block_size=16, stride=8)
    assert result["passed"] is False
    assert result["overlaps"]["train__test"] > 0


def test_frozen_evaluation_is_deterministic_and_respects_token_limit(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, "dataset", "deterministic evaluation corpus " * 8)
    model = DRMEmitterModel(
        DRMConfig(
            vocab_size=256,
            d_token=8,
            d_state=12,
            n_directions=4,
            metric_rank=2,
            hidden_size=16,
            max_seq_len=8,
        )
    )
    with MemmapTokenDataset(manifest, split="train") as dataset:
        first = evaluate_sequential(model, dataset, "drm", 8, 21, 2, torch.device("cpu"))
        second = evaluate_sequential(model, dataset, "drm", 8, 21, 2, torch.device("cpu"))
    assert first == pytest.approx(second)
    assert first[1] == 21
    assert first[0] > 0


def test_gpt2_training_loss_uses_explicit_next_token_targets() -> None:
    class ExplicitLogitModel(torch.nn.Module):
        def forward(self, *, input_ids: torch.Tensor):
            logits = torch.full((*input_ids.shape, 4), -20.0)
            targets = (input_ids + 1) % 4
            logits.scatter_(-1, targets.unsqueeze(-1), 20.0)
            return type("Output", (), {"logits": logits})()

    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, 0]])
    assert float(next_token_ce(ExplicitLogitModel(), x, y)) < 1e-6


def test_prepare_independent_benchmark_deduplicates_before_tokenization(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.txt"
    train.write_text("aaaaaaaaaaaaaaaaaaaa\n\nzzzzzzzzzzzzzzzzzzzz", encoding="utf-8")
    validation.write_text(
        '{"text": "bbbbbbbbbbbbbbbbbbbb"}\n{"text": "zzzzzzzzzzzzzzzzzzzz"}\n',
        encoding="utf-8",
    )
    test.write_text("cccccccccccccccccccc", encoding="utf-8")

    root = tmp_path / "prepared"
    provenance = prepare_independent_benchmark(
        {"train": [train], "validation": [validation], "test": [test]},
        root,
        shard_bytes=16,
        min_doc_chars=5,
    )

    assert provenance["accepted_documents"] == {"train": 2, "validation": 1, "test": 1}
    assert provenance["duplicates_removed"]["validation"] == 1
    for split in ("train", "validation", "test"):
        with MemmapTokenDataset(root / split / "manifest.json", split=split) as dataset:
            assert len(dataset) > 0
    contamination = audit_manifests(
        {split: root / split / "manifest.json" for split in ("train", "validation", "test")},
        block_size=8,
        stride=4,
    )
    assert contamination["passed"] is True


def test_wikipedia_document_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "wikipedia.txt"
    source.write_text(
        "\n\n".join(f"document-{index}-" + chr(65 + index) * 40 for index in range(20)),
        encoding="utf-8",
    )
    first = prepare_wikipedia_document_split(
        source,
        tmp_path / "first",
        validation_fraction=0.5,
        shard_bytes=64,
        min_doc_chars=10,
    )
    second = prepare_wikipedia_document_split(
        source,
        tmp_path / "second",
        validation_fraction=0.5,
        shard_bytes=64,
        min_doc_chars=10,
    )
    assert first["counts"] == second["counts"]
    for split in ("train", "validation"):
        assert first["manifests"][split]["corpus_sha256"] == second["manifests"][split]["corpus_sha256"]
    audit = audit_manifests(
        {
            "train": tmp_path / "first" / "train" / "manifest.json",
            "validation": tmp_path / "first" / "validation" / "manifest.json",
        },
        block_size=16,
        stride=8,
    )
    assert audit["passed"] is True
