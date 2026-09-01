"""Document-level byte corpus splits for ASM-VR Phase 3A."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class ByteCorpusSplits:
    train: Tensor
    validation: Tensor
    test: Tensor
    manifest: dict[str, object]


def _tensor_from_documents(documents: list[bytes]) -> Tensor:
    payload = bytearray(b"\n\n".join(documents))
    return torch.frombuffer(payload, dtype=torch.uint8)


def load_document_hash_splits(path: str | Path) -> ByteCorpusSplits:
    """Split documents by SHA-256 bucket into deterministic 90/5/5 partitions."""
    source = Path(path)
    raw = source.read_bytes()
    documents = [item for item in raw.split(b"\n\n") if item]
    groups: dict[str, list[bytes]] = {"train": [], "validation": [], "test": []}
    for document in documents:
        bucket = int.from_bytes(hashlib.sha256(document).digest()[:8], "big") % 20
        split = "train" if bucket < 18 else "validation" if bucket == 18 else "test"
        groups[split].append(document)
    tensors = {name: _tensor_from_documents(items) for name, items in groups.items()}
    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "split_method": "sha256(document) first 64 bits modulo 20; 18/1/1",
        "documents": {name: len(items) for name, items in groups.items()},
        "bytes": {name: int(value.numel()) for name, value in tensors.items()},
        "sha256": {
            name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for name, value in tensors.items()
        },
    }
    if any(value.numel() < 2 for value in tensors.values()):
        raise ValueError("every corpus split must contain at least two bytes")
    return ByteCorpusSplits(
        tensors["train"], tensors["validation"], tensors["test"], manifest
    )


def sample_byte_windows(
    tokens: Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    seed: int,
    step: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Sample a deterministic batch of shifted byte windows."""
    if tokens.ndim != 1 or tokens.dtype is not torch.uint8:
        raise TypeError("tokens must be a one-dimensional uint8 tensor")
    if batch_size < 1 or sequence_length < 1 or step < 0:
        raise ValueError("batch_size/sequence_length must be positive and step non-negative")
    maximum = tokens.numel() - sequence_length - 1
    if maximum < 1:
        raise ValueError("token split is too short for the requested window")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 1_000_003 + int(step))
    offsets = torch.randint(0, maximum, (batch_size,), generator=generator)
    windows = torch.stack(
        [tokens[int(offset) : int(offset) + sequence_length + 1] for offset in offsets]
    ).long()
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


__all__ = ["ByteCorpusSplits", "load_document_hash_splits", "sample_byte_windows"]
