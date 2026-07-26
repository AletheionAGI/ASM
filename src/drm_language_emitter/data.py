from __future__ import annotations

import bisect
import json
import mmap
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any

import torch

from .tokenizer import ByteTokenizer, CharTokenizer, make_tokenizer


DEFAULT_TINY_TEXT = (
    "Directional relational manifolds guide language as trajectories. "
    "The emitter learns low action motion through active directions. "
    "This tiny corpus is only a smoke test for geometry and generation.\n"
)


def ensure_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_TINY_TEXT * 8, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def make_lm_batch(ids: list[int], batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(ids) < seq_len + 2:
        ids = ids * ((seq_len + 2) // max(len(ids), 1) + 1)
    starts = torch.randint(0, len(ids) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(ids[s : s + seq_len], dtype=torch.long) for s in starts]).to(device)
    y = torch.stack([torch.tensor(ids[s + 1 : s + seq_len + 1], dtype=torch.long) for s in starts]).to(device)
    return x, y


def build_tokenizer(text: str, tokenizer_type: str = "byte") -> ByteTokenizer | CharTokenizer:
    return make_tokenizer(text, tokenizer_type)


class MemmapTokenDataset(AbstractContextManager["MemmapTokenDataset"]):
    """Read fixed uint8 token shards without loading the corpus into RAM."""

    def __init__(self, manifest_path: str | Path, split: str = "train") -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("dtype") != "uint8":
            raise ValueError(f"unsupported token dtype: {self.manifest.get('dtype')!r}")
        self.shards = [shard for shard in self.manifest.get("shards", []) if shard.get("split") == split]
        if not self.shards:
            raise ValueError(f"manifest has no shards for split={split!r}")
        self._paths = [self.root / shard["path"] for shard in self.shards]
        self._lengths = [int(shard["bytes"]) for shard in self.shards]
        self._handles = [path.open("rb") for path in self._paths]
        self._maps = [mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) for handle in self._handles]
        self._ends: list[int] = []
        total = 0
        for length in self._lengths:
            total += length
            self._ends.append(total)
        self.total_tokens = total

    def __len__(self) -> int:
        return self.total_tokens

    def close(self) -> None:
        for mapped in self._maps:
            mapped.close()
        for handle in self._handles:
            handle.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _read_range(self, start: int, length: int) -> bytes:
        if start < 0 or length < 0 or start + length > self.total_tokens:
            raise IndexError("token range is outside dataset bounds")
        remaining = length
        offset = start
        chunks: list[bytes] = []
        while remaining > 0:
            shard_index = bisect.bisect_right(self._ends, offset)
            shard_start = 0 if shard_index == 0 else self._ends[shard_index - 1]
            within_shard = offset - shard_start
            take = min(remaining, self._lengths[shard_index] - within_shard)
            chunks.append(self._maps[shard_index][within_shard : within_shard + take])
            offset += take
            remaining -= take
        return b"".join(chunks)

    def window(self, start: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self._read_range(start, seq_len + 1)
        values = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.long)
        return values[:-1], values[1:]

    def make_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        rank: int = 0,
        world_size: int = 1,
        pin_memory: bool = False,
        non_blocking: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_cpu, y_cpu = self.make_batch_cpu(batch_size, seq_len, generator, rank, world_size, pin_memory and device.type == "cuda")
        return self.move_batch_to_device(x_cpu, y_cpu, device, non_blocking=non_blocking)

    def make_batch_cpu(
        self,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator | None = None,
        rank: int = 0,
        world_size: int = 1,
        pin_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.total_tokens < seq_len + 2:
            raise ValueError("token dataset is too small for requested seq_len")
        max_start = self.total_tokens - seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,), generator=generator)
        if world_size > 1:
            starts = (starts * max(world_size, 1) + rank) % max_start
        x_cpu = torch.empty((batch_size, seq_len), dtype=torch.long)
        y_cpu = torch.empty((batch_size, seq_len), dtype=torch.long)
        for row_index, start in enumerate(starts.tolist()):
            x_row, y_row = self.window(int(start), seq_len)
            x_cpu[row_index].copy_(x_row)
            y_cpu[row_index].copy_(y_row)
        if pin_memory:
            x_cpu = x_cpu.pin_memory()
            y_cpu = y_cpu.pin_memory()
        return x_cpu, y_cpu

    @staticmethod
    def move_batch_to_device(
        x_cpu: torch.Tensor,
        y_cpu: torch.Tensor,
        device: torch.device,
        non_blocking: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_cpu.to(device, non_blocking=non_blocking and x_cpu.is_pinned())
        y = y_cpu.to(device, non_blocking=non_blocking and y_cpu.is_pinned())
        return x, y


def make_memmap_lm_batch(
    dataset: MemmapTokenDataset,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return dataset.make_batch(batch_size, seq_len, device, generator, rank, world_size)
