"""Small contracts at the model, data, statistics, and supervision boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

Batch = Mapping[str, Any]


@runtime_checkable
class ArmRuntime(Protocol):
    def train_batch(
        self, model_input: Batch, targets: Any, *, rng: Any, stream: Any
    ) -> float: ...
    def evaluate_batch(
        self, model_input: Batch, *, rng: Any, stream: Any
    ) -> Mapping[str, Any]: ...
    def checkpoint_bytes(self, *, update: int) -> bytes: ...


class ArmFactory(Protocol):
    def __call__(self, arm: str, training_seed: int, device: str) -> ArmRuntime: ...


class DataFactory(Protocol):
    def train_batches(self, training_seed: int) -> Iterable[Batch]: ...
    def evaluation_batches(self, training_seed: int) -> Iterable[Batch]: ...


class StatsSink(Protocol):
    def record(
        self, arm: str, training_seed: int, payload: Mapping[str, Any]
    ) -> None: ...


class PeakSampler(Protocol):
    def start(self, arm: str, training_seed: int) -> None: ...
    def sample(self) -> None: ...
    def stop(self) -> Mapping[str, Any]: ...


class PrearmedSupervisor(Protocol):
    """A status-only supervisor; implementations must never receive metrics."""

    def assert_prearmed(self) -> None: ...
    def completed(self, receipt_path: Path) -> None: ...
    def crashed(self, reason: str) -> None: ...
