"""Sequential, single-stream CUDA training/evaluation orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .checkpoint import write_exact_checkpoint
from .interfaces import (
    ArmFactory,
    DataFactory,
    PeakSampler,
    PrearmedSupervisor,
    StatsSink,
)
from .policy import (
    ARMS,
    CANDIDATES,
    TRAINING_SEEDS,
    DraftRunConfig,
    configure_torch,
    derive_seed64,
    prepare_deterministic_environment,
)

MODEL_FIELDS = ("history_bytes", "candidate4s", "masks", "logical_lengths")


@dataclass(frozen=True)
class ArmReceipt:
    arm: str
    training_seed: int
    updates: int
    checkpoint: str
    checkpoint_sha256: str
    peak: Mapping[str, Any]
    evaluations: int


@dataclass(frozen=True)
class RunReceipt:
    status: str
    topology: str
    synthetic: bool
    arms: tuple[ArmReceipt, ...]
    receipt_path: str


class _SyntheticRng:
    def __init__(self, seed: int) -> None:
        self.seed = seed


class DraftEngine:
    """Fail-closed driver. It does not generate data, select checkpoints, or lock protocol."""

    def __init__(
        self,
        *,
        config: DraftRunConfig,
        arm_factory: ArmFactory,
        data: DataFactory,
        stats: StatsSink,
        peak_sampler: PeakSampler,
        supervisor: PrearmedSupervisor,
        output_root: Path,
    ) -> None:
        self.config = config
        self.arm_factory = arm_factory
        self.data = data
        self.stats = stats
        self.peak_sampler = peak_sampler
        self.supervisor = supervisor
        self.output_root = Path(output_root)
        self._torch: Any | None = None

    def run(self) -> RunReceipt:
        self.supervisor.assert_prearmed()
        prepare_deterministic_environment()
        try:
            receipts = self._run_sequential()
            receipt = self._write_receipt(receipts)
            self.supervisor.completed(Path(receipt.receipt_path))
            return receipt
        except BaseException as error:
            self.supervisor.crashed(f"{type(error).__name__}: {error}")
            raise

    def _runtime(self) -> tuple[str, Any]:
        if self.config.synthetic:
            return "synthetic", None
        import torch  # lazy: environment is frozen before import

        self._torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is prohibited")
        configure_torch(torch)
        return "cuda", torch

    def _run_sequential(self) -> tuple[ArmReceipt, ...]:
        device, torch = self._runtime()
        receipts: list[ArmReceipt] = []
        expected_manifests: dict[int, tuple[str, ...]] = {}
        for seed in TRAINING_SEEDS:
            for arm in ARMS:
                stream = None if torch is None else torch.cuda.Stream()
                with self._stream_context(stream):
                    runtime = self.arm_factory(arm, seed, device)
                rng_train = self._rng("train", seed, torch)
                rng_eval = self._rng("evaluation", seed, torch)
                self.peak_sampler.start(arm, seed)
                manifests = self._train(runtime, seed, stream, rng_train)
                if seed in expected_manifests and manifests != expected_manifests[seed]:
                    raise RuntimeError("batch manifests differ across arms")
                expected_manifests.setdefault(seed, manifests)
                evaluations = self._evaluate(runtime, arm, seed, stream, rng_eval)
                if torch is not None:
                    stream.synchronize()
                peak = self.peak_sampler.stop()
                if int(peak["peak_bytes"]) > self.config.peak_cap_bytes:
                    raise RuntimeError("20 GiB peak cap exceeded")
                payload = runtime.checkpoint_bytes(update=self.config.updates)
                path, digest = write_exact_checkpoint(
                    self.output_root / "checkpoints",
                    arm,
                    seed,
                    self.config.updates,
                    payload,
                    synthetic=self.config.synthetic,
                )
                receipts.append(
                    ArmReceipt(
                        arm,
                        seed,
                        self.config.updates,
                        str(path),
                        digest,
                        peak,
                        evaluations,
                    )
                )
        return tuple(receipts)

    def _stream_context(self, stream: Any) -> Any:
        if self._torch is None:
            return nullcontext()
        return self._torch.cuda.stream(stream)

    def _rng(self, purpose: str, seed: int, torch: Any | None) -> Any:
        derived = derive_seed64(purpose, seed)
        if torch is None:
            return _SyntheticRng(derived)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(derived)
        return generator

    def _train(self, runtime: Any, seed: int, stream: Any, rng: Any) -> tuple[str, ...]:
        iterator = iter(self.data.train_batches(seed))
        manifests: list[str] = []
        for update in range(1, self.config.updates + 1):
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    f"training data ended before update {update}"
                ) from error
            model_input = _model_input(batch)
            if "targets" not in batch or "manifest_hash" not in batch:
                raise ValueError("training batch lacks targets or manifest_hash")
            with self._stream_context(stream):
                loss = runtime.train_batch(
                    model_input, batch["targets"], rng=rng, stream=stream
                )
            if not _finite(loss):
                raise RuntimeError(f"nonfinite loss at update {update}")
            manifests.append(str(batch["manifest_hash"]))
            self.peak_sampler.sample()
        try:
            next(iterator)
        except StopIteration:
            return tuple(manifests)
        raise RuntimeError(
            "training data contains batches beyond the frozen update count"
        )

    def _evaluate(
        self, runtime: Any, arm: str, seed: int, stream: Any, rng: Any
    ) -> int:
        count = 0
        for batch in self.data.evaluation_batches(seed):
            with self._stream_context(stream):
                scores = runtime.evaluate_batch(
                    _model_input(batch), rng=rng, stream=stream
                )
            if not _finite(scores):
                raise RuntimeError("nonfinite evaluation output")
            self.stats.record(arm, seed, scores)
            self.peak_sampler.sample()
            count += 1
        if count == 0:
            raise RuntimeError("evaluation batches are empty")
        return count

    def _write_receipt(self, arms: tuple[ArmReceipt, ...]) -> RunReceipt:
        self.output_root.mkdir(parents=True, exist_ok=True)
        path = self.output_root / "draft-run-receipt.json"
        if path.exists():
            raise FileExistsError(f"receipt already exists: {path}")
        raw = {
            "status": "COMPLETED",
            "topology": "seed-major/arm-minor/sequential/one-stream",
            "synthetic": self.config.synthetic,
            "arms": [asdict(item) for item in arms],
        }
        raw["payload_sha256"] = hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path.write_text(
            json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return RunReceipt(
            "COMPLETED", raw["topology"], self.config.synthetic, arms, str(path)
        )


def _model_input(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    if any(field not in batch for field in MODEL_FIELDS):
        raise ValueError("batch lacks an exact model input field")
    result = {field: batch[field] for field in MODEL_FIELDS}
    histories = result["history_bytes"]
    candidates = result["candidate4s"]
    batch_size = getattr(histories, "shape", (None, None))[0]
    if getattr(histories, "shape", None) != (batch_size, 256):
        raise ValueError("history_bytes must have shape [batch,256]")
    if getattr(candidates, "shape", None) != (batch_size, len(CANDIDATES), 4):
        raise ValueError("candidate4s must have shape [batch,6,4]")
    if getattr(result["masks"], "shape", None) != (batch_size, len(CANDIDATES)):
        raise ValueError("masks must have shape [batch,6]")
    if getattr(result["logical_lengths"], "shape", None) != (batch_size,):
        raise ValueError("logical_lengths must have shape [batch]")
    return result


def _finite(value: Any) -> bool:
    import math

    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if hasattr(value, "isfinite"):
        result = value.isfinite()
        return bool(result.all()) if hasattr(result, "all") else bool(result)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if hasattr(value, "__array__"):
        import numpy as np  # lazy synthetic/test adapter

        return bool(np.isfinite(np.asarray(value)).all())
    return True
