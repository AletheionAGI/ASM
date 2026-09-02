from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from attr_rtg_rcmz.engine import DraftEngine
from attr_rtg_rcmz.monitoring import SyntheticPeakSampler
from attr_rtg_rcmz.policy import ARMS, TRAINING_SEEDS, DraftRunConfig, derive_seed64


class SyntheticData:
    def __init__(self, updates: int) -> None:
        self.updates = updates

    @staticmethod
    def _batch(index: int, training: bool) -> dict:
        batch = {
            "history_bytes": np.zeros((2, 256), dtype=np.uint8),
            "candidate4s": np.zeros((2, 6, 4), dtype=np.float32),
            "masks": np.ones((2, 6), dtype=bool),
            "logical_lengths": np.full((2,), 256),
            "private_id": "must-not-reach-model",
        }
        if training:
            batch.update(
                targets=np.zeros((2, 6), dtype=np.float32),
                manifest_hash=f"batch-{index}",
            )
        return batch

    def train_batches(self, training_seed: int):
        return (self._batch(index, True) for index in range(self.updates))

    def evaluation_batches(self, training_seed: int):
        return iter((self._batch(0, False),))


class SyntheticArm:
    def __init__(self, arm: str, seed: int, calls: list) -> None:
        self.arm, self.seed, self.calls = arm, seed, calls
        self.updates = 0

    def train_batch(self, model_input, targets, *, rng, stream):
        assert tuple(model_input) == (
            "history_bytes",
            "candidate4s",
            "masks",
            "logical_lengths",
        )
        assert model_input["candidate4s"].shape == (2, 6, 4)
        assert stream is None
        self.updates += 1
        return 0.5

    def evaluate_batch(self, model_input, *, rng, stream):
        self.calls.append((self.seed, self.arm, self.updates, rng.seed))
        return {"common24": np.zeros((2, 6)), "native": np.zeros((2, 6))}

    def checkpoint_bytes(self, *, update):
        assert update == self.updates
        return f"{self.seed}:{self.arm}:{update}".encode()


class Sink:
    def __init__(self):
        self.rows = []

    def record(self, arm, training_seed, payload):
        self.rows.append((arm, training_seed, payload))


class Supervisor:
    def __init__(self):
        self.events = []

    def assert_prearmed(self):
        self.events.append("ARMED")

    def completed(self, receipt_path):
        self.events.append(("COMPLETED", receipt_path))

    def crashed(self, reason):
        self.events.append(("CRASH", reason))


def test_synthetic_dry_run_is_seed_major_and_arm_sequential(tmp_path: Path):
    calls, sink, supervisor = [], Sink(), Supervisor()
    engine = DraftEngine(
        config=DraftRunConfig(updates=2, synthetic=True),
        arm_factory=lambda arm, seed, device: SyntheticArm(arm, seed, calls),
        data=SyntheticData(2),
        stats=sink,
        peak_sampler=SyntheticPeakSampler(),
        supervisor=supervisor,
        output_root=tmp_path / "synthetic",
    )
    receipt = engine.run()
    assert [(seed, arm) for seed, arm, _, _ in calls] == [
        (s, a) for s in TRAINING_SEEDS for a in ARMS
    ]
    assert all(updates == 2 for _, _, updates, _ in calls)
    assert len(receipt.arms) == 20
    assert supervisor.events[0] == "ARMED" and supervisor.events[-1][0] == "COMPLETED"
    payload = json.loads(Path(receipt.receipt_path).read_text())
    assert payload["status"] == "COMPLETED"
    assert len(sink.rows) == 20


def test_seed_derivation_is_domain_separated_and_stable():
    assert derive_seed64("train", 29) == derive_seed64("train", 29)
    assert derive_seed64("train", 29) != derive_seed64("evaluation", 29)


def test_non_synthetic_update_count_and_synthetic_limit_fail_closed():
    with pytest.raises(ValueError, match="2,000"):
        DraftRunConfig(updates=2)
    with pytest.raises(ValueError, match="16"):
        DraftRunConfig(updates=17, synthetic=True)
