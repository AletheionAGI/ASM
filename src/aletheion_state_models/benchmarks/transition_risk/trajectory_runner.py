"""Public phase facade for the sealed ATTR-TG1 experiment."""

from __future__ import annotations

from pathlib import Path

import torch

from .trajectory_manifests import default_trajectory_protocol
from .trajectory_models import build_trajectory_arm
from .trajectory_protocol_io import (
    BATCH_SIZE,
    LEARNING_RATE,
    UPDATES,
    WEIGHT_DECAY,
    TrajectoryPaths,
    create_protocol_preseal,
)
from .trajectory_test_runner import (
    checkpoint_matrix,
    evaluate_fresh_test,
    open_fresh_test,
    seal_validation,
)
from .trajectory_train_runner import train_one, validate_one


class TrajectoryRunner:
    """Expose one fail-closed method per registered protocol phase."""

    def __init__(
        self,
        root: str | Path,
        *,
        factory=build_trajectory_arm,
        device: str | torch.device = "cpu",
    ) -> None:
        self.paths = TrajectoryPaths(Path(root).resolve())
        self.factory = factory
        self.device = torch.device(device)
        self.protocol = default_trajectory_protocol()

    def preseal(self) -> Path:
        return create_protocol_preseal(self.paths, self.protocol)

    def train(self, arm: str, seed: int) -> Path:
        return train_one(
            self.paths, self.protocol, self.factory, self.device, arm, seed
        )

    def validate(self, arm: str, seed: int) -> Path:
        return validate_one(
            self.paths, self.protocol, self.factory, self.device, arm, seed
        )

    def checkpoint_matrix(self):
        return checkpoint_matrix(self.paths)

    def validation_seal(self) -> Path:
        return seal_validation(self.paths, self.protocol)

    def open_test(self) -> tuple:
        return open_fresh_test(self.paths)

    def evaluate_test(self):
        return evaluate_fresh_test(self.paths, self.protocol, self.factory, self.device)


__all__ = [
    "BATCH_SIZE",
    "LEARNING_RATE",
    "UPDATES",
    "WEIGHT_DECAY",
    "TrajectoryPaths",
    "TrajectoryRunner",
]
