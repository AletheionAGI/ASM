"""Paths, immutable protocol files, and common corpora for ATTR-TG1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .trajectory_checkpoint import atomic_write_json, digest_files
from .trajectory_manifests import ARMS, OPTIMIZER_SEEDS, TrajectoryProtocol
from .trajectory_runtime import episode_digest, generate_split, write_common_manifests
from .trajectory_seal import (
    create_trajectory_preseal,
    read_trajectory_preseal,
    write_trajectory_preseal,
)

UPDATES = 1_000
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01


@dataclass(frozen=True)
class TrajectoryPaths:
    root: Path

    @property
    def artifacts(self) -> Path:
        return self.root / "runs/attr_trajectory_grounded_tg1"

    @property
    def data(self) -> Path:
        return self.artifacts / "data"

    @property
    def checkpoints(self) -> Path:
        return self.artifacts / "checkpoints"

    @property
    def predictions(self) -> Path:
        return self.artifacts / "predictions"

    @property
    def results(self) -> Path:
        return self.artifacts / "results"

    @property
    def preseal(self) -> Path:
        return self.artifacts / "trajectory_protocol_preseal.json"

    @property
    def seal(self) -> Path:
        return self.artifacts / "trajectory_checkpoint_seal.json"

    def checkpoint(self, arm: str, seed: int) -> Path:
        return self.checkpoints / f"seed_{seed}__{arm}.pt"

    def prediction(self, arm: str, seed: int, split: str) -> Path:
        return self.predictions / split / f"seed_{seed}__{arm}.jsonl"

    def result(self, arm: str, seed: int) -> Path:
        return self.results / f"seed_{seed}__{arm}.json"


def code_paths(paths: TrajectoryPaths) -> dict[str, Path]:
    root = paths.root
    package = root / "src/aletheion_state_models/benchmarks/transition_risk"
    files = list(package.glob("trajectory_*.py"))
    files += list((root / "world_model").glob("*.py"))
    files += list(root.glob("drm*.py"))
    files += list((root / "transformer").glob("*.py"))
    files += [
        package / "dataset.py",
        package / "model_adapters.py",
        root / "scripts/run_attr_trajectory_grounded.py",
        root / "configs/tiny_drm_stronger.yaml",
        root / "transformer/tiny_transformer_220k.yaml",
        root / "docs/ATTR_TG1_TRAJECTORY_GROUNDED_PROTOCOL.md",
        root / "docs/ATTR_TG1_TRAJECTORY_GROUNDED_PROTOCOL_ptbr.md",
    ]
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(set(files))
        if path.is_file()
    }


def data_paths(paths: TrajectoryPaths) -> dict[str, Path]:
    return {
        name: paths.data / f"{name}.manifest.json"
        for name in ("protocol", "train", "validation")
    }


def protocol_manifest(protocol: TrajectoryProtocol) -> dict:
    return {
        "protocol": "ATTR-TG1",
        "arms": list(ARMS),
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "selection": "terminal checkpoint only",
        "objective": "physical categorical trajectory NLL only",
        "forbidden_outputs": ["hazard", "unsafe", "severity", "time_to_hazard"],
        "common_head": "72-wide autoregressive H8 decoder",
        "asm_width_bridge": "parameter-free zero padding 64 to 72",
        "action_plan": "causal frozen open-loop H8; no STOP",
        "risk": "K256 physical rollouts through fixed external predicate",
        "test_control": "procedural fresh splits generated only after checkpoint seal",
        "registered_protocol": asdict(protocol),
    }


def create_protocol_preseal(
    paths: TrajectoryPaths, protocol: TrajectoryProtocol
) -> Path:
    """Generate train/validation manifests only, then freeze code and protocol."""
    if paths.artifacts.exists():
        raise FileExistsError("ATTR-TG1 artifacts already exist; preseal is one-shot")
    splits = tuple(
        item for item in protocol.splits if item.name in {"train", "validation"}
    )
    write_common_manifests(paths.data, splits)
    atomic_write_json(data_paths(paths)["protocol"], protocol_manifest(protocol))
    preseal = create_trajectory_preseal(
        code_paths(paths), data_paths(paths), protocol=protocol
    )
    return write_trajectory_preseal(preseal, paths.preseal)


def verify_preseal(paths: TrajectoryPaths) -> None:
    preseal = read_trajectory_preseal(paths.preseal)
    if preseal.code_manifest != digest_files(code_paths(paths)):
        raise ValueError("ATTR-TG1 code changed after preseal")
    if preseal.data_manifest != digest_files(data_paths(paths)):
        raise ValueError("ATTR-TG1 data manifest changed after preseal")


def common_episodes(paths: TrajectoryPaths, protocol: TrajectoryProtocol, name: str):
    spec = next(item for item in protocol.splits if item.name == name)
    episodes = generate_split(spec)
    manifest = json.loads(data_paths(paths)[name].read_text())
    expected = {
        "split": asdict(spec),
        "episodes": len(episodes),
        "sha256": episode_digest(episodes),
    }
    if manifest != expected:
        raise ValueError(f"deterministic {name} corpus changed after preseal")
    return episodes


def validate_identity(arm: str, seed: int) -> None:
    if arm not in ARMS or seed not in OPTIMIZER_SEEDS:
        raise ValueError("unregistered ATTR-TG1 arm or optimizer seed")


__all__ = [
    "BATCH_SIZE",
    "LEARNING_RATE",
    "UPDATES",
    "WEIGHT_DECAY",
    "TrajectoryPaths",
    "code_paths",
    "common_episodes",
    "create_protocol_preseal",
    "data_paths",
    "protocol_manifest",
    "validate_identity",
    "verify_preseal",
]
