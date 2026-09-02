"""Terminal validation orchestration for allowed ATTR-RTG artifacts."""

from __future__ import annotations

from pathlib import Path

import torch

from .rtg_artifacts import atomic_write_json
from .rtg_backbones import load_terminal_backbone
from .rtg_checkpoint import load_terminal_checkpoint
from .rtg_config import TRAINING_SEEDS
from .rtg_heads import PhysicalD, TransitionG
from .rtg_pipeline_train import (
    _arm_dir,
    _load_normalization,
    _load_records,
    _metadata,
    _verified_allowed_data,
)
from .rtg_validation import preliminary_rtg1, terminal_ce


def run_validation(
    root: str | Path,
    output: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Path, ...]:
    """Evaluate only terminal checkpoints on the frozen validation split."""
    data = _verified_allowed_data(root, output)
    validation_episodes = data[1].episodes
    output = Path(output)
    paths = []
    for kind in ("asm", "transformer"):
        for seed in TRAINING_SEEDS:
            arm = _arm_dir(output, kind, seed)
            backbone = load_terminal_backbone(
                root, kind, seed, arm / "backbone.pt", device=device
            )
            normalization = _load_normalization(arm / "normalization.json")
            records = _load_records(arm / "states_validation.pt")
            g = load_terminal_checkpoint(
                arm / "G.pt",
                TransitionG(),
                expected_kind=f"{kind}-G",
                expected_seed=seed,
                expected_metadata=_metadata(),
            ).to(device)
            d = load_terminal_checkpoint(
                arm / "D.pt",
                PhysicalD(),
                expected_kind=f"{kind}-D",
                expected_seed=seed,
                expected_metadata=_metadata(),
            ).to(device)
            metrics = preliminary_rtg1(
                records, normalization, g, d, seed, device=device
            )
            metrics["terminal_next_byte_ce"] = terminal_ce(
                backbone, validation_episodes, device=device
            )
            metrics.update(
                {
                    "kind": kind,
                    "training_seed": seed,
                    "split": "validation",
                    "terminal_update": 1000,
                }
            )
            paths.append(
                atomic_write_json(arm / "validation_metrics.json", metrics)
            )
    return tuple(paths)
