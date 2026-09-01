"""Train-only ATTR orchestration with paired data, objectives, and audits."""

from __future__ import annotations
import json
from pathlib import Path
import torch
import yaml
from drm_language_emitter import DRMConfig, DRMEmitterModel
from transformer.tiny_transformer import TinyTransformerConfig, TinyTransformerLM
from .baselines import evaluate_controls
from .dataset import MODEL_FEATURES, make_episodes, make_worlds
from .leakage import (
    FeatureAvailability,
    audit_episode_splits,
    audit_feature_availability,
)
from .model_adapters import ASMModelAdapter, TransformerModelAdapter
from .model_heads import TransitionRiskHeads
from .render import render_summary
from .pilot import event_prevalence, write_pilot_manifest
from .training import train_arm


def _load_yaml(path: Path):
    with path.open() as handle:
        return yaml.safe_load(handle)


def _build_arms(root: Path, seed: int):
    torch.manual_seed(seed)
    asm_config = DRMConfig.from_dict(
        _load_yaml(root / "configs/tiny_drm_stronger.yaml")
        | {"sequence_mode": "directional_candidates"}
    )
    asm = DRMEmitterModel(asm_config)
    torch.manual_seed(seed)
    transformer_config = TinyTransformerConfig.from_dict(
        _load_yaml(root / "transformer/tiny_transformer_220k.yaml")
    )
    transformer = TinyTransformerLM(transformer_config)
    return [
        (
            "asm_x_directional",
            ASMModelAdapter(asm),
            TransitionRiskHeads(asm_config.d_state, 6, hidden_dim=32),
        ),
        (
            "tiny_transformer_220k",
            TransformerModelAdapter(transformer),
            TransitionRiskHeads(transformer_config.d_model, 6, hidden_dim=32),
        ),
    ]


def _audit(train, validation):
    feature = audit_feature_availability(
        [FeatureAvailability(name, 0, 0) for name in MODEL_FEATURES],
        {"normalization": "train"},
        "validation",
    )
    split = audit_episode_splits(
        [item.episode_id for item in train + validation],
        ["train"] * len(train) + ["validation"] * len(validation),
    )
    feature.require_pass()
    split.require_pass()
    return feature, split


def run_attr_train_only(
    root: str | Path,
    output_dir: str | Path,
    *,
    updates: int,
    seed: int = 17,
    device: str = "cpu",
    train_world_count: int = 8,
    validation_world_count: int = 4,
    episodes_per_world: int = 2,
    status: str = "train_only_pilot_test_sealed",
) -> dict:
    """Train paired arms without generating or inspecting any test world."""
    if updates < 1 or train_world_count < 1 or validation_world_count < 1:
        raise ValueError("updates and world counts must be positive")
    root = Path(root)
    output = Path(output_dir)
    worlds = make_worlds(train_world_count + validation_world_count, seed, max_steps=16)
    train_worlds = worlds[:train_world_count]
    validation_worlds = worlds[train_world_count:]
    train = make_episodes(train_worlds, episodes_per_world, seed)
    validation = make_episodes(validation_worlds, episodes_per_world, seed + 1)
    feature_audit, split_audit = _audit(train, validation)
    controls = evaluate_controls(train, validation)
    results = [
        train_arm(
            name,
            adapter,
            heads,
            train,
            validation,
            updates=updates,
            batch_size=4,
            seed=seed,
            device=device,
        ).to_dict()
        for name, adapter, heads in _build_arms(root, seed)
    ]
    left, right = results
    mismatch = abs(left["trainable_parameters"] - right["trainable_parameters"]) / max(
        left["trainable_parameters"], right["trainable_parameters"]
    )
    summary = {
        "status": status,
        "seed": seed,
        "updates": updates,
        "device": device,
        "test_worlds_generated": False,
        "worlds": {
            "train": len(train_worlds),
            "validation": len(validation_worlds),
            "episodes_per_world": episodes_per_world,
        },
        "horizons": [1, 4, 8, 16],
        "features": list(MODEL_FEATURES),
        "audits": {
            "feature_leakage": feature_audit.passed,
            "episode_split": split_audit.passed,
            "threshold_selection": "validation_only",
        },
        "parameter_mismatch_fraction": mismatch,
        "event_prevalence": {
            "train": event_prevalence(train),
            "validation": event_prevalence(validation),
        },
        "controls": {
            "persistence_next_state_mse": controls.persistence_next_state_mse,
            "kalman_next_state_mse": controls.kalman_next_state_mse,
            "markov_h8": {
                "auprc": controls.markov_h8.auprc,
                "brier": controls.markov_h8.brier,
            },
        },
        "arms": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if status == "train_only_pilot_test_sealed":
        write_pilot_manifest(root, output, summary)
    render_summary(output / "summary.json")
    return summary


def run_attr_p0_smoke(
    root: str | Path,
    output_dir: str | Path,
    *,
    updates: int = 2,
    seed: int = 17,
    device: str = "cpu",
) -> dict:
    """Run a small integration test; this is not sealed benchmark evidence."""
    return run_attr_train_only(
        root,
        output_dir,
        updates=updates,
        seed=seed,
        device=device,
        status="smoke_only_not_sealed",
    )


__all__ = ["run_attr_p0_smoke", "run_attr_train_only"]
