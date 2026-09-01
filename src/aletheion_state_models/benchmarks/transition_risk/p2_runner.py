"""Two-stage orchestration for the sealed five-seed ATTR P2 benchmark."""

from __future__ import annotations
from dataclasses import asdict
import gc
import json
from pathlib import Path
import torch
from .dataset import MODEL_FEATURES, make_episodes, make_worlds
from .leakage import (
    FeatureAvailability,
    audit_episode_splits,
    audit_feature_availability,
)
from .p2_checkpoint import load_terminal_checkpoint, save_terminal_checkpoint
from .p2_evaluation import evaluate_episodes, write_episode_records_jsonl
from .p2_models import P2_ARMS, build_p2_arm
from .p2_seal import (
    canonical_sha256,
    create_p2_seal,
    default_p2_specs,
    open_p2_seal,
    write_p2_seal,
)
from .pilot import event_prevalence
from .training import train_arm

P2_SEEDS = (29, 43, 71, 89, 107)
P2_UPDATES = 1000
TRAIN_WORLDS = 64
VALIDATION_WORLDS = 16
EPISODES_PER_WORLD = 4


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_path(run_root: str | Path, seed: int, arm: str) -> Path:
    return Path(run_root) / "checkpoints" / f"seed_{seed}__{arm}.pt"


def result_path(run_root: str | Path, seed: int, arm: str) -> Path:
    return Path(run_root) / "results" / f"seed_{seed}__{arm}.json"


def prediction_path(run_root: str | Path, split: str, seed: int, arm: str) -> Path:
    return Path(run_root) / "predictions" / split / f"seed_{seed}__{arm}.jsonl"


def all_checkpoint_paths(run_root: str | Path) -> list[Path]:
    return [
        checkpoint_path(run_root, seed, arm) for seed in P2_SEEDS for arm in P2_ARMS
    ]


def checkpoint_matrix(run_root: str | Path) -> dict[tuple[str, int], Path]:
    return {
        (arm, seed): checkpoint_path(run_root, seed, arm)
        for arm in P2_ARMS
        for seed in P2_SEEDS
    }


def _preseal_payload(run_root: str | Path) -> dict:
    return {
        "experiment": "ATTR P2 sealed predictive benchmark",
        "training_seeds": list(P2_SEEDS),
        "arms": list(P2_ARMS),
        "updates_per_arm": P2_UPDATES,
        "train_worlds": TRAIN_WORLDS,
        "validation_worlds": VALIDATION_WORLDS,
        "episodes_per_world": EPISODES_PER_WORLD,
        "test_specs": [asdict(spec) for spec in default_p2_specs()],
        "checkpoint_ids": [path.name for path in all_checkpoint_paths(run_root)],
        "calibration": "per-seed validation threshold at FPR<=0.05",
        "g1_next_state_nll_noninferiority_margin": 0.02,
        "g2_min_auprc_gain": 0.03,
        "g2_max_brier_degradation": 0.01,
        "g5_min_positive_training_seeds": 4,
        "test_opened": False,
    }


def ensure_preseal(output: str | Path, run_root: str | Path) -> dict:
    """Write immutable test specs before any P2 training starts."""
    output = Path(output)
    payload = _preseal_payload(run_root)
    document = {"payload": payload, "sha256": canonical_sha256(payload)}
    path = output / "test_spec_preseal.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != document:
            raise ValueError(
                "existing ATTR P2 test preseal does not match registered protocol"
            )
    else:
        _write_json(path, document)
    return document


def _training_data(seed: int):
    worlds = make_worlds(TRAIN_WORLDS + VALIDATION_WORLDS, seed, max_steps=16)
    train_worlds = worlds[:TRAIN_WORLDS]
    validation_worlds = worlds[TRAIN_WORLDS:]
    train = make_episodes(train_worlds, EPISODES_PER_WORLD, seed)
    validation = make_episodes(validation_worlds, EPISODES_PER_WORLD, seed + 1)
    feature_audit = audit_feature_availability(
        [FeatureAvailability(name, 0, 0) for name in MODEL_FEATURES],
        {"normalization": "train"},
        "validation",
    )
    split_audit = audit_episode_splits(
        [item.episode_id for item in train + validation],
        ["train"] * len(train) + ["validation"] * len(validation),
    )
    feature_audit.require_pass()
    split_audit.require_pass()
    return train, validation


def train_and_freeze(
    root: str | Path, output: str | Path, run_root: str | Path, *, device: str = "cuda"
) -> list[dict]:
    """Train all 30 arms without constructing a test episode."""
    root = Path(root)
    output = Path(output)
    run_root = Path(run_root)
    ensure_preseal(output, run_root)
    results = []
    for seed in P2_SEEDS:
        train, validation = _training_data(seed)
        for arm in P2_ARMS:
            checkpoint = checkpoint_path(run_root, seed, arm)
            result_file = result_path(run_root, seed, arm)
            if checkpoint.exists() and result_file.exists():
                results.append(json.loads(result_file.read_text()))
                continue
            adapter, heads, metadata = build_p2_arm(root, arm, seed, P2_UPDATES)
            trained = train_arm(
                arm,
                adapter,
                heads,
                train,
                validation,
                updates=P2_UPDATES,
                batch_size=4,
                seed=seed,
                device=device,
            ).to_dict()
            trained.update(metadata)
            trained.update(
                {
                    "seed": seed,
                    "test_opened": False,
                    "train_prevalence": event_prevalence(train),
                    "validation_prevalence": event_prevalence(validation),
                }
            )
            checkpoint_record = save_terminal_checkpoint(
                checkpoint,
                adapter,
                heads,
                {"arm": arm, "seed": seed, "updates": P2_UPDATES, "test_opened": False},
            )
            evaluation = evaluate_episodes(adapter, heads, validation, device=device)
            write_episode_records_jsonl(
                prediction_path(run_root, "validation", seed, arm), evaluation.records
            )
            trained["validation_episode_metrics"] = evaluation.metrics
            trained["checkpoint"] = checkpoint_record
            _write_json(result_file, trained)
            results.append(trained)
            print(
                {
                    "p2_frozen": arm,
                    "seed": seed,
                    "checkpoint_sha256": checkpoint_record["sha256"],
                },
                flush=True,
            )
            del adapter, heads, evaluation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def seal_and_open(output: str | Path, run_root: str | Path):
    """Hash all terminal checkpoints, verify the preseal, then generate test data."""
    output = Path(output)
    run_root = Path(run_root)
    preseal = ensure_preseal(output, run_root)
    checkpoints = checkpoint_matrix(run_root)
    seal = create_p2_seal(checkpoints)
    if [asdict(spec) for spec in seal.splits] != preseal["payload"]["test_specs"]:
        raise ValueError("final P2 seal changed the pre-registered test specs")
    write_p2_seal(seal, output / "dataset_seal.json")
    specs = open_p2_seal(seal, checkpoints)
    opened = {}
    for spec in specs:
        worlds = make_worlds(
            spec.world_count,
            spec.seed,
            dynamic_family=spec.dynamic_family,
            max_steps=spec.max_steps,
        )
        opened[spec.test_id] = tuple(
            make_episodes(worlds, spec.episodes_per_world, spec.seed)
        )
    _write_json(
        output / "test_open_event.json",
        {
            "seal_sha256": seal.sha256,
            "checkpoint_count": len(checkpoints),
            "test_opened_after_all_checkpoints": True,
        },
    )
    return opened


def evaluate_opened(
    root: str | Path, run_root: str | Path, opened, *, device: str = "cuda"
) -> None:
    """Evaluate immutable checkpoints on opened ID/shift/OOD episodes."""
    root = Path(root)
    run_root = Path(run_root)
    for seed in P2_SEEDS:
        for arm in P2_ARMS:
            missing = [
                (split_name, episodes)
                for split_name, episodes in opened.items()
                if not prediction_path(run_root, split_name, seed, arm).exists()
            ]
            if not missing:
                continue
            adapter, heads, _ = build_p2_arm(root, arm, seed, P2_UPDATES)
            load_terminal_checkpoint(
                checkpoint_path(run_root, seed, arm), adapter, heads, device=device
            )
            for split_name, episodes in missing:
                evaluation = evaluate_episodes(adapter, heads, episodes, device=device)
                write_episode_records_jsonl(
                    prediction_path(run_root, split_name, seed, arm),
                    evaluation.records,
                )
                print(
                    {
                        "p2_evaluated": arm,
                        "seed": seed,
                        "split": split_name,
                        "metrics": evaluation.metrics,
                    },
                    flush=True,
                )
            del adapter, heads, evaluation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


__all__ = [
    "P2_SEEDS",
    "P2_UPDATES",
    "all_checkpoint_paths",
    "ensure_preseal",
    "evaluate_opened",
    "seal_and_open",
    "train_and_freeze",
]
