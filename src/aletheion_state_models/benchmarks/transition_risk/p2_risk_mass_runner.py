"""Train and evaluate the post-hoc ASM-X native-risk-mass diagnostic."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch

from .dataset import make_episodes, make_worlds
from .p2_checkpoint import load_terminal_checkpoint, save_terminal_checkpoint
from .p2_evaluation import evaluate_episodes, write_episode_records_jsonl
from .p2_risk_mass_models import (
    BASELINE_ARM,
    RISK_MASS_ARM,
    build_risk_mass_arm,
    verify_parameter_and_initialization_parity,
)
from .p2_runner import (
    P2_SEEDS,
    P2_UPDATES,
    _training_data,
    checkpoint_matrix,
    checkpoint_path,
    prediction_path,
    result_path,
)
from .p2_seal import canonical_sha256, open_p2_seal, read_p2_seal
from .pilot import event_prevalence
from .training import train_arm

MANIFEST_NAME = "risk_mass_extension_pretrain_manifest.json"
CHECKPOINT_SEAL_NAME = "risk_mass_extension_checkpoint_seal.json"
_TRAINING_SOURCES = (
    "configs/tiny_drm_stronger.yaml",
    "src/drm_language_emitter/config.py",
    "src/drm_language_emitter/geometric_steps.py",
    "src/drm_language_emitter/metric.py",
    "src/drm_language_emitter/model.py",
    "src/drm_language_emitter/risk.py",
    "src/aletheion_state_models/benchmarks/transition_risk/dataset.py",
    "src/aletheion_state_models/benchmarks/transition_risk/model_adapters.py",
    "src/aletheion_state_models/benchmarks/transition_risk/model_heads.py",
    "src/aletheion_state_models/benchmarks/transition_risk/training.py",
    "src/aletheion_state_models/benchmarks/transition_risk/p2_risk_mass_models.py",
    "src/aletheion_state_models/benchmarks/transition_risk/p2_risk_mass_runner.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _manifest_payload(root: Path, output: Path, run_root: Path) -> dict:
    original_preseal = output / "test_spec_preseal.json"
    original_seal = output / "dataset_seal.json"
    open_event = output / "test_open_event.json"
    if not all(
        path.is_file() for path in (original_preseal, original_seal, open_event)
    ):
        raise ValueError("completed sealed P2 artifacts are required")
    baseline = []
    for seed in P2_SEEDS:
        path = checkpoint_path(run_root, seed, BASELINE_ARM)
        if not path.is_file():
            raise ValueError(f"missing frozen baseline checkpoint: {path}")
        baseline.append({"seed": seed, "path": path.name, "sha256": _sha256(path)})
    return {
        "experiment": "ATTR P2 post-hoc native-risk-mass diagnostic",
        "confirmatory_status": "posthoc_exploratory_not_registered_p2_arm",
        "public_names": {
            BASELINE_ARM: "ASM-X Base",
            RISK_MASS_ARM: "ASM-X + Native Risk Mass",
        },
        "selection_timing": "requested after sealed P2 results were observed",
        "training_seeds": list(P2_SEEDS),
        "updates_per_arm": P2_UPDATES,
        "data_and_heads": "identical to sealed P2; calibration remains validation-only",
        "sole_config_delta": {
            "use_powerlaw_risk": {"baseline": False, "variant": True}
        },
        "parameter_parity": verify_parameter_and_initialization_parity(root),
        "baseline_checkpoints": baseline,
        "extension_checkpoint_ids": [
            checkpoint_path(run_root, seed, RISK_MASS_ARM).name for seed in P2_SEEDS
        ],
        "original_preseal_sha256": _sha256(original_preseal),
        "original_dataset_seal_sha256": _sha256(original_seal),
        "original_test_open_event_sha256": _sha256(open_event),
        "test_already_open_before_selection": True,
        "test_access_during_extension_training": False,
        "training_source_sha256": {
            name: _sha256(root / name) for name in _TRAINING_SOURCES
        },
        "claim_limit": "diagnostic comparison only; cannot revise registered G2 or claim confirmatory superiority",
    }


def ensure_extension_manifest(
    root: str | Path, output: str | Path, run_root: str | Path
) -> dict:
    """Freeze the post-hoc diagnostic before its first training update."""
    root, output, run_root = Path(root), Path(output), Path(run_root)
    payload = _manifest_payload(root, output, run_root)
    document = {"payload": payload, "sha256": canonical_sha256(payload)}
    path = output / MANIFEST_NAME
    if path.exists():
        if json.loads(path.read_text()) != document:
            raise ValueError("risk-mass extension manifest changed after freezing")
    else:
        _write_json(path, document)
    return document


def train_risk_mass_extension(
    root: str | Path, output: str | Path, run_root: str | Path, *, device: str = "cuda"
) -> list[dict]:
    """Train five risk-mass checkpoints without reading any test prediction."""
    root, output, run_root = Path(root), Path(output), Path(run_root)
    ensure_extension_manifest(root, output, run_root)
    results = []
    for seed in P2_SEEDS:
        train, validation = _training_data(seed)
        checkpoint = checkpoint_path(run_root, seed, RISK_MASS_ARM)
        result_file = result_path(run_root, seed, RISK_MASS_ARM)
        if checkpoint.exists() and result_file.exists():
            results.append(json.loads(result_file.read_text()))
            continue
        adapter, heads, metadata = build_risk_mass_arm(root, seed, P2_UPDATES)
        trained = train_arm(
            RISK_MASS_ARM,
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
                "test_was_already_open": True,
                "test_accessed_during_training": False,
                "train_prevalence": event_prevalence(train),
                "validation_prevalence": event_prevalence(validation),
            }
        )
        record = save_terminal_checkpoint(
            checkpoint,
            adapter,
            heads,
            {
                "arm": RISK_MASS_ARM,
                "seed": seed,
                "updates": P2_UPDATES,
                "posthoc": True,
                "test_accessed_during_training": False,
            },
        )
        evaluation = evaluate_episodes(adapter, heads, validation, device=device)
        write_episode_records_jsonl(
            prediction_path(run_root, "validation", seed, RISK_MASS_ARM),
            evaluation.records,
        )
        trained["validation_episode_metrics"] = evaluation.metrics
        trained["checkpoint"] = record
        _write_json(result_file, trained)
        results.append(trained)
        print(
            {"risk_mass_frozen": seed, "checkpoint_sha256": record["sha256"]},
            flush=True,
        )
        del adapter, heads, evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def seal_extension(root: str | Path, output: str | Path, run_root: str | Path) -> dict:
    """Verify source, baseline, and all five extension checkpoints."""
    root, output, run_root = Path(root), Path(output), Path(run_root)
    manifest = ensure_extension_manifest(root, output, run_root)
    checkpoints = []
    for seed in P2_SEEDS:
        result = json.loads(result_path(run_root, seed, RISK_MASS_ARM).read_text())
        path = checkpoint_path(run_root, seed, RISK_MASS_ARM)
        digest = _sha256(path)
        if digest != result["checkpoint"]["sha256"]:
            raise ValueError(f"risk-mass checkpoint hash mismatch for seed {seed}")
        checkpoints.append({"seed": seed, "path": path.name, "sha256": digest})
    payload = {"manifest_sha256": manifest["sha256"], "checkpoints": checkpoints}
    document = {"payload": payload, "sha256": canonical_sha256(payload)}
    path = output / CHECKPOINT_SEAL_NAME
    if path.exists() and json.loads(path.read_text()) != document:
        raise ValueError("risk-mass checkpoint seal changed")
    _write_json(path, document)
    return document


def evaluate_risk_mass_extension(
    root: str | Path, output: str | Path, run_root: str | Path, *, device: str = "cuda"
) -> None:
    """Evaluate only after all five post-hoc checkpoints are sealed."""
    root, output, run_root = Path(root), Path(output), Path(run_root)
    seal_extension(root, output, run_root)
    original_seal = read_p2_seal(output / "dataset_seal.json")
    specs = open_p2_seal(original_seal, checkpoint_matrix(run_root))
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
    for seed in P2_SEEDS:
        missing = [
            (name, episodes)
            for name, episodes in opened.items()
            if not prediction_path(run_root, name, seed, RISK_MASS_ARM).exists()
        ]
        if not missing:
            continue
        adapter, heads, _ = build_risk_mass_arm(root, seed, P2_UPDATES)
        load_terminal_checkpoint(
            checkpoint_path(run_root, seed, RISK_MASS_ARM),
            adapter,
            heads,
            device=device,
        )
        for split, episodes in missing:
            evaluation = evaluate_episodes(adapter, heads, episodes, device=device)
            write_episode_records_jsonl(
                prediction_path(run_root, split, seed, RISK_MASS_ARM),
                evaluation.records,
            )
            print(
                {
                    "risk_mass_evaluated": seed,
                    "split": split,
                    "metrics": evaluation.metrics,
                },
                flush=True,
            )
        del adapter, heads, evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "ensure_extension_manifest",
    "evaluate_risk_mass_extension",
    "seal_extension",
    "train_risk_mass_extension",
]
