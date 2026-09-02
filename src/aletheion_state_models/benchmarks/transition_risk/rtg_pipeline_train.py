"""Causal production orchestration for allowed pre-test ATTR-RTG training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .rtg_allowed_manifest import (
    AllowedSplitData,
    build_allowed_manifest,
    prepare_allowed_data,
    write_allowed_manifest,
)
from .rtg_artifacts import atomic_write_json, canonical_json
from .rtg_backbones import build_registered_backbone
from .rtg_checkpoint import (
    _atomic_create,
    save_terminal_checkpoint,
)
from .rtg_cloning import materialize_origin_truth
from .rtg_config import TRAINING_SEEDS
from .rtg_normalization import StateNormalization, fit_train_normalization
from .rtg_pipeline_truth import (
    validate_arm_input_equivalence,
    write_training_manifest,
    write_truth_manifest,
)
from .rtg_projection import make_registered_projection
from .rtg_state_export import ASMStateExporter, TransformerReadoutExporter
from .rtg_state_records import (
    CandidateStateInputRecord,
    CandidateStateRecord,
    attach_candidate_truths,
    export_candidate_state_inputs,
    stack_record_states,
)
from .rtg_train_backbone import train_backbone
from .rtg_train_heads import train_direct_c, train_physical_d, train_transition_g

ARMS = tuple(
    (kind, seed) for kind in ("asm", "transformer") for seed in TRAINING_SEEDS
)


@dataclass(frozen=True)
class TrainingArtifactSet:
    kind: str
    training_seed: int
    paths: tuple[Path, ...]


def _arm_dir(output: Path, kind: str, seed: int) -> Path:
    return output / f"{kind}_seed{seed}"


def _arm_name(kind: str, seed: int) -> str:
    return f"{kind}.seed{seed}"


def _metadata() -> dict[str, object]:
    return {"split": "train", "sealable": True, "protocol": "ATTR-RTG"}


def _normalization_payload(value: StateNormalization) -> dict[str, object]:
    return {
        "pre_mean": value.pre_mean.tolist(), "pre_std": value.pre_std.tolist(),
        "next_mean": value.next_mean.tolist(), "next_std": value.next_std.tolist(),
    }


def _input_record_payload(records: tuple[CandidateStateInputRecord, ...]) -> dict[str, object]:
    return {
        "schema": "ATTR-RTG-STATE-INPUTS-V1",
        "identity": [[item.split_id, item.world_id, item.episode_id, item.t,
                      item.action_index] for item in records],
        "pre_state": torch.stack([item.pre_state for item in records]),
        "next_state": torch.stack([item.next_state for item in records]),
        "fixed_frame": torch.stack([item.fixed_frame for item in records]),
    }


def _save_input_records(path: Path, records: tuple[CandidateStateInputRecord, ...]) -> Path:
    if not records:
        raise ValueError("cannot persist empty input state records")
    return _atomic_create(path, lambda stream: torch.save(_input_record_payload(records), stream))


def _record_payload(records: tuple[CandidateStateRecord, ...]) -> dict[str, object]:
    return {
        "identity": [
            [item.split_id, item.world_id, item.episode_id, item.t, item.action_index]
            for item in records
        ],
        "pre_state": torch.stack([item.pre_state for item in records]),
        "next_state": torch.stack([item.next_state for item in records]),
        "fixed_frame": torch.stack([item.fixed_frame for item in records]),
        "physical_target": torch.tensor([item.physical_target for item in records], dtype=torch.long),
        "persistence_target": torch.tensor([item.persistence_target for item in records], dtype=torch.long),
        "unsafe": torch.tensor([item.unsafe for item in records], dtype=torch.bool),
        "failure_delay": torch.tensor([item.failure_delay for item in records], dtype=torch.long),
    }


def _save_records(path: Path, records: tuple[CandidateStateRecord, ...]) -> Path:
    if not records:
        raise ValueError("cannot persist empty state records")
    return _atomic_create(path, lambda stream: torch.save(_record_payload(records), stream))


def _load_records(path: Path) -> tuple[CandidateStateRecord, ...]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"identity", "pre_state", "next_state", "fixed_frame", "physical_target",
                "persistence_target", "unsafe", "failure_delay"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("state record artifact schema differs")
    count = len(payload["identity"])
    if any(len(payload[key]) != count for key in required - {"identity"}):
        raise ValueError("state record artifact columns differ")
    return tuple(CandidateStateRecord(
        *identity, payload["pre_state"][index], payload["next_state"][index],
        payload["fixed_frame"][index],
        tuple(int(value) for value in payload["physical_target"][index]),
        tuple(int(value) for value in payload["persistence_target"][index]),
        bool(payload["unsafe"][index]), int(payload["failure_delay"][index]),
    ) for index, identity in enumerate(payload["identity"]))


def _load_normalization(path: Path) -> StateNormalization:
    values = json.loads(path.read_text(encoding="utf-8"))
    return StateNormalization(*(torch.tensor(values[key], dtype=torch.float32)
        for key in ("pre_mean", "pre_std", "next_mean", "next_std")))


def run_prepare(root: str | Path, output: str | Path) -> Path:
    data = prepare_allowed_data()
    return write_allowed_manifest(Path(output) / "allowed_manifest.json", root, data)


def _verified_allowed_data(root: str | Path, output: str | Path) -> tuple[AllowedSplitData, ...]:
    data = prepare_allowed_data()
    path = Path(output) / "allowed_manifest.json"
    expected = canonical_json(build_allowed_manifest(root, data)) + b"\n"
    if not path.is_file() or path.read_bytes() != expected:
        raise ValueError("allowed data differ from the precomputed manifest")
    return data


def _train_all_backbones(root, output, train, updates, device):
    models, paths = {}, {}
    for kind, seed in ARMS:
        model = build_registered_backbone(root, kind, seed)
        trained = train_backbone(model, train.episodes, seed, updates=updates, device=device)
        arm = _arm_dir(output, kind, seed)
        arm.mkdir(parents=True, exist_ok=True)
        path = save_terminal_checkpoint(
            arm / "backbone.pt", trained.model, kind=f"{kind}-backbone",
            training_seed=seed, terminal_update=updates, metadata=_metadata(),
            config=trained.model.config.to_dict(),
        )
        models[_arm_name(kind, seed)] = trained.model
        paths[_arm_name(kind, seed)] = path
    return models, paths


def _export_all_inputs(data, models):
    exports: dict[str, dict[str, tuple[CandidateStateInputRecord, ...]]] = {}
    for kind, seed in ARMS:
        model = models[_arm_name(kind, seed)]
        exporter = ASMStateExporter(model) if kind == "asm" else TransformerReadoutExporter(model)
        projection = make_registered_projection(kind)
        exports[_arm_name(kind, seed)] = {
            split.name: export_candidate_state_inputs(split.origins, exporter, projection)
            for split in data
        }
    validate_arm_input_equivalence(exports)
    return exports


def _materialize_all_truths(data):
    """This phase must be called only after `_export_all_inputs` fully returns."""
    return {
        split.name: tuple(materialize_origin_truth(origin) for origin in split.origins)
        for split in data
    }


def _save_all_input_states(output, exports):
    paths = {}
    for arm_name, splits in exports.items():
        kind, seed_text = arm_name.split(".seed")
        arm = _arm_dir(output, kind, int(seed_text))
        for split, values in splits.items():
            key = f"{arm_name}.input.{split}"
            paths[key] = _save_input_records(
                arm / f"state_inputs_{split}.pt", values
            )
    return paths


def _causal_export_then_truth(data, models, output):
    """Export all 30 batches, persist input-only ledgers, then branch truth."""
    exports = _export_all_inputs(data, models)
    input_paths = _save_all_input_states(output, exports)
    truths = _materialize_all_truths(data)
    return exports, input_paths, truths


def _attach_all(exports, truths):
    return {
        arm: {split: attach_candidate_truths(records, truths[split])
              for split, records in groups.items()}
        for arm, groups in exports.items()
    }


def _save_all_states(output, records):
    paths = {}
    for arm_name, splits in records.items():
        kind, seed_text = arm_name.split(".seed")
        arm = _arm_dir(output, kind, int(seed_text))
        for split, values in splits.items():
            key = f"{arm_name}.{split}"
            paths[key] = _save_records(arm / f"states_{split}.pt", values)
    return paths


def _train_all_heads(output, records, updates, device):
    head_paths, result_sets = {}, []
    for kind, seed in ARMS:
        name = _arm_name(kind, seed)
        train_records = records[name]["train"]
        pre, nxt = stack_record_states(train_records)
        normalization = fit_train_normalization(pre, nxt)
        arm = _arm_dir(output, kind, seed)
        normalization_path = atomic_write_json(
            arm / "normalization.json", _normalization_payload(normalization)
        )
        results = (
            train_transition_g(train_records, normalization, seed, updates=updates, device=device),
            train_physical_d(train_records, normalization, seed, updates=updates, device=device),
            train_direct_c(train_records, normalization, seed, updates=updates, device=device),
        )
        paths = [normalization_path]
        for head, result in zip(("G", "D", "C"), results, strict=True):
            path = save_terminal_checkpoint(
                arm / f"{head}.pt", result.module, kind=f"{kind}-{head}",
                training_seed=seed, terminal_update=updates, metadata=_metadata(),
            )
            head_paths[f"{name}.{head}"] = path
            paths.append(path)
        result_sets.append(TrainingArtifactSet(kind, seed, tuple(paths)))
    return head_paths, tuple(result_sets)


def run_training(root: str | Path, output: str | Path, *, updates: int = 1_000,
                 device: torch.device | str = "cpu") -> tuple[TrainingArtifactSet, ...]:
    """Complete every causal export before materializing any privileged truth."""
    if updates != 1_000:
        raise ValueError("production ATTR-RTG training requires exactly 1000 updates")
    root, output = Path(root), Path(output)
    data = _verified_allowed_data(root, output)
    for kind in ("asm", "transformer"):
        projection = make_registered_projection(kind)
        _atomic_create(output / f"projection_{kind}.pt",
            lambda stream, key=kind, value=projection: torch.save({"kind": key, "projection": value}, stream))
    models, backbone_paths = _train_all_backbones(root, output, data[0], updates, device)
    exports, input_state_paths, truths = _causal_export_then_truth(
        data, models, output
    )
    truth_path = write_truth_manifest(output / "truth_manifest.json", truths)
    records = _attach_all(exports, truths)
    state_paths = _save_all_states(output, records)
    head_paths, results = _train_all_heads(output, records, updates, device)
    write_training_manifest(
        output / "training_manifest.json", allowed_manifest=output / "allowed_manifest.json",
        truth_manifest=truth_path, backbone_checkpoints=backbone_paths,
        head_checkpoints=head_paths, state_records=input_state_paths | state_paths,
    )
    return results


