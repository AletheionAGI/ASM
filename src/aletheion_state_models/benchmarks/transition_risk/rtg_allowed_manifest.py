"""Canonical pre-test manifests for the three allowed ATTR-RTG splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from world_model.hazard_world_types import HazardWorldConfig

from .rtg_artifacts import atomic_write_json, canonical_sha256, file_sha256
from .rtg_batching import make_batch_plan
from .rtg_calibration import partition_calibration_worlds
from .rtg_config import TRAINING_SEEDS, registered_config_paths, verify_preregistration
from .rtg_dataset import prepare_rtg_input_origins
from .rtg_splits import assert_disjoint_world_ids, make_allowed_worlds, split_spec
from .rtg_training_data import (
    BehavioralEpisode,
    collate_ce_episodes,
    make_behavioral_episodes,
)
from .rtg_types import OriginMetadata, PreparedRtgOrigin

ALLOWED_SPLITS = ("train", "validation", "calibration")


@dataclass(frozen=True)
class AllowedSplitData:
    name: str
    worlds: tuple[HazardWorldConfig, ...]
    episodes: tuple[BehavioralEpisode, ...]
    origins: tuple[PreparedRtgOrigin, ...]


def prepare_allowed_data() -> tuple[AllowedSplitData, ...]:
    """Materialize only allowed worlds through the capability-safe registry."""
    worlds_by_split = tuple(make_allowed_worlds(name) for name in ALLOWED_SPLITS)
    assert_disjoint_world_ids(worlds_by_split)
    prepared = []
    for name, worlds in zip(ALLOWED_SPLITS, worlds_by_split, strict=True):
        spec = split_spec(name)
        episodes = make_behavioral_episodes(
            worlds, episodes_per_world=spec.episodes_per_world,
            split_id=name, split_seed=spec.split_seed,
        )
        origins = prepare_rtg_input_origins(
            worlds, episodes_per_world=spec.episodes_per_world,
            split_id=name, split_seed=spec.split_seed,
        )
        prepared.append(AllowedSplitData(name, worlds, episodes, origins))
    return tuple(prepared)


def _episode_payload(episode: BehavioralEpisode) -> dict[str, object]:
    batch = collate_ce_episodes((episode,), sequence_length=64)
    targets = batch.targets[0]
    content: dict[str, object] = {
        "logical_length": len(episode.tokens),
        "offsets": [0, len(episode.tokens)],
        "input_ids": batch.input_ids[0].tolist(),
        "targets": targets.tolist(),
        "target_mask": (targets != -100).tolist(),
    }
    return {"episode_id": episode.episode_id, **content,
            "episode_sha256": canonical_sha256(content)}


def _metadata_identity(metadata: OriginMetadata) -> tuple[str, str, str, int]:
    return (
        metadata.split_id,
        metadata.world_id,
        metadata.episode_id,
        metadata.t,
    )


def _config_content(origin: PreparedRtgOrigin) -> dict[str, object]:
    values = asdict(origin.snapshot.config)
    for field in ("world_id", "seed", "namespace"):
        values.pop(field, None)
    return values


def _origin_content(origin: PreparedRtgOrigin) -> dict[str, object]:
    return {
        "config": _config_content(origin),
        "state": asdict(origin.snapshot.state),
        "history": list(origin.inputs.history),
        "candidates": [
            {"action_index": index, "frame": list(candidate.frame),
             "fixed_frame": candidate.fixed_frame.tolist()}
            for index, candidate in enumerate(origin.inputs.candidates)
        ],
    }


def _candidate_payload(origin: PreparedRtgOrigin, action_index: int) -> dict[str, object]:
    candidate = origin.inputs.candidates[action_index]
    content = {
        "origin": _origin_content(origin),
        "action_index": action_index,
        "frame": list(candidate.frame),
        "fixed_frame": candidate.fixed_frame.tolist(),
    }
    return {
        "action_index": action_index,
        "frame": content["frame"],
        "fixed_frame": content["fixed_frame"],
        "candidate_input_sha256": canonical_sha256(content),
    }

def _origin_payload(origin: PreparedRtgOrigin) -> dict[str, object]:
    candidates = [
        _candidate_payload(origin, action_index)
        for action_index in range(len(origin.inputs.candidates))
    ]
    identity_and_input = {
        "identity": list(_metadata_identity(origin.metadata)),
        "history": list(origin.inputs.history),
        "candidates": [
            {
                "action_index": candidate["action_index"],
                "frame": candidate["frame"],
                "fixed_frame": candidate["fixed_frame"],
            }
            for candidate in candidates
        ],
    }
    return {
        "metadata": asdict(origin.metadata),
        "history": identity_and_input["history"],
        "candidates": candidates,
        "config_sha256": canonical_sha256(_config_content(origin)),
        "state_sha256": canonical_sha256(asdict(origin.snapshot.state)),
        "origin_input_sha256": canonical_sha256(_origin_content(origin)),
    }


def _audit_split(item: AllowedSplitData) -> dict[str, object]:
    spec = split_spec(item.name)
    expected_episode_count = spec.world_count * spec.episodes_per_world
    if len(item.worlds) != spec.world_count or len(item.episodes) != expected_episode_count:
        raise ValueError("allowed split world or episode count differs")
    expected_order = sorted(
        item.origins,
        key=lambda origin: (
            origin.metadata.world_id,
            origin.metadata.episode_id,
            origin.metadata.t,
        ),
    )
    if list(item.origins) != expected_order:
        raise ValueError("allowed origins are not in lexical order")
    episodes = {episode.episode_id: episode for episode in item.episodes}
    for origin in item.origins:
        episode = episodes.get(origin.metadata.episode_id)
        if episode is None or origin.inputs.history != episode.tokens[: 4 * origin.metadata.t]:
            raise ValueError("episode and causal origin history differ")
        if len(origin.inputs.candidates) != 6:
            raise ValueError("allowed origin does not contain six candidates")
    minimum = 500 if item.name == "train" else 100
    if len(item.origins) < minimum:
        raise ValueError("allowed split fails frozen origin minimum")
    origin_episodes = {origin.metadata.episode_id for origin in item.origins}
    return {
        "world_count": len(item.worlds),
        "episode_count": len(item.episodes),
        "origin_count": len(item.origins),
        "candidate_count": len(item.origins) * 6,
        "zero_origin_episode_ids": [
            episode.episode_id
            for episode in item.episodes
            if episode.episode_id not in origin_episodes
        ],
    }


def _assert_pairwise_disjoint(groups: tuple[set[object], ...], label: str) -> None:
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            if not left.isdisjoint(right):
                raise ValueError(f"allowed split {label} are not disjoint")


def _audit_disjointness(data: tuple[AllowedSplitData, ...]) -> dict[str, bool]:
    payloads = tuple((
        [_episode_payload(episode) for episode in item.episodes],
        [_origin_payload(origin) for origin in item.origins],
    ) for item in data)
    named_groups = {
        "config_sha256": tuple({row["config_sha256"] for row in origins}
                               for _, origins in payloads),
        "episode_sha256": tuple({row["episode_sha256"] for row in episodes}
                                for episodes, _ in payloads),
        "origin_input_sha256": tuple({row["origin_input_sha256"] for row in origins}
                                     for _, origins in payloads),
        "candidate_input_sha256": tuple({candidate["candidate_input_sha256"]
            for row in origins for candidate in row["candidates"]}
            for _, origins in payloads),
    }
    for label, groups in named_groups.items():
        _assert_pairwise_disjoint(groups, label)
    return {label: True for label in named_groups}



def build_allowed_manifest(
    root: str | Path, data: tuple[AllowedSplitData, ...]
) -> dict[str, object]:
    """Describe configs, complete episodes, origins, and frozen train plans."""
    root = Path(root)
    preregistration_sha256 = verify_preregistration(root)
    if tuple(item.name for item in data) != ALLOWED_SPLITS:
        raise ValueError("allowed data must be train, validation, calibration in order")
    train = data[0]
    split_payload = []
    for item in data:
        spec = split_spec(item.name)
        split_payload.append({
            "name": item.name,
            "audit": _audit_split(item),
            "split_seed": spec.split_seed,
            "worlds": [
                {
                    "world_id": world.world_id,
                    "config_sha256": canonical_sha256({key: value for key, value in
                    asdict(world).items() if key not in {"world_id", "seed", "namespace"}}),
                }
                for world in item.worlds
            ],
            "episodes": [_episode_payload(episode) for episode in item.episodes],
            "origins": [_origin_payload(origin) for origin in item.origins],
        })
    plans = []
    for seed in TRAINING_SEEDS:
        for namespace, count in (
            ("backbone", len(train.episodes)),
            ("auxiliary", len(train.origins) * 6),
        ):
            plan = make_batch_plan(count, seed, namespace)
            plans.append(asdict(plan))
    configs = [
        {"path": str(path), "bytes": (root / path).stat().st_size,
         "sha256": file_sha256(root / path)}
        for path in registered_config_paths()
    ]
    body: dict[str, object] = {
        "schema": "ATTR-RTG-ALLOWED-MANIFEST-V1",
        "status": "allowed-data-materialized",
        "preregistration_sha256": preregistration_sha256,
        "configs": configs,
        "splits": split_payload,
        "disjointness_audit": _audit_disjointness(data),
        "batch_plans": plans,
        "calibration_partition": dict(zip(
            ("temperature_world_ids", "residual_world_ids"),
            partition_calibration_worlds([world.world_id for world in data[2].worlds]),
            strict=True,
        )),
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def write_allowed_manifest(
    path: str | Path, root: str | Path, data: tuple[AllowedSplitData, ...]
) -> Path:
    return atomic_write_json(path, build_allowed_manifest(root, data))
