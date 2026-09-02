"""Post-export truth audits and immutable training manifests for ATTR-RTG."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .rtg_artifacts import (
    atomic_write_json,
    canonical_sha256,
    digest_files,
    digest_payload,
)
from .rtg_physical_targets import unsafe_predicate
from .rtg_types import RtgOrigin


def origin_identity(origin: RtgOrigin) -> tuple[str, str, str, int]:
    item = origin.metadata
    return item.split_id, item.world_id, item.episode_id, item.t


def audit_materialized_truths(
    groups: Mapping[str, Sequence[RtgOrigin]],
) -> dict[str, object]:
    """Audit split disjunction, class minima, and canonical truth hashes."""
    if tuple(groups) != ("train", "validation", "calibration"):
        raise ValueError("truth groups must use the three allowed splits in order")
    seen_origins: set[tuple[str, str, str, int]] = set()
    seen_candidates: set[tuple[str, str, str, int, int]] = set()
    splits = []
    for split, origins in groups.items():
        identities = [origin_identity(origin) for origin in origins]
        if len(set(identities)) != len(identities) or seen_origins.intersection(identities):
            raise ValueError("truth origin identities overlap")
        seen_origins.update(identities)
        rows, positives = [], 0
        for origin in origins:
            if origin.metadata.split_id != split:
                raise ValueError("truth split identity differs")
            candidates = []
            for action_index, truth in enumerate(origin.truth.candidates):
                identity = (*origin_identity(origin), action_index)
                if identity in seen_candidates:
                    raise ValueError("truth candidate identity overlaps")
                seen_candidates.add(identity)
                if unsafe_predicate(truth.target, origin.truth.failure_delay) != truth.unsafe:
                    raise ValueError("materialized truth differs from unsafe predicate")
                positives += int(truth.unsafe)
                candidates.append({
                    "action_index": action_index,
                    "target_sha256": canonical_sha256(list(truth.target)),
                    "unsafe": truth.unsafe,
                })
            rows.append({
                "identity": list(origin_identity(origin)),
                "persistence_sha256": canonical_sha256(
                    list(origin.truth.persistence_target)
                ),
                "failure_delay": origin.truth.failure_delay,
                "candidates": candidates,
            })
        candidate_count = len(origins) * 6
        minimum = 500 if split == "train" else 100
        if len(origins) < minimum or positives < 25 or candidate_count - positives < 25:
            raise ValueError("materialized truth fails frozen origin or class minima")
        splits.append({
            "name": split,
            "origin_count": len(origins),
            "candidate_count": candidate_count,
            "positive_labels": positives,
            "negative_labels": candidate_count - positives,
            "truth_sha256": canonical_sha256(rows),
        })
    body: dict[str, object] = {
        "schema": "ATTR-RTG-TRUTH-MANIFEST-V1",
        "splits": splits,
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def write_truth_manifest(
    path: str | Path, groups: Mapping[str, Sequence[RtgOrigin]]
) -> Path:
    return atomic_write_json(path, audit_materialized_truths(groups))


def validate_arm_input_equivalence(
    arm_records: Mapping[str, Mapping[str, Sequence[object]]],
) -> None:
    """Require identical candidate identity order across all ten exported arms."""
    if len(arm_records) != 10:
        raise ValueError("exactly ten backbone export arms are required")
    reference: dict[str, tuple[tuple[object, ...], ...]] | None = None
    for splits in arm_records.values():
        current = {
            split: tuple(
                (record.split_id, record.world_id, record.episode_id,
                 record.t, record.action_index)
                for record in records
            )
            for split, records in splits.items()
        }
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError("candidate export identities differ across arms")


def write_training_manifest(
    path: str | Path,
    *,
    allowed_manifest: str | Path,
    truth_manifest: str | Path,
    backbone_checkpoints: Mapping[str, str | Path],
    head_checkpoints: Mapping[str, str | Path],
    state_records: Mapping[str, str | Path],
) -> Path:
    artifacts = {
        "allowed_manifest": allowed_manifest,
        "truth_manifest": truth_manifest,
        **{f"backbone.{key}": value for key, value in backbone_checkpoints.items()},
        **{f"head.{key}": value for key, value in head_checkpoints.items()},
        **{f"states.{key}": value for key, value in state_records.items()},
    }
    body: dict[str, object] = {
        "schema": "ATTR-RTG-TRAINING-MANIFEST-V1",
        "artifacts": digest_payload(digest_files(artifacts)),
    }
    body["content_sha256"] = canonical_sha256(body)
    return atomic_write_json(path, body)
