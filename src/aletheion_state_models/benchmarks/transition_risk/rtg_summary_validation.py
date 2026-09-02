"""Schema, ordering, and byte-pair validation for registered RTG summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .rtg_config import TRAINING_SEEDS
from .rtg_metrics_state import validate_six_candidate_clusters
from .rtg_pairing import canonical_records, require_byte_equivalent
from .rtg_pipeline_seal import BACKBONES

Record = Mapping[str, Any]
_SPLITS = ("test_id", "test_shift", "test_ood")
_HEADS = ("G", "C")



def validate_registered_evaluation(
    evaluation: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, tuple[Record, ...]]]]:
    if set(evaluation) != {"kind", "schema_version", "episodes_per_world", "splits"}:
        raise ValueError("registered evaluation top-level schema differs")
    if (
        evaluation["kind"] != "attr_rtg_registered_evaluation"
        or evaluation["schema_version"] != 1
        or evaluation["episodes_per_world"] != 4
    ):
        raise ValueError("registered evaluation metadata differs")
    if not isinstance(evaluation["splits"], Mapping) or set(
        evaluation["splits"]
    ) != set(_SPLITS):
        raise ValueError("registered evaluation requires exactly three test splits")
    output: dict[str, dict[str, dict[str, tuple[Record, ...]]]] = {}
    expected_systems = {
        f"{kind}.seed{seed}" for kind in BACKBONES for seed in TRAINING_SEEDS
    }
    for split in _SPLITS:
        payload = evaluation["splits"][split]
        if not isinstance(payload, Mapping) or set(payload) != {"systems"}:
            raise ValueError(f"{split} schema differs")
        systems = payload["systems"]
        if not isinstance(systems, Mapping) or set(systems) != expected_systems:
            raise ValueError(f"{split} requires the complete registered system matrix")
        output[split] = {kind: {head: () for head in _HEADS} for kind in BACKBONES}
        references: dict[tuple[int, str], tuple[Record, ...]] = {}
        for kind in BACKBONES:
            for seed in TRAINING_SEEDS:
                system = systems[f"{kind}.seed{seed}"]
                if not isinstance(system, Mapping) or set(system) != set(_HEADS):
                    raise ValueError("registered system schema differs")
                batches: dict[str, tuple[Record, ...]] = {}
                for head in _HEADS:
                    batch = system[head]
                    if not isinstance(batch, Mapping) or set(batch) != {"records"}:
                        raise ValueError(
                            "registered batch schema differs or contains stale metrics"
                        )
                    rows = canonical_records(tuple(batch["records"]))
                    validate_six_candidate_clusters(rows)
                    if any(
                        row.get("seed") != seed or row.get("split_id") != split
                        for row in rows
                    ):
                        raise ValueError("record seed or split identity differs")
                    reference = references.setdefault((seed, head), rows)
                    require_byte_equivalent(reference, rows)
                    batches[head] = rows
                    output[split][kind][head] += rows
                require_byte_equivalent(batches["G"], batches["C"])
        origins = {
            (row["world_id"], row["episode_id"], row["t"])
            for row in output[split][BACKBONES[0]]["G"]
            if row["seed"] == TRAINING_SEEDS[0]
        }
        labels = [
            row["candidate_unsafe"]
            for row in output[split][BACKBONES[0]]["G"]
            if row["seed"] == TRAINING_SEEDS[0]
        ]
        if any(
            type(value) not in (bool, int) or value not in (0, 1) for value in labels
        ):
            raise ValueError("candidate labels must be binary")
        positives = sum(int(value) for value in labels)
        if len(origins) < 200 or positives < 25 or len(labels) - positives < 25:
            raise ValueError(f"{split} lacks 200 origins or 25 labels of each class")
    return output

