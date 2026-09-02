"""Broker-owned origins/truth joined only after all arm scores freeze."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .constants import ARMS
from .official_data import Origin, TruthCache, materialize, truths_after_forward
from .scorer import (
    ScorerRequest,
    ScorerResponse,
    score_in_clean_process,
    serialize_message,
)


@dataclass(frozen=True)
class ScorerRefs:
    config_ref: str
    checkpoint_ref: str


@dataclass(frozen=True)
class FrozenArmScores:
    arm: str
    batches: tuple[ScorerResponse, ...]


@dataclass(frozen=True)
class BrokerEvaluation:
    scores: tuple[FrozenArmScores, ...]
    truth: tuple[tuple[float, ...], ...]
    identities: tuple[tuple[str, int, int, int], ...]


def freeze_all_arms_then_join_truth(
    origins: tuple[Origin, ...],
    refs: Mapping[str, ScorerRefs],
    *,
    device: str,
    batch_size: int,
    truth_cache: TruthCache,
) -> BrokerEvaluation:
    """Run R,CM,Z,T sequentially; only then materialize and join privileged H8."""
    if tuple(refs) != ARMS:
        raise ValueError("scorer references must be ordered R,CM,Z,T")
    batches = tuple(
        tuple(range(offset, min(offset + batch_size, len(origins))))
        for offset in range(0, len(origins), batch_size)
    )
    frozen = []
    for arm in ARMS:
        arm_refs = refs[arm]
        requests = []
        for indices in batches:
            batch = materialize(origins, list(indices), "cpu")
            requests.append(
                ScorerRequest(
                    arm,
                    arm_refs.config_ref,
                    arm_refs.checkpoint_ref,
                    serialize_message(batch["message"]),
                )
            )
        responses = score_in_clean_process(requests, device=device)
        frozen.append(FrozenArmScores(arm, responses))
    # This is deliberately below completion/join of every scorer arm.
    labels = truths_after_forward(list(origins), "cpu", truth_cache)
    truth = tuple(tuple(float(value) for value in row) for row in labels.tolist())
    return BrokerEvaluation(
        tuple(frozen), truth, tuple(origin.identity for origin in origins)
    )
