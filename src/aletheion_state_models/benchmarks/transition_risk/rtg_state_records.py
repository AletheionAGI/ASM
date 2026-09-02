"""Causal projected state export and candidate training records for ATTR-RTG."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rtg_projection import project_state
from .rtg_state_export import CausalStateExporter
from .rtg_types import PreparedRtgOrigin, RtgOrigin


@dataclass(frozen=True)
class CandidateStateInputRecord:
    """Projected candidate inputs with identity, but no privileged truth."""

    split_id: str
    world_id: str
    episode_id: str
    t: int
    action_index: int
    pre_state: torch.Tensor
    next_state: torch.Tensor
    fixed_frame: torch.Tensor

    def __post_init__(self) -> None:
        if not self.split_id or not self.world_id or not self.episode_id or self.t < 1:
            raise ValueError("candidate record identity is invalid")
        if self.action_index not in range(6):
            raise ValueError("candidate action index must be in 0..5")
        if self.pre_state.shape != (28,) or self.next_state.shape != (28,):
            raise ValueError("candidate states must be projected to 28 dimensions")
        if self.fixed_frame.shape != (32,):
            raise ValueError("candidate frame has invalid shape")
        tensors = (self.pre_state, self.next_state, self.fixed_frame)
        if any(value.dtype != torch.float32 or value.requires_grad for value in tensors):
            raise ValueError("record tensors must be detached float32")
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("record tensors must be finite")


@dataclass(frozen=True)
class CandidateStateRecord(CandidateStateInputRecord):
    """Candidate inputs with privileged training truth attached."""

    physical_target: tuple[int, ...]
    persistence_target: tuple[int, ...]
    unsafe: bool
    failure_delay: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.physical_target) != 11 or len(self.persistence_target) != 11:
            raise ValueError("candidate physical target has invalid shape")
        if self.failure_delay not in {1, 3}:
            raise ValueError("candidate failure_delay must be 1 or 3")


def _record_identity(record: CandidateStateInputRecord) -> tuple[str, str, str, int, int]:
    return (
        record.split_id,
        record.world_id,
        record.episode_id,
        record.t,
        record.action_index,
    )


def _origin_identity(origin: RtgOrigin) -> tuple[str, str, str, int]:
    metadata = origin.metadata
    return metadata.split_id, metadata.world_id, metadata.episode_id, metadata.t


def _export_input_records(
    origins: tuple[PreparedRtgOrigin, ...],
    exporter: CausalStateExporter,
    projection: torch.Tensor,
) -> tuple[CandidateStateInputRecord, ...]:
    records: list[CandidateStateInputRecord] = []
    for origin in origins:
        history = torch.tensor(origin.inputs.history, dtype=torch.long).unsqueeze(0)
        pre_raw = exporter.export(history)
        pre = project_state(pre_raw, projection).squeeze(0).detach().cpu().float()
        for action_index, candidate in enumerate(origin.inputs.candidates):
            frame = torch.tensor(candidate.frame, dtype=torch.long).unsqueeze(0)
            post_raw = exporter.export(torch.cat((history, frame), dim=1))
            nxt = project_state(post_raw, projection).squeeze(0).detach().cpu().float()
            metadata = origin.metadata
            records.append(
                CandidateStateInputRecord(
                    metadata.split_id,
                    metadata.world_id,
                    metadata.episode_id,
                    metadata.t,
                    action_index,
                    pre.clone(),
                    nxt,
                    candidate.fixed_frame.detach().cpu().clone().float(),
                )
            )
    records.sort(key=_record_identity)
    identities = [_record_identity(record) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate input record identities must be unique")
    return tuple(records)


def export_candidate_state_inputs(
    origins: tuple[PreparedRtgOrigin, ...],
    exporter: CausalStateExporter,
    projection: torch.Tensor,
) -> tuple[CandidateStateInputRecord, ...]:
    """Export causal states from prepared origins before truth materialization."""
    return _export_input_records(origins, exporter, projection)


def attach_candidate_truths(
    records: tuple[CandidateStateInputRecord, ...],
    origins: tuple[RtgOrigin, ...],
) -> tuple[CandidateStateRecord, ...]:
    """Attach privileged truths by stable origin and candidate identity."""
    truth_by_identity = {}
    for origin in origins:
        origin_key = _origin_identity(origin)
        for action_index, truth in enumerate(origin.truth.candidates):
            key = (*origin_key, action_index)
            if key in truth_by_identity:
                raise ValueError("candidate truth identities must be unique")
            truth_by_identity[key] = (truth, origin.truth)

    record_identities = {_record_identity(record) for record in records}
    if len(record_identities) != len(records):
        raise ValueError("candidate input record identities must be unique")
    truth_identities = set(truth_by_identity)
    if record_identities != truth_identities:
        raise ValueError("candidate input and truth identities do not match")

    attached = []
    for record in records:
        truth, origin_truth = truth_by_identity[_record_identity(record)]
        attached.append(
            CandidateStateRecord(
                record.split_id,
                record.world_id,
                record.episode_id,
                record.t,
                record.action_index,
                record.pre_state,
                record.next_state,
                record.fixed_frame,
                truth.target,
                origin_truth.persistence_target,
                truth.unsafe,
                origin_truth.failure_delay,
            )
        )
    attached.sort(key=_record_identity)
    return tuple(attached)


def stack_record_states(
    records: tuple[CandidateStateRecord, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("state records must not be empty")
    return torch.stack([item.pre_state for item in records]), torch.stack(
        [item.next_state for item in records]
    )
