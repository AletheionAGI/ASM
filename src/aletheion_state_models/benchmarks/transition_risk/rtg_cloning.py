"""Post-export one-step HazardWorld truth materialization for ATTR-RTG."""

from __future__ import annotations

from world_model.hazard_world import HazardWorld

from .rtg_physical_targets import audit_transition_target, encode_y_common
from .rtg_types import (
    NONSTOP_ACTIONS,
    CandidateTruth,
    OriginTruth,
    PreparedRtgOrigin,
    RtgOrigin,
)


def branch_candidate(origin: HazardWorld, action: str) -> CandidateTruth:
    """Execute one action on a fresh clone and retain its physical truth only."""
    if origin.state.terminal or origin.state.unsafe:
        raise ValueError("ATTR-RTG origins must be pre-terminal and safe")
    if action not in NONSTOP_ACTIONS:
        raise ValueError("candidate branch action must be non-STOP")
    branch = origin.clone()
    transition = branch.step(action)
    target = encode_y_common(origin.config, transition.state)
    audit_transition_target(target, transition, origin.config.failure_delay)
    return CandidateTruth(
        target=target, unsafe=transition.unsafe, transition=transition
    )


def materialize_origin_truth(prepared: PreparedRtgOrigin) -> RtgOrigin:
    """Materialize persistence and six truths only after caller completes export."""
    snapshot = prepared.snapshot
    origin = HazardWorld(snapshot.config, snapshot.state)
    initial_state = origin.state
    persistence = encode_y_common(origin.config, initial_state)
    candidates = tuple(branch_candidate(origin, action) for action in NONSTOP_ACTIONS)
    if origin.state != initial_state:
        raise RuntimeError("candidate branching mutated the restored origin")
    return RtgOrigin(
        metadata=prepared.metadata,
        inputs=prepared.inputs,
        truth=OriginTruth(
            persistence_target=persistence,
            candidates=candidates,
            failure_delay=origin.config.failure_delay,
        ),
    )


def branch_fallback(origin: HazardWorld) -> CandidateTruth:
    """Evaluate the registered ABSTAIN fallback on its own one-step clone."""
    return branch_candidate(origin, "BRAKE")
