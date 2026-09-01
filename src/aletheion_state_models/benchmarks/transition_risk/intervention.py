"""Utilities for paired interventions on cloned simulator states."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BranchRollout:
    actions: tuple[Any, ...]
    trajectory: tuple[Any, ...]

    @property
    def final_state(self) -> Any:
        return self.trajectory[-1] if self.trajectory else None


@dataclass(frozen=True)
class ClonedIntervention:
    intervention: BranchRollout
    control: BranchRollout


def clone_simulator(simulator: Any) -> Any:
    """Clone through the simulator contract, falling back to ``deepcopy``."""
    clone = getattr(simulator, "clone", None)
    return clone() if callable(clone) else copy.deepcopy(simulator)


def _default_step(simulator: Any, action: Any, noise: Any) -> Any:
    try:
        result = simulator.step(action, noise=noise)
    except TypeError:
        try:
            result = simulator.step(action, noise)
        except TypeError:
            result = simulator.step(action)
    # Store the full return value: callers can preserve observations, reward,
    # termination, and diagnostics without this utility knowing their schema.
    return copy.deepcopy(result)


def rollout_branch(
    simulator: Any,
    actions: Sequence[Any],
    future_noise: Sequence[Any],
    step_fn: Callable[[Any, Any, Any], Any] | None = None,
) -> BranchRollout:
    """Run an already-cloned branch with an explicit exogenous-noise tape."""
    if len(actions) != len(future_noise):
        raise ValueError("actions and future_noise must have equal length")
    advance = step_fn or _default_step
    trajectory = tuple(
        advance(simulator, action, copy.deepcopy(noise))
        for action, noise in zip(actions, future_noise)
    )
    return BranchRollout(tuple(actions), trajectory)


def run_cloned_intervention(
    simulator: Any,
    intervention_action: Any,
    control_action: Any,
    future_noise: Iterable[Any],
    *,
    continuation_action: Any | None = None,
    step_fn: Callable[[Any, Any, Any], Any] | None = None,
) -> ClonedIntervention:
    """Compare ``do(action)`` with ``do(control)`` under identical future noise.

    Only the first action differs. Later steps use ``continuation_action`` when
    provided, otherwise ``control_action``. The input simulator is never stepped.
    """
    noise = tuple(copy.deepcopy(tuple(future_noise)))
    if not noise:
        raise ValueError("future_noise must contain at least one step")
    continuation = (
        control_action if continuation_action is None else continuation_action
    )
    intervention_actions = (intervention_action,) + (continuation,) * (len(noise) - 1)
    control_actions = (control_action,) * len(noise)
    treated_simulator = clone_simulator(simulator)
    control_simulator = clone_simulator(simulator)
    treated = rollout_branch(treated_simulator, intervention_actions, noise, step_fn)
    control = rollout_branch(control_simulator, control_actions, noise, step_fn)
    return ClonedIntervention(treated, control)


# Short alias for orchestration code.
cloned_intervention = run_cloned_intervention
