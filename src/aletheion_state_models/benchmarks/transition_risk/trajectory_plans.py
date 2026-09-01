"""Frozen causal action plans and cloned branch execution."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from world_model.hazard_world import ACTIONS, HazardWorld
from world_model.hazard_world_types import HazardTransition

from .trajectory_types import PLAN_HORIZON

PLAN_ACTIONS = tuple(action for action in ACTIONS if action != "STOP")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def frozen_action_plan(
    current_action: str,
    episode_id: str,
    origin: int,
    *,
    horizon: int = PLAN_HORIZON,
) -> tuple[str, ...]:
    """Make a plan without consulting observations or actions after ``origin``."""
    if current_action not in PLAN_ACTIONS:
        raise ValueError("the current action must be a non-STOP action")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    suffix = []
    for offset in range(1, horizon):
        digest = hashlib.sha256(
            f"ATTR-TG1:{episode_id}:{origin}:{offset}".encode()
        ).digest()
        suffix.append(
            PLAN_ACTIONS[int.from_bytes(digest[:8], "big") % len(PLAN_ACTIONS)]
        )
    return (current_action, *suffix)


def encode_actions(actions: Sequence[str]) -> list[int]:
    """Encode actions with the stable HazardWorld action vocabulary."""
    try:
        return [ACTION_TO_INDEX[action] for action in actions]
    except KeyError as error:
        raise ValueError(f"unknown action: {error.args[0]}") from error


def rollout_cloned_plan(
    origin: HazardWorld, actions: Sequence[str]
) -> tuple[HazardTransition, ...]:
    """Execute the supplied plan only on a clone of the origin simulator."""
    branch = origin.clone()
    transitions = []
    for action in actions:
        if branch.state.terminal:
            break
        transitions.append(branch.step(action))
    return tuple(transitions)


# Descriptive aliases retained for a small, discoverable public surface.
frozen_plan = frozen_action_plan
rollout_plan = rollout_cloned_plan
