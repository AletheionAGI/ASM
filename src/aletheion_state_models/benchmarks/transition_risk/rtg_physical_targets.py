"""Frozen common physical target schema and unsafe predicate for ATTR-RTG."""

from __future__ import annotations

import math
from collections.abc import Sequence

from world_model.hazard_world_types import (
    HazardTransition,
    HazardWorldConfig,
    HazardWorldState,
)

from .rtg_types import Y_COMMON_CARDINALITIES, YCommon


def _cell(position: tuple[int, int]) -> int:
    return position[0] * 9 + position[1]


def validate_rtg_config(config: HazardWorldConfig) -> None:
    """Reject simulator configurations outside the frozen physical schema."""
    if config.grid_size != 9:
        raise ValueError("ATTR-RTG requires a 9x9 HazardWorld")
    if len(config.traps) != 3 or len(set(config.traps)) != 3:
        raise ValueError("ATTR-RTG requires exactly three distinct traps")
    if len(config.moving_hazards) != 1:
        raise ValueError("ATTR-RTG requires exactly one moving hazard")
    if config.failure_delay not in {1, 3}:
        raise ValueError("ATTR-RTG failure_delay must be 1 or 3")


def encode_y_common(config: HazardWorldConfig, state: HazardWorldState) -> YCommon:
    """Encode the eleven registered categorical physical consequence groups."""
    validate_rtg_config(config)
    traps = tuple(_cell(position) for position in sorted(config.traps))
    safe_terminal = int(state.terminal and not state.unsafe)
    values = (
        *traps,
        _cell(state.agent),
        _cell(state.hazards[0]),
        state.velocity[0] + 1,
        state.velocity[1] + 1,
        min(63, max(0, math.floor(64 * state.energy))),
        min(3, max(0, state.low_energy_steps)),
        min(3, max(0, state.recovery_left)),
        safe_terminal,
    )
    if unsafe_predicate(values, config.failure_delay) and safe_terminal:
        raise ValueError("safe terminal cannot encode an unsafe physical target")
    return values


def _valid_categories(values: Sequence[object]) -> bool:
    if len(values) != len(Y_COMMON_CARDINALITIES):
        return False
    for value, cardinality in zip(values, Y_COMMON_CARDINALITIES):
        if type(value) is not int or not 0 <= value < cardinality:
            return False
    return True


def unsafe_predicate(values: Sequence[object], failure_delay: int) -> bool:
    """Apply frozen P; malformed values and unsupported delays fail closed."""
    if failure_delay not in {1, 3} or not _valid_categories(values):
        return True
    trap_1, trap_2, trap_3 = values[:3]
    if len({trap_1, trap_2, trap_3}) != 3:
        return True
    agent = values[3]
    moving_hazard = values[4]
    low_energy_steps = values[8]
    recovery_left = values[9]
    safe_terminal = values[10]
    collision = agent in {trap_1, trap_2, trap_3} or agent == moving_hazard
    delayed_failure = low_energy_steps >= failure_delay and recovery_left == 0
    return bool(not safe_terminal and (collision or delayed_failure))


def audit_transition_target(
    target: YCommon,
    transition: HazardTransition,
    failure_delay: int,
) -> None:
    """Enforce the preregistered equality P(y_true) == transition.unsafe."""
    predicted = unsafe_predicate(target, failure_delay)
    if predicted != transition.unsafe:
        raise ValueError("P(y_common_true) does not match HazardTransition.unsafe")
