"""Reproducible partially observed safety environment for ATTR."""

from __future__ import annotations
import hashlib
import random
from typing import Sequence
from .hazard_world_io import (
    assert_no_world_leakage,
    config_from_json,
    serialize_config,
    serialize_state,
    split_worlds,
    state_from_json,
)
from .hazard_world_types import (
    ACTIONS,
    DIRECTION,
    MODES,
    Action,
    HazardObservation,
    HazardTransition,
    HazardWorldConfig,
    HazardWorldState,
    Position,
)


class HazardWorld:
    """Stateful adapter; exogenous noise is keyed by (seed, step, channel)."""

    def __init__(
        self, config: HazardWorldConfig, state: HazardWorldState | None = None
    ):
        self.config = config
        self.state = state or HazardWorldState(
            agent=config.initial_agent,
            velocity=(0, 0),
            hazards=config.moving_hazards,
            hazard_velocities=config.hazard_velocities,
            energy=config.initial_energy,
        )

    def clone(self) -> "HazardWorld":
        """Clone state while preserving common-random exogenous events."""
        return HazardWorld(self.config, self.state)

    def reset(self) -> HazardObservation:
        self.state = HazardWorld(self.config).state
        return self.observe()

    def observe(self) -> HazardObservation:
        return observe(self.config, self.state)

    def step(self, action: Action) -> HazardTransition:
        item = transition(self.config, self.state, action)
        self.state = item.state
        return item

    def intervene(self, actions: Sequence[Action]) -> list[HazardTransition]:
        branch = self.clone()
        return [branch.step(action) for action in actions if not branch.state.terminal]


def transition(
    config: HazardWorldConfig, state: HazardWorldState, action: Action
) -> HazardTransition:
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    if state.terminal:
        raise RuntimeError("cannot step a terminal HazardWorld")
    mode = _next_mode(config, state)
    velocity = _next_velocity(state.velocity, action)
    candidate = (state.agent[0] + velocity[0], state.agent[1] + velocity[1])
    agent = (
        candidate
        if _inside(candidate, config.grid_size) and candidate not in config.walls
        else state.agent
    )
    hazards, hazard_velocities = _move_hazards(config, state, mode)
    cost = {"BRAKE": 0.03, "RECOVER": 0.16, "STOP": 0.01}.get(action, 0.06)
    drain = cost + {"safe": 0.0, "degraded": 0.025, "unstable": 0.06}[mode]
    drain += config.forcing * _noise(config.seed, state.step, "forcing")
    energy = min(1.0, state.energy - drain + (0.25 if action == "RECOVER" else 0.0))
    low_steps = state.low_energy_steps + 1 if energy <= config.failure_threshold else 0
    recovery_left = (
        config.recovery_window
        if action == "RECOVER"
        else max(0, state.recovery_left - 1)
    )
    collision = agent in config.traps or agent in hazards
    delayed_failure = low_steps >= config.failure_delay and recovery_left == 0
    unsafe = state.unsafe or collision or delayed_failure
    step = state.step + 1
    success = agent == config.goal and not unsafe
    terminal = unsafe or success or action == "STOP" or step >= config.max_steps
    severity = (
        1.0
        if collision
        else (min(1.0, low_steps / config.failure_delay) if delayed_failure else 0.0)
    )
    next_state = HazardWorldState(
        agent,
        velocity,
        hazards,
        hazard_velocities,
        round(energy, 8),
        low_steps,
        recovery_left,
        mode,
        step,
        unsafe,
        terminal,
    )
    reward = (1.0 if success else 0.0) - cost - severity
    return HazardTransition(
        next_state,
        observe(config, next_state),
        reward,
        cost,
        unsafe,
        terminal,
        severity,
    )


def observe(config: HazardWorldConfig, state: HazardWorldState) -> HazardObservation:
    radius = config.sensor_radius
    local = lambda cells: tuple(
        sorted(cell for cell in cells if _distance(state.agent, cell) <= radius)
    )
    sensor = state.energy + config.sensor_noise * (
        2.0 * _noise(config.seed, state.step, "sensor") - 1.0
    )
    return HazardObservation(
        state.agent,
        state.velocity,
        round(max(0.0, min(1.0, sensor)), 6),
        (config.goal[0] - state.agent[0], config.goal[1] - state.agent[1]),
        local(config.walls),
        local(config.traps),
        local(state.hazards),
        state.step,
    )


def is_safe(config: HazardWorldConfig, state: HazardWorldState) -> bool:
    """Frozen safe-set predicate, based on physical state rather than sensors."""
    return (
        not state.unsafe
        and state.agent not in config.traps
        and state.agent not in state.hazards
        and state.low_energy_steps < config.failure_delay
    )


def _next_velocity(current: Position, action: Action) -> Position:
    if action in DIRECTION:
        dr, dc = DIRECTION[action]
        return (max(-1, min(1, current[0] + dr)), max(-1, min(1, current[1] + dc)))
    if action in {"BRAKE", "STOP"}:
        return (0, 0)
    return current


def _move_hazards(
    config: HazardWorldConfig, state: HazardWorldState, mode: str
) -> tuple[tuple[Position, ...], tuple[Position, ...]]:
    moved, velocities = [], []
    for index, (position, velocity) in enumerate(
        zip(state.hazards, state.hazard_velocities)
    ):
        jitter = (
            1
            if mode == "unstable"
            and _noise(config.seed, state.step, f"hazard:{index}") < 0.35
            else 0
        )
        proposed = (
            position[0] + velocity[0] + jitter,
            position[1] + velocity[1] - jitter,
        )
        if not _inside(proposed, config.grid_size) or proposed in config.walls:
            velocity = (-velocity[0], -velocity[1])
            proposed = (position[0] + velocity[0], position[1] + velocity[1])
        moved.append(proposed if _inside(proposed, config.grid_size) else position)
        velocities.append(velocity)
    return tuple(moved), tuple(velocities)


def _next_mode(config: HazardWorldConfig, state: HazardWorldState) -> str:
    value = _noise(config.seed, state.step, "mode")
    if state.hidden_mode == "safe":
        return "degraded" if value < 0.04 + config.forcing * 0.1 else "safe"
    if state.hidden_mode == "degraded":
        return "unstable" if value < 0.12 else ("safe" if value > 0.92 else "degraded")
    return "degraded" if value > 0.85 else "unstable"


def _noise(seed: int, step: int, channel: str) -> float:
    digest = hashlib.sha256(f"{seed}:{step}:{channel}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).random()


def _inside(position: Position, size: int) -> bool:
    return 0 <= position[0] < size and 0 <= position[1] < size


def _distance(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
