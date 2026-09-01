"""Immutable value types and action constants for HazardWorld."""

from __future__ import annotations
from dataclasses import dataclass

Position = tuple[int, int]
Action = str
ACTIONS: tuple[Action, ...] = ("U", "D", "L", "R", "BRAKE", "RECOVER", "STOP")
DIRECTION: dict[Action, Position] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
MODES = ("safe", "degraded", "unstable")


def _inside(position: Position, size: int) -> bool:
    return 0 <= position[0] < size and 0 <= position[1] < size


@dataclass(frozen=True)
class HazardWorldConfig:
    world_id: str
    seed: int
    grid_size: int = 9
    max_steps: int = 64
    goal: Position = (8, 8)
    walls: tuple[Position, ...] = ()
    traps: tuple[Position, ...] = ()
    moving_hazards: tuple[Position, ...] = ((4, 4),)
    hazard_velocities: tuple[Position, ...] = ((0, 1),)
    initial_agent: Position = (0, 0)
    initial_energy: float = 1.0
    failure_threshold: float = 0.12
    failure_delay: int = 3
    recovery_window: int = 3
    sensor_radius: int = 2
    sensor_noise: float = 0.04
    forcing: float = 0.10
    dynamic_family: str = "baseline"

    def __post_init__(self) -> None:
        if self.grid_size < 3 or self.failure_delay < 1 or self.recovery_window < 1:
            raise ValueError("invalid HazardWorld dimensions or timing")
        if len(self.moving_hazards) != len(self.hazard_velocities):
            raise ValueError("each moving hazard requires a velocity")
        occupied = {
            self.initial_agent,
            self.goal,
            *self.walls,
            *self.traps,
            *self.moving_hazards,
        }
        if any(not _inside(cell, self.grid_size) for cell in occupied):
            raise ValueError("world positions must be inside the grid")


@dataclass(frozen=True)
class HazardWorldState:
    agent: Position
    velocity: Position
    hazards: tuple[Position, ...]
    hazard_velocities: tuple[Position, ...]
    energy: float
    low_energy_steps: int = 0
    recovery_left: int = 0
    hidden_mode: str = "safe"
    step: int = 0
    unsafe: bool = False
    terminal: bool = False


@dataclass(frozen=True)
class HazardObservation:
    agent: Position
    velocity: Position
    energy_sensor: float
    goal_delta: Position
    local_walls: tuple[Position, ...]
    local_traps: tuple[Position, ...]
    local_hazards: tuple[Position, ...]
    step: int


@dataclass(frozen=True)
class HazardTransition:
    state: HazardWorldState
    observation: HazardObservation
    reward: float
    action_cost: float
    unsafe: bool
    done: bool
    severity: float
