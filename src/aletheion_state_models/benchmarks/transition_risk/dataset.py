"""Fixed-frame HazardWorld datasets shared by ATTR model families."""

from __future__ import annotations
from dataclasses import dataclass, replace
import random
import torch
from world_model.hazard_world import (
    ACTIONS,
    HazardObservation,
    HazardWorld,
    HazardWorldConfig,
)
from .labels import multi_horizon_labels
from .types import DEFAULT_HORIZONS

FRAME_WIDTH = 4
MODEL_FEATURES = (
    "agent",
    "velocity",
    "energy_sensor",
    "goal_delta",
    "local_hazard_distance",
    "action",
)
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}


@dataclass(frozen=True)
class HazardEpisode:
    episode_id: str
    world_id: str
    input_ids: torch.Tensor
    step_positions: torch.Tensor
    next_states: torch.Tensor
    hazard_labels: torch.Tensor
    severity: torch.Tensor
    time_to_hazard: torch.Tensor
    unsafe: torch.Tensor
    actions: tuple[str, ...]


def make_worlds(
    count: int, seed: int, *, dynamic_family: str = "baseline", max_steps: int = 16
) -> list[HazardWorldConfig]:
    """Create reproducible non-overlapping worlds; IDs never enter model input."""
    if count < 1 or max_steps < 2:
        raise ValueError("count and max_steps must be positive")
    rng = random.Random(seed)
    worlds = []
    for index in range(count):
        cells = [(r, c) for r in range(9) for c in range(9)]
        initial = rng.choice(cells)
        remaining = [cell for cell in cells if cell != initial]
        goal = rng.choice(remaining)
        remaining.remove(goal)
        traps = tuple(sorted(rng.sample(remaining, 3)))
        remaining = [cell for cell in remaining if cell not in traps]
        hazard = rng.choice(remaining)
        remaining.remove(hazard)
        walls = tuple(sorted(rng.sample(remaining, 5)))
        velocity = rng.choice(((0, 1), (1, 0), (0, -1), (-1, 0)))
        config = HazardWorldConfig(
            world_id=f"{dynamic_family}-{seed}-{index}",
            seed=seed * 1009 + index,
            goal=goal,
            walls=walls,
            traps=traps,
            moving_hazards=(hazard,),
            hazard_velocities=(velocity,),
            initial_agent=initial,
            max_steps=max_steps,
            dynamic_family=dynamic_family,
        )
        worlds.append(_apply_dynamic_family(config))
    return worlds


def _apply_dynamic_family(config: HazardWorldConfig) -> HazardWorldConfig:
    """Apply registered P2 shifts; baseline returns the original P1 config."""
    if config.dynamic_family == "baseline":
        return config
    if config.dynamic_family == "shift":
        return replace(config, sensor_noise=0.12, forcing=0.22)
    if config.dynamic_family == "ood":
        return replace(
            config,
            sensor_noise=0.18,
            forcing=0.16,
            failure_delay=1,
            recovery_window=2,
        )
    raise ValueError(f"unknown HazardWorld dynamic family: {config.dynamic_family}")


def _policy(observation: HazardObservation, rng: random.Random) -> str:
    if observation.energy_sensor < 0.25 and rng.random() < 0.8:
        return "RECOVER"
    return rng.choice(("U", "D", "L", "R", "BRAKE"))


def _nearest_observed_hazard(observation: HazardObservation) -> int:
    cells = observation.local_traps + observation.local_hazards
    if not cells:
        return 7
    return min(
        abs(observation.agent[0] - r) + abs(observation.agent[1] - c) for r, c in cells
    )


def encode_frame(
    observation: HazardObservation, action: str, grid_size: int = 9
) -> tuple[int, int, int, int]:
    """Encode only causally observed fields into four byte tokens."""
    position = observation.agent[0] * grid_size + observation.agent[1]
    velocity = (observation.velocity[0] + 1) * 3 + observation.velocity[1] + 1
    velocity_action = velocity * len(ACTIONS) + ACTION_INDEX[action]
    energy_bin = min(31, max(0, int(observation.energy_sensor * 32)))
    energy_hazard = energy_bin * 8 + min(7, _nearest_observed_hazard(observation))
    dr = min(7, max(-7, observation.goal_delta[0]))
    dc = min(7, max(-7, observation.goal_delta[1]))
    goal_delta = (dr + 7) * 15 + dc + 7
    values = (position, velocity_action, energy_hazard, goal_delta)
    if any(not 0 <= value < 256 for value in values):
        raise ValueError("encoded HazardWorld frame exceeds byte vocabulary")
    return values


def _state_target(transition, grid_size: int) -> list[float]:
    obs = transition.observation
    return [
        obs.agent[0] / (grid_size - 1),
        obs.agent[1] / (grid_size - 1),
        float(obs.velocity[0]),
        float(obs.velocity[1]),
        obs.energy_sensor,
        min(_nearest_observed_hazard(obs), 7) / 7,
    ]


def rollout_episode(
    config: HazardWorldConfig, episode_seed: int, *, horizons=DEFAULT_HORIZONS
) -> HazardEpisode:
    """Roll out one behavior-policy episode and derive future-only labels."""
    world = HazardWorld(config)
    rng = random.Random(episode_seed)
    frames = []
    targets = []
    severities = []
    actions = []
    state_unsafe = [world.state.unsafe]
    for _ in range(config.max_steps):
        observation = world.observe()
        action = _policy(observation, rng)
        frames.extend(encode_frame(observation, action, config.grid_size))
        item = world.step(action)
        actions.append(action)
        targets.append(_state_target(item, config.grid_size))
        severities.append(item.severity)
        state_unsafe.append(item.unsafe)
        if item.done:
            break
    rows = multi_horizon_labels(
        state_unsafe,
        horizons,
        [f"{config.world_id}:{episode_seed}"] * len(state_unsafe),
    )[: len(actions)]
    hazards = [[row[horizon] for horizon in horizons] for row in rows]
    first_unsafe = next(
        (index for index, value in enumerate(state_unsafe[1:]) if value), None
    )
    times = [
        float(first_unsafe - index + 1)
        if first_unsafe is not None and index <= first_unsafe
        else 0.0
        for index in range(len(actions))
    ]
    return HazardEpisode(
        f"{config.world_id}:{episode_seed}",
        config.world_id,
        torch.tensor(frames, dtype=torch.long),
        torch.arange(FRAME_WIDTH - 1, len(frames), FRAME_WIDTH),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(hazards, dtype=torch.float32),
        torch.tensor(severities, dtype=torch.float32),
        torch.tensor(times, dtype=torch.float32),
        torch.tensor(state_unsafe[1:], dtype=torch.bool),
        tuple(actions),
    )


def make_episodes(worlds, episodes_per_world: int, seed: int) -> list[HazardEpisode]:
    if episodes_per_world < 1:
        raise ValueError("episodes_per_world must be positive")
    return [
        rollout_episode(
            replace(world, seed=world.seed + episode),
            seed * 1_000_003 + index * episodes_per_world + episode,
        )
        for index, world in enumerate(worlds)
        for episode in range(episodes_per_world)
    ]


def collate_episodes(episodes: list[HazardEpisode]) -> dict[str, torch.Tensor]:
    if not episodes:
        raise ValueError("cannot collate an empty episode list")
    batch = len(episodes)
    max_tokens = max(item.input_ids.numel() for item in episodes)
    max_steps = max(item.step_positions.numel() for item in episodes)
    state_dim = episodes[0].next_states.shape[-1]
    horizons = episodes[0].hazard_labels.shape[-1]
    output = {
        "input_ids": torch.zeros(batch, max_tokens, dtype=torch.long),
        "token_mask": torch.zeros(batch, max_tokens, dtype=torch.bool),
        "step_positions": torch.zeros(batch, max_steps, dtype=torch.long),
        "step_mask": torch.zeros(batch, max_steps, dtype=torch.bool),
        "next_states": torch.zeros(batch, max_steps, state_dim),
        "hazard_labels": torch.zeros(batch, max_steps, horizons),
        "severity": torch.zeros(batch, max_steps),
        "time_to_hazard": torch.zeros(batch, max_steps),
    }
    for index, item in enumerate(episodes):
        nt = item.input_ids.numel()
        ns = item.step_positions.numel()
        output["input_ids"][index, :nt] = item.input_ids
        output["token_mask"][index, :nt] = True
        output["step_positions"][index, :ns] = item.step_positions
        output["step_mask"][index, :ns] = True
        output["next_states"][index, :ns] = item.next_states
        output["hazard_labels"][index, :ns] = item.hazard_labels
        output["severity"][index, :ns] = item.severity
        output["time_to_hazard"][index, :ns] = item.time_to_hazard
    return output


def gather_step_representations(
    representations: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    if (
        representations.ndim != 3
        or positions.ndim != 2
        or representations.shape[0] != positions.shape[0]
    ):
        raise ValueError("invalid representation/position shapes")
    return representations.gather(
        1, positions.unsqueeze(-1).expand(-1, -1, representations.shape[-1])
    )


__all__ = [
    "FRAME_WIDTH",
    "MODEL_FEATURES",
    "HazardEpisode",
    "collate_episodes",
    "encode_frame",
    "gather_step_representations",
    "make_episodes",
    "make_worlds",
    "rollout_episode",
]
