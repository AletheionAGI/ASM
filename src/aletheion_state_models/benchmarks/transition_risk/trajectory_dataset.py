"""Causal behavior data with frozen cloned trajectory branches."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import fields

import torch

from world_model.hazard_world import HazardObservation, HazardWorld, HazardWorldConfig
from world_model.hazard_world_types import MODES

from .dataset import FRAME_WIDTH, encode_frame, make_worlds
from .trajectory_plans import encode_actions, frozen_action_plan, rollout_cloned_plan
from .trajectory_types import PLAN_HORIZON, TrajectoryEpisode, TrajectoryTargets

BehaviorPolicy = Callable[[HazardObservation, random.Random], str]
_BEHAVIOR_ACTIONS = ("U", "D", "L", "R", "BRAKE", "RECOVER")


def causal_behavior_policy(observation: HazardObservation, rng: random.Random) -> str:
    """Choose from the current observation only; no simulator state is accepted."""
    if observation.energy_sensor < 0.25 and rng.random() < 0.8:
        return "RECOVER"
    return rng.choice(_BEHAVIOR_ACTIONS[:-1])


def _cell(position: tuple[int, int], grid_size: int) -> int:
    return position[0] * grid_size + position[1]


def _target_row(state, config: HazardWorldConfig) -> dict[str, int]:
    safe_terminal = state.terminal and not state.unsafe
    return {
        "agent_cell": _cell(state.agent, config.grid_size),
        "moving_hazard_cell": _cell(state.hazards[0], config.grid_size),
        "velocity_row": state.velocity[0] + 1,
        "velocity_col": state.velocity[1] + 1,
        "energy_bin": min(63, max(0, int(state.energy * 64))),
        "low_energy_steps": min(3, state.low_energy_steps),
        "recovery_left": min(3, state.recovery_left),
        "hidden_mode": MODES.index(state.hidden_mode),
        "safe_terminal": int(safe_terminal),
    }


def _predicate_truth(state, config: HazardWorldConfig) -> bool:
    collision = state.agent in config.traps or state.agent in state.hazards
    delayed = (
        state.low_energy_steps >= config.failure_delay and state.recovery_left == 0
    )
    safe_terminal = state.terminal and not state.unsafe
    return (collision or delayed) and not safe_terminal


def rollout_trajectory_episode(
    config: HazardWorldConfig,
    episode_seed: int,
    *,
    policy: BehaviorPolicy = causal_behavior_policy,
    behavior_actions: Sequence[str] | None = None,
) -> TrajectoryEpisode:
    """Roll out behavior and supervise frozen plans from each causal origin."""
    if config.grid_size != 9:
        raise ValueError("ATTR-TG1 categorical cells require a 9x9 world")
    if len(config.traps) != 3 or len(config.moving_hazards) != 1:
        raise ValueError("ATTR-TG1 requires exactly three traps and one moving hazard")
    world = HazardWorld(config)
    rng = random.Random(episode_seed)
    episode_id = f"{config.world_id}:{episode_seed}"
    frames: list[int] = []
    chosen: list[str] = []
    plans: list[list[int]] = []
    branch_rows: list[list[dict[str, int]]] = []
    branch_truth: list[list[bool]] = []
    branch_valid: list[list[bool]] = []

    for origin in range(config.max_steps):
        observation = world.observe()
        action = (
            behavior_actions[origin]
            if behavior_actions is not None and origin < len(behavior_actions)
            else policy(observation, rng)
        )
        if action not in _BEHAVIOR_ACTIONS:
            raise ValueError("behavior actions must be valid and exclude STOP")
        plan = frozen_action_plan(action, episode_id, origin)
        transitions = rollout_cloned_plan(world, plan)
        rows = [_target_row(item.state, config) for item in transitions]
        truth = [_predicate_truth(item.state, config) for item in transitions]
        valid = [True] * len(rows)
        zero = {item.name: 0 for item in fields(TrajectoryTargets)}
        rows.extend(dict(zero) for _ in range(PLAN_HORIZON - len(rows)))
        truth.extend(False for _ in range(PLAN_HORIZON - len(truth)))
        valid.extend(False for _ in range(PLAN_HORIZON - len(valid)))

        frames.extend(encode_frame(observation, action, config.grid_size))
        chosen.append(action)
        plans.append(encode_actions(plan))
        branch_rows.append(rows)
        branch_truth.append(truth)
        branch_valid.append(valid)
        transition = world.step(action)
        if transition.done:
            break

    steps = len(chosen)
    target_tensors = {
        item.name: torch.tensor(
            [[row[item.name] for row in origin_rows] for origin_rows in branch_rows],
            dtype=torch.long,
        ).reshape(steps, PLAN_HORIZON)
        for item in fields(TrajectoryTargets)
    }
    traps = [_cell(cell, config.grid_size) for cell in config.traps]
    return TrajectoryEpisode(
        episode_id=episode_id,
        world_id=config.world_id,
        input_ids=torch.tensor(frames, dtype=torch.long),
        step_positions=torch.arange(FRAME_WIDTH - 1, len(frames), FRAME_WIDTH),
        plan_actions=torch.tensor(plans, dtype=torch.long),
        trap_cells=torch.tensor([traps] * steps, dtype=torch.long),
        targets=TrajectoryTargets(**target_tensors),
        valid_mask=torch.tensor(branch_valid, dtype=torch.bool),
        unsafe_truth=torch.tensor(branch_truth, dtype=torch.bool),
        behavior_actions=tuple(chosen),
        failure_delay=config.failure_delay,
    )


def make_trajectory_episodes(
    worlds: Sequence[HazardWorldConfig], episodes_per_world: int, seed: int
) -> list[TrajectoryEpisode]:
    """Build reproducible trajectory episodes for supplied world configs."""
    if episodes_per_world < 1:
        raise ValueError("episodes_per_world must be positive")
    return [
        rollout_trajectory_episode(
            world, seed * 1_000_003 + index * episodes_per_world + episode
        )
        for index, world in enumerate(worlds)
        for episode in range(episodes_per_world)
    ]


def collate_trajectory_episodes(
    episodes: Sequence[TrajectoryEpisode],
) -> dict[str, object]:
    """Pad behavior origins while preserving the fixed plan axis."""
    if not episodes:
        raise ValueError("cannot collate an empty episode list")
    batch = len(episodes)
    max_tokens = max(item.input_ids.numel() for item in episodes)
    max_steps = max(item.step_positions.numel() for item in episodes)
    output: dict[str, object] = {
        "input_ids": torch.zeros(batch, max_tokens, dtype=torch.long),
        "token_mask": torch.zeros(batch, max_tokens, dtype=torch.bool),
        "step_positions": torch.zeros(batch, max_steps, dtype=torch.long),
        "step_mask": torch.zeros(batch, max_steps, dtype=torch.bool),
        "plan_actions": torch.zeros(batch, max_steps, PLAN_HORIZON, dtype=torch.long),
        "trap_cells": torch.zeros(batch, max_steps, 3, dtype=torch.long),
        "valid_mask": torch.zeros(batch, max_steps, PLAN_HORIZON, dtype=torch.bool),
        "unsafe_truth": torch.zeros(batch, max_steps, PLAN_HORIZON, dtype=torch.bool),
    }
    target_batch = {
        item.name: torch.zeros(batch, max_steps, PLAN_HORIZON, dtype=torch.long)
        for item in fields(TrajectoryTargets)
    }
    output["targets"] = target_batch
    for row, episode in enumerate(episodes):
        nt, ns = episode.input_ids.numel(), episode.step_positions.numel()
        output["input_ids"][row, :nt] = episode.input_ids
        output["token_mask"][row, :nt] = True
        output["step_positions"][row, :ns] = episode.step_positions
        output["step_mask"][row, :ns] = True
        for name in ("plan_actions", "trap_cells", "valid_mask", "unsafe_truth"):
            output[name][row, :ns] = getattr(episode, name)
        for name, value in episode.targets.as_dict().items():
            target_batch[name][row, :ns] = value
    return output


# Short aliases for callers that already operate inside the trajectory namespace.
rollout_episode = rollout_trajectory_episode
make_episodes = make_trajectory_episodes
collate_episodes = collate_trajectory_episodes

__all__ = [
    "TrajectoryEpisode",
    "TrajectoryTargets",
    "causal_behavior_policy",
    "collate_episodes",
    "collate_trajectory_episodes",
    "make_episodes",
    "make_trajectory_episodes",
    "make_worlds",
    "rollout_episode",
    "rollout_trajectory_episode",
]
