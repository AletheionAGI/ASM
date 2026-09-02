"""Complete behavioral episodes and boundary-safe CE collation for ATTR-RTG."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from world_model.hazard_world import HazardWorld
from world_model.hazard_world_types import HazardWorldConfig

from .rtg_batching import make_batch_plan
from .rtg_dataset import episode_identifier
from .rtg_encoding import encode_candidate
from .rtg_physical_targets import validate_rtg_config
from .rtg_policy import choose_behavior_action


@dataclass(frozen=True, order=True)
class BehavioralEpisode:
    """One complete episode byte stream; episodes are never concatenated."""

    episode_id: str
    tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.episode_id or len(self.tokens) < 2:
            raise ValueError("behavioral episode must have an id and at least two bytes")
        if len(self.tokens) > 64 or any(type(token) is not int or not 0 <= token < 256 for token in self.tokens):
            raise ValueError("episode tokens must be at most 64 valid bytes")


@dataclass(frozen=True)
class CEBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor


def rollout_behavioral_episode(
    config: HazardWorldConfig, *, split_id: str, split_seed: int, episode_index: int
) -> BehavioralEpisode:
    """Roll out the causal behavior policy and retain every real frame."""
    validate_rtg_config(config)
    episode_id = episode_identifier(split_id, config.world_id, episode_index)
    world = HazardWorld(config)
    tokens: list[int] = []
    for t in range(config.max_steps):
        if world.state.terminal:
            break
        observation = world.observe()
        action = choose_behavior_action(observation, split_seed, episode_id, t)
        tokens.extend(encode_candidate(observation, action, grid_size=config.grid_size).frame)
        if world.step(action).done:
            break
    return BehavioralEpisode(episode_id, tuple(tokens))


def make_behavioral_episodes(
    worlds: Sequence[HazardWorldConfig], *, episodes_per_world: int, split_id: str, split_seed: int
) -> tuple[BehavioralEpisode, ...]:
    if episodes_per_world < 1 or len({world.world_id for world in worlds}) != len(worlds):
        raise ValueError("world ids must be unique and episodes_per_world positive")
    episodes = [
        rollout_behavioral_episode(world, split_id=split_id, split_seed=split_seed, episode_index=index)
        for world in worlds for index in range(episodes_per_world)
    ]
    return tuple(sorted(episodes))


def collate_ce_episodes(
    episodes: Sequence[BehavioralEpisode], *, sequence_length: int = 64
) -> CEBatch:
    """Pad inputs but mark padding and the final real byte as non-targets."""
    if not episodes or sequence_length < 2:
        raise ValueError("CE collation needs episodes and sequence_length >= 2")
    input_ids = torch.zeros((len(episodes), sequence_length), dtype=torch.long)
    targets = torch.full_like(input_ids, -100)
    for row, episode in enumerate(episodes):
        if len(episode.tokens) > sequence_length:
            raise ValueError("complete episode exceeds requested sequence length")
        values = torch.tensor(episode.tokens, dtype=torch.long)
        input_ids[row, : len(values)] = values
        targets[row, : len(values) - 1] = values[1:]
    return CEBatch(input_ids, targets)


def backbone_batch_indices(
    item_count: int, training_seed: int, *, updates: int = 1_000, batch_size: int = 4
) -> tuple[tuple[int, ...], ...]:
    """Permute once with PCG64(40000+seed), then cycle exactly."""
    if item_count < 1 or updates < 1 or batch_size < 1:
        raise ValueError("batch dimensions must be positive")
    return make_batch_plan(
        item_count,
        training_seed,
        "backbone",
        batch_size=batch_size,
        updates=updates,
    ).batches
