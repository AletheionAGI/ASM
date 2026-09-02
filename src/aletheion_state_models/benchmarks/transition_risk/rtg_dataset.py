"""Pre-export causal input preparation for the frozen ATTR-RTG dataset."""

from __future__ import annotations

from collections.abc import Sequence

from world_model.hazard_world import HazardWorld
from world_model.hazard_world_types import HazardWorldConfig

from .rtg_encoding import encode_all_candidates, encode_candidate
from .rtg_physical_targets import validate_rtg_config
from .rtg_policy import choose_behavior_action
from .rtg_types import (
    OriginInput,
    OriginMetadata,
    PhysicalSnapshot,
    PreparedRtgOrigin,
)


def episode_identifier(split_id: str, world_id: str, episode_index: int) -> str:
    """Build a deterministic split-namespaced episode identifier."""
    if not split_id or not world_id or "|" in split_id or "|" in world_id:
        raise ValueError(
            "split and world identifiers must be non-empty and delimiter-free"
        )
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode index must be a non-negative integer")
    return f"{split_id}:{world_id}:{episode_index}"


def prepare_rtg_episode(
    config: HazardWorldConfig,
    *,
    split_id: str,
    split_seed: int,
    episode_index: int,
) -> tuple[PreparedRtgOrigin, ...]:
    """Prepare t>=1 inputs and snapshots without materializing branch truth."""
    validate_rtg_config(config)
    episode_id = episode_identifier(split_id, config.world_id, episode_index)
    world = HazardWorld(config)
    history: list[int] = []
    origins: list[PreparedRtgOrigin] = []

    for t in range(config.max_steps):
        if world.state.terminal:
            break
        observation = world.observe()
        if t >= 1:
            origins.append(
                PreparedRtgOrigin(
                    metadata=OriginMetadata(
                        split_id=split_id,
                        world_id=config.world_id,
                        episode_id=episode_id,
                        t=t,
                    ),
                    inputs=OriginInput(
                        history=tuple(history),
                        candidates=encode_all_candidates(
                            observation, grid_size=config.grid_size
                        ),
                    ),
                    snapshot=PhysicalSnapshot(config=config, state=world.state),
                )
            )
        action = choose_behavior_action(observation, split_seed, episode_id, t)
        behavior_frame = encode_candidate(
            observation, action, grid_size=config.grid_size
        ).frame
        transition = world.step(action)
        history.extend(behavior_frame)
        if transition.done:
            break
    return tuple(origins)


def prepare_rtg_input_origins(
    worlds: Sequence[HazardWorldConfig],
    *,
    episodes_per_world: int,
    split_id: str,
    split_seed: int,
) -> tuple[PreparedRtgOrigin, ...]:
    """Prepare supplied episodes in lexical order, without labels or branches."""
    if episodes_per_world < 1:
        raise ValueError("episodes_per_world must be positive")
    if len({world.world_id for world in worlds}) != len(worlds):
        raise ValueError("duplicate world_id would violate dataset disjointness")
    origins = [
        origin
        for world in worlds
        for episode_index in range(episodes_per_world)
        for origin in prepare_rtg_episode(
            world,
            split_id=split_id,
            split_seed=split_seed,
            episode_index=episode_index,
        )
    ]
    origins.sort(
        key=lambda item: (
            item.metadata.world_id,
            item.metadata.episode_id,
            item.metadata.t,
        )
    )
    return tuple(origins)
