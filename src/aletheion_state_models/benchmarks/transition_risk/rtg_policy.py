"""Counter-based SHA-256 behavior policy frozen for ATTR-RTG."""

from __future__ import annotations

import hashlib

from world_model.hazard_world_types import HazardObservation

BEHAVIOR_ACTIONS = ("U", "D", "L", "R", "BRAKE")
_POLICY_DOMAIN = "ATTR-RTG-POLICY-V1"


def _digest(split_seed: int, episode_id: str, t: int, channel: str) -> bytes:
    if type(split_seed) is not int or type(t) is not int or t < 0:
        raise ValueError("policy seed and non-negative step must be integers")
    if not episode_id or not channel or "|" in episode_id or "|" in channel:
        raise ValueError("policy key fields must be non-empty and delimiter-free")
    key = f"{_POLICY_DOMAIN}|{split_seed}|{episode_id}|{t}|{channel}"
    return hashlib.sha256(key.encode("utf-8")).digest()


def keyed_uniform(split_seed: int, episode_id: str, t: int, channel: str) -> float:
    """Return a deterministic open-interval uniform from the first 64 digest bits."""
    integer = int.from_bytes(_digest(split_seed, episode_id, t, channel)[:8], "big")
    return (integer + 0.5) / 2**64


def keyed_index(
    split_seed: int, episode_id: str, t: int, channel: str, size: int
) -> int:
    """Return a deterministic categorical index without global RNG state."""
    if type(size) is not int or size < 1:
        raise ValueError("categorical size must be a positive integer")
    integer = int.from_bytes(_digest(split_seed, episode_id, t, channel)[:8], "big")
    return integer % size


def choose_behavior_action(
    observation: HazardObservation,
    split_seed: int,
    episode_id: str,
    t: int,
) -> str:
    """Choose RECOVER when registered, otherwise one of five service actions."""
    if (
        observation.energy_sensor < 0.25
        and keyed_uniform(split_seed, episode_id, t, "recover") < 0.8
    ):
        return "RECOVER"
    index = keyed_index(split_seed, episode_id, t, "recover", len(BEHAVIOR_ACTIONS))
    return BEHAVIOR_ACTIONS[index]
