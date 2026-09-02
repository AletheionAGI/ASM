"""Registered HazardWorld origin preparation with strict truth custody."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .constants import CANDIDATES, SPLIT_ROWS

_WORLD_ACTIONS = (*CANDIDATES, "STOP")


@dataclass(frozen=True)
class Origin:
    split: str
    world_index: int
    episode: int
    origin: int
    history: bytes
    candidate4s: tuple[tuple[int, int, int, int], ...]
    world: Any

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return self.split, self.world_index, self.episode, self.origin


@dataclass
class TruthCache:
    """Privileged cache. It is never consulted by input materialization."""

    _rows: dict[tuple[str, int, int, int], tuple[bool, ...]] = field(
        default_factory=dict
    )

    def get(self, origin: Origin) -> tuple[bool, ...] | None:
        return self._rows.get(origin.identity)

    def put(self, origin: Origin, truth: tuple[bool, ...]) -> None:
        if len(truth) != len(CANDIDATES):
            raise ValueError("truth must have exactly six candidates")
        self._rows[origin.identity] = truth


def generate_registered_origins(
    *, miniature: bool = False, lock: dict[str, object] | None = None
) -> dict[str, tuple[Origin, ...]]:
    """Generate smoke data, or official data only behind an anchored lock."""
    if not miniature:
        from .lock_guard import verify_runtime_lock

        verify_runtime_lock(lock)
    from world_model.hazard_world import HazardWorld
    from world_model.hazard_world_types import HazardWorldConfig

    result: dict[str, tuple[Origin, ...]] = {}
    for split, registered_count, regime in SPLIT_ROWS:
        world_count = 1 if miniature else registered_count
        episodes = 1 if miniature else 4
        max_steps = 3 if miniature else 64
        origins: list[Origin] = []
        for world_index in range(world_count):
            config = _world_config(
                HazardWorldConfig, split, regime, world_index, max_steps
            )
            for episode in range(episodes):
                world = HazardWorld(config)
                history = bytearray()
                for origin_index in range(max_steps):
                    if world.state.terminal:
                        break
                    observation = world.observe()
                    frames = tuple(
                        _candidate_frame(observation, action, config.grid_size)
                        for action in CANDIDATES
                    )
                    # Origins start after one complete prior four-byte behavior frame.
                    if origin_index >= 1:
                        origins.append(
                            Origin(
                                split,
                                world_index,
                                episode,
                                origin_index,
                                bytes(history[-256:]),
                                frames,
                                world.clone(),
                            )
                        )
                    action = CANDIDATES[
                        (world_index + episode + origin_index) % len(CANDIDATES)
                    ]
                    history.extend(
                        _candidate_frame(observation, action, config.grid_size)
                    )
                    world.step(action)
        result[split] = tuple(origins)
    return result


def materialize(
    origins: tuple[Origin, ...], indices: list[int], device: str
) -> dict[str, Any]:
    """Create the exact four-field causal message; truth is unreachable here."""
    import torch

    selected = [origins[index] for index in indices]
    histories = torch.zeros((len(selected), 256), dtype=torch.uint8)
    lengths = torch.empty(len(selected), dtype=torch.int64)
    for row, origin in enumerate(selected):
        payload = origin.history[-256:]
        if not payload or len(payload) % 4:
            raise ValueError(
                "origin history must contain complete prior four-byte frames"
            )
        histories[row, : len(payload)] = torch.tensor(tuple(payload), dtype=torch.uint8)
        lengths[row] = len(payload)
    frames = torch.tensor(
        [origin.candidate4s for origin in selected], dtype=torch.float32
    )
    masks = torch.ones((len(selected), len(CANDIDATES)), dtype=torch.bool)
    return {
        "message": {
            "history_bytes": histories.to(device),
            "candidate4s": frames.to(device),
            "masks": masks.to(device),
            "logical_lengths": lengths.to(device),
        },
        "origins": selected,
    }


def truths_after_forward(
    origins: list[Origin], device: str, cache: TruthCache | None = None
) -> Any:
    """Attach H8 only after model return; reuse privileged truth outside its message."""
    import torch

    from .h8 import h8_all_candidates

    truth_cache = cache if cache is not None else TruthCache()
    rows = []
    for origin in origins:
        cached = truth_cache.get(origin)
        if cached is None:
            truth = h8_all_candidates(origin.world)
            if not all(item.valid for item in truth):
                raise RuntimeError("invalid H8 truth")
            cached = tuple(bool(item.unsafe) for item in truth)
            truth_cache.put(origin, cached)
        rows.append(cached)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def canonical_order(origins: tuple[Origin, ...]) -> tuple[int, ...]:
    return tuple(range(len(origins)))


def training_order(origins: tuple[Origin, ...], seed: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(origins)),
            key=lambda index: hashlib.sha256(
                f"batch:{seed}:{origins[index].world_index}:{origins[index].episode}:{origins[index].origin}".encode()
            ).digest(),
        )
    )


def _candidate_frame(
    observation: Any, action: str, grid_size: int
) -> tuple[int, int, int, int]:
    position = observation.agent[0] * grid_size + observation.agent[1]
    velocity = (observation.velocity[0] + 1) * 3 + observation.velocity[1] + 1
    velocity_action = velocity * len(_WORLD_ACTIONS) + _WORLD_ACTIONS.index(action)
    energy_bin = min(31, max(0, int(observation.energy_sensor * 32)))
    cells = observation.local_traps + observation.local_hazards
    nearest = (
        7
        if not cells
        else min(
            abs(observation.agent[0] - row) + abs(observation.agent[1] - column)
            for row, column in cells
        )
    )
    energy_hazard = energy_bin * 8 + min(7, nearest)
    dr = min(7, max(-7, observation.goal_delta[0]))
    dc = min(7, max(-7, observation.goal_delta[1]))
    frame = (position, velocity_action, energy_hazard, (dr + 7) * 15 + dc + 7)
    if any(not 0 <= value < 256 for value in frame):
        raise ValueError("candidate frame exceeds byte vocabulary")
    return frame


def _world_config(cls: Any, split: str, regime: str, index: int, max_steps: int) -> Any:
    seed_material = hashlib.sha256(
        f"ATTR-RTG-RCMZ-V1:{split}:{index}".encode()
    ).digest()
    seed = int.from_bytes(seed_material[:8], "big")
    shift = regime != "baseline"
    ood = regime == "OOD"
    return cls(
        world_id=f"{split}-{index:04d}",
        seed=seed,
        max_steps=max_steps,
        traps=((2, 2), (6, 6)) if not ood else ((1, 3), (5, 7), (7, 2)),
        moving_hazards=((4, 4),),
        hazard_velocities=((0, 1),),
        sensor_noise=0.08 if shift else 0.04,
        forcing=0.22 if ood else (0.16 if shift else 0.10),
        dynamic_family=regime,
    )
