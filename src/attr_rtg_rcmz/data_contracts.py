"""Immutable data contracts with explicit input/truth custody boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import CANDIDATES, EPISODES_PER_WORLD, MAX_EPISODE_LENGTH


@dataclass(frozen=True, order=True)
class OriginKey:
    """Audit identity. This value must remain outside the model message."""

    split: str
    world: str
    episode: int
    origin: int

    def __post_init__(self) -> None:
        if not self.split or not self.world:
            raise ValueError("split and world must be non-empty")
        if type(self.episode) is not int or not 0 <= self.episode < EPISODES_PER_WORLD:
            raise ValueError("episode must be in [0, 4)")
        if type(self.origin) is not int or not 0 <= self.origin < MAX_EPISODE_LENGTH:
            raise ValueError("origin must be in [0, 64)")


@dataclass(frozen=True)
class ModelProcessInput:
    """The exact four-field payload permitted to enter a model process."""

    history_bytes: tuple[tuple[int, ...], ...]
    candidate4s: tuple[tuple[tuple[int, int, int, int], ...], ...]
    masks: tuple[tuple[bool, ...], ...]
    logical_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        batch = len(self.history_bytes)
        if not batch or any(
            len(field) != batch
            for field in (self.candidate4s, self.masks, self.logical_lengths)
        ):
            raise ValueError(
                "four model fields must have one equal non-empty batch axis"
            )
        for history, candidates, masks, length in zip(
            self.history_bytes,
            self.candidate4s,
            self.masks,
            self.logical_lengths,
            strict=True,
        ):
            if any(
                type(value) is not int or not 0 <= value <= 255 for value in history
            ):
                raise ValueError("history_bytes must contain bytes")
            if type(length) is not int or not 0 <= length <= len(history):
                raise ValueError("logical length is outside its history storage")
            if len(candidates) != len(CANDIDATES) or len(masks) != len(CANDIDATES):
                raise ValueError("candidate and mask axes must have length six")
            if any(
                len(frame) != 4
                or any(type(v) is not int or not 0 <= v <= 255 for v in frame)
                for frame in candidates
            ):
                raise ValueError("candidate4s must contain four-byte frames")
            if any(type(value) is not bool for value in masks):
                raise ValueError("masks must be boolean")

    def as_dict(self) -> dict[str, Any]:
        """Return only the exact allowed process fields."""
        return {
            "history_bytes": self.history_bytes,
            "candidate4s": self.candidate4s,
            "masks": self.masks,
            "logical_lengths": self.logical_lengths,
        }


@dataclass(frozen=True)
class H8Truth:
    candidate: str
    unsafe: bool | None
    valid: bool
    transitions: int
    terminal: bool
    error: str | None = None

    def __post_init__(self) -> None:
        if self.candidate not in CANDIDATES:
            raise ValueError("unregistered candidate")
        if self.valid != (self.unsafe is not None and self.error is None):
            raise ValueError("valid truth must contain one boolean and no error")
        if not 0 <= self.transitions <= 8:
            raise ValueError("transition count must be in [0, 8]")
