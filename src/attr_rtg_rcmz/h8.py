"""Exact H8 truth on cloned HazardWorld states."""

from __future__ import annotations

from typing import Any, Protocol

from .constants import CANDIDATES, HORIZON
from .data_contracts import H8Truth


class CloneWorld(Protocol):
    state: Any

    def clone(self) -> CloneWorld: ...
    def step(self, action: str) -> Any: ...


def _flag(value: object, name: str) -> bool:
    field = getattr(value, name, None)
    if type(field) is not bool:
        raise ValueError(f"world state has no boolean {name}")
    return field


def _transition_flags(world: CloneWorld, transition: object) -> tuple[bool, bool]:
    state = getattr(transition, "state", None)
    if state is None or getattr(world, "state", None) is not state:
        raise ValueError("transition is missing the installed next state")
    unsafe = _flag(state, "unsafe")
    terminal = _flag(state, "terminal")
    exposed_unsafe = getattr(transition, "unsafe", unsafe)
    exposed_done = getattr(transition, "done", terminal)
    if type(exposed_unsafe) is not bool or exposed_unsafe != unsafe:
        raise ValueError("transition unsafe bit differs from state")
    if type(exposed_done) is not bool or exposed_done != terminal:
        raise ValueError("transition terminal bit differs from state")
    return unsafe, terminal


def h8_truth(world: CloneWorld, candidate: str) -> H8Truth:
    """Execute candidate at t=1 and BRAKE at t=2..8 on one private clone."""
    if candidate not in CANDIDATES:
        raise ValueError("H8 candidate must be one of the fixed six")
    count = 0
    terminal = False
    try:
        if _flag(world.state, "terminal"):
            raise ValueError("origin is already terminal")
        clone = world.clone()
        if clone is world or getattr(clone, "state", None) is None:
            raise ValueError("invalid clone")
        unsafe = False
        for transition_number in range(1, HORIZON + 1):
            action = candidate if transition_number == 1 else "BRAKE"
            item = clone.step(action)
            if item is None:
                raise ValueError("missing transition")
            count += 1
            step_unsafe, terminal = _transition_flags(clone, item)
            unsafe = unsafe or step_unsafe
            if terminal:
                break
        return H8Truth(candidate, unsafe, True, count, terminal)
    except (
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        return H8Truth(
            candidate, None, False, count, terminal, f"{type(error).__name__}: {error}"
        )


def h8_all_candidates(world: CloneWorld) -> tuple[H8Truth, ...]:
    """Compute truth in fixed order; each branch clones the same origin."""
    return tuple(h8_truth(world, candidate) for candidate in CANDIDATES)
