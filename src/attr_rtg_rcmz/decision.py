"""Fail-closed BLOCK/ABSTAIN decision rule."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import ACTIONS


@dataclass(frozen=True)
class Decision:
    outcome: str
    selected: str | None
    executed: str | None
    coverage: int | None
    invalid: bool


def decide(
    probabilities, mask, tau: float, *, schema_valid=True, brake_valid=True
) -> Decision:
    try:
        p = tuple(float(v) for v in probabilities)
        allowed = tuple(bool(v) for v in mask)
    except (TypeError, ValueError):
        return Decision("BLOCK", None, None, None, True)
    malformed = (
        not schema_valid
        or not brake_valid
        or len(p) != 6
        or len(allowed) != 6
        or not math.isfinite(float(tau))
        or not 0 <= tau <= 1
        or not any(allowed)
        or not allowed[4]
        or any(
            allowed[i] and (not math.isfinite(p[i]) or not 0 <= p[i] <= 1)
            for i in range(6)
        )
    )
    if malformed:
        return Decision("BLOCK", None, None, None, True)
    index = min((i for i in range(6) if allowed[i]), key=lambda i: (p[i], i))
    selected = ACTIONS[index]
    if p[index] <= tau:
        return Decision("EXECUTE", selected, selected, 1, False)
    return Decision("ABSTAIN", selected, "BRAKE", 0, False)
