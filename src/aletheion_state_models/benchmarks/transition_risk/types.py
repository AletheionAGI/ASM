"""Shared value types for the ATTR transition-risk benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Tuple

DEFAULT_HORIZONS: Tuple[int, ...] = (1, 4, 8, 16)


@dataclass(frozen=True)
class HorizonLabels:
    """Binary future-entry labels for one observation time."""

    step: int
    labels: Mapping[int, int]
    episode_id: Hashable | None = None

    def __getitem__(self, horizon: int) -> int:
        return self.labels[horizon]


@dataclass(frozen=True)
class HazardPrediction:
    """Calibrated risk information for one candidate action."""

    action: Hashable
    probability: float
    upper_bound: float
    utility: float = 0.0
    is_ood: bool = False


@dataclass(frozen=True)
class ShieldDecision:
    """Auditable output of the common external hard shield."""

    action: Hashable
    accepted_actions: Tuple[Hashable, ...]
    rejected_actions: Tuple[Hashable, ...]
    abstained: bool
    reason: str
