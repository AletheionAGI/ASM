"""Hazard-conditioned dynamics metrics for sealed ATTR records."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .p2_evaluation import EpisodePrediction


def next_state_nll_by_hazard(
    records: Iterable[EpisodePrediction], *, horizon: int = 8
) -> dict[str, dict[str, float | int]]:
    """Split next-state NLL by whether a future hazard occurs within H."""
    groups: dict[int, list[float]] = {0: [], 1: []}
    seen = False
    for record in records:
        if horizon not in record.horizons:
            raise ValueError(f"horizon {horizon} is absent from prediction record")
        column = record.horizons.index(horizon)
        if len(record.hazard_labels) != len(record.next_state_nll):
            raise ValueError("hazard labels and next-state NLL must align")
        for labels, nll in zip(record.hazard_labels, record.next_state_nll):
            value = float(nll)
            if not math.isfinite(value):
                raise ValueError("next-state NLL must be finite")
            groups[int(labels[column])].append(value)
            seen = True
    if not seen or not groups[0] or not groups[1]:
        raise ValueError("both H-negative and H-positive records are required")
    return {
        "negative": {
            "mean": math.fsum(groups[0]) / len(groups[0]),
            "n": len(groups[0]),
        },
        "positive": {
            "mean": math.fsum(groups[1]) / len(groups[1]),
            "n": len(groups[1]),
        },
    }


__all__ = ["next_state_nll_by_hazard"]
