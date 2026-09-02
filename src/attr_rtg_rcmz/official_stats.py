"""Exact calibration and raw sufficient statistics for one arm/seed."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .constants import CANDIDATES, TEMPERATURE_GRID


def calibrate(records: list[dict[str, Any]]) -> tuple[float, float]:
    """Fit T with candidate→origin→episode→world NLL and pooled safe q95."""
    import torch

    from .quantiles import type7

    eligible = [row for row in records if row["valid"]]
    if not eligible:
        raise RuntimeError("empty calibration set")
    scores = []
    for temperature in TEMPERATURE_GRID:
        origins = []
        for row in eligible:
            p = torch.sigmoid(row["logits"].double() / temperature).clamp(
                2**-24, 1 - 2**-24
            )
            y = row["labels"].double()
            loss = -y * torch.log(p) - (1 - y) * torch.log1p(-p)
            origins.append((row["world"], row["episode"], float(loss.mean())))
        scores.append(_fold_origin_tuples(origins))
    score_tensor = torch.tensor(
        scores, dtype=torch.float64, device=eligible[0]["logits"].device
    )
    if not bool(torch.isfinite(score_tensor).all()):
        raise RuntimeError("nonfinite temperature score")
    temperature = TEMPERATURE_GRID[int(torch.argmin(score_tensor).item())]
    safe = []
    for row in eligible:  # records are already canonical world/episode/origin order
        probabilities = torch.sigmoid(row["logits"].double() / temperature)
        safe.extend(probabilities[row["labels"] == 0].unbind())
    if not safe:
        raise RuntimeError("empty safe calibration vector")
    return float(temperature), float(type7(torch.stack(safe), 0.95).item())


def summarize(
    records: list[dict[str, Any]],
    temperature: float,
    tau: float,
    *,
    arm: str,
    seed: int,
    regime: str,
    peak_bytes: int,
    elapsed: float,
) -> dict[str, Any]:
    """Return display scalars plus episode-level origin-folded sufficient values."""
    import torch

    from .ece import origin_ece15

    grouped: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    for row in records:
        if not row["valid"]:
            raise RuntimeError("invalid test origin")
        probabilities = torch.sigmoid(row["logits"].double() / temperature)
        labels = row["labels"].double()
        clipped = probabilities.clamp(2**-24, 1 - 2**-24)
        losses = -labels * torch.log(clipped) - (1 - labels) * torch.log1p(-clipped)
        ece, valid = origin_ece15(probabilities[None, :], labels[None, :])
        if not bool(valid.item()):
            raise RuntimeError("invalid ECE origin")
        best = int(torch.argmin(probabilities).item())
        abstain = float(probabilities[best].item() > tau)
        executed = CANDIDATES.index("BRAKE") if abstain else best
        unsafe = float(labels[executed].item())
        has_safe = bool((labels == 0).any())
        grouped[(row["world"], row["episode"])].append(
            {
                "h8_nll": float(losses.mean()),
                "ece": float(ece.item()),
                "unsafe_selection": unsafe,
                "safe_service": (1 - unsafe) if has_safe else math.nan,
                "coverage": 1 - abstain,
                "abstention": abstain,
            }
        )
    names = (
        "h8_nll",
        "ece",
        "unsafe_selection",
        "safe_service",
        "coverage",
        "abstention",
    )
    sufficient = {name: _episode_sufficient(grouped, name) for name in names}
    metrics = {name: _fold_episode_sufficient(sufficient[name]) for name in names}
    return {
        "status": "VALID",
        "arm": arm,
        "seed": seed,
        "regime": regime,
        "temperature": temperature,
        "tau": tau,
        **metrics,
        "peak_vram_bytes": peak_bytes,
        "elapsed_seconds": elapsed,
        "_sufficient": sufficient,
    }


def invalid_row(
    *,
    arm: str,
    seed: int,
    regime: str,
    reason: str,
    peak_bytes: int,
    elapsed: float,
    temperature=None,
    tau=None,
    computed_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return a complete fail-closed cell while retaining finite computed metrics."""
    names = (
        "h8_nll",
        "ece",
        "unsafe_selection",
        "safe_service",
        "coverage",
        "abstention",
    )
    available = computed_metrics or {}
    metrics = {
        name: float(available[name])
        if name in available and math.isfinite(float(available[name]))
        else None
        for name in names
    }
    return {
        "status": "INVALID",
        "arm": arm,
        "seed": seed,
        "regime": regime,
        "invalid_reason": reason,
        "temperature": temperature,
        "tau": tau,
        **metrics,
        "peak_vram_bytes": peak_bytes,
        "elapsed_seconds": elapsed,
    }


def _episode_sufficient(groups, name: str) -> list[dict[str, float | int]]:
    output = []
    for (world, episode), rows in sorted(groups.items()):
        values = [row[name] for row in rows if math.isfinite(row[name])]
        if not values:
            raise RuntimeError(f"empty origin fold for {name} at {world}/{episode}")
        output.append(
            {"world": world, "episode": episode, "value": sum(values) / len(values)}
        )
    return output


def _fold_episode_sufficient(rows) -> float:
    worlds: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        worlds[int(row["world"])].append(float(row["value"]))
    if not worlds or any(not values for values in worlds.values()):
        raise RuntimeError("empty episode/world fold")
    return sum(sum(values) / len(values) for values in worlds.values()) / len(worlds)


def _fold_origin_tuples(rows) -> float:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for world, episode, value in rows:
        grouped[(world, episode)].append(value)
    episode_rows = [
        {"world": world, "episode": episode, "value": sum(values) / len(values)}
        for (world, episode), values in sorted(grouped.items())
        if values
    ]
    if len(episode_rows) != len(grouped):
        raise RuntimeError("empty calibration origin fold")
    return _fold_episode_sufficient(episode_rows)
