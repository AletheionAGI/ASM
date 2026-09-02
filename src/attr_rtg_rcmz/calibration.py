"""Frozen temperature-grid calibration."""

from __future__ import annotations

import torch

from .constants import TEMPERATURE_GRID
from .nll import h8_nll
from .validation import binary, fp64


def calibrated_probabilities(logits, temperature: float | torch.Tensor) -> torch.Tensor:
    z = fp64(logits)
    t = float(torch.as_tensor(temperature).item())
    if not t > 0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid(z / t)


def fit_temperature(logits, labels, eligible=None) -> tuple[float, torch.Tensor]:
    z, y = fp64(logits), binary(labels)
    if z.shape != y.shape:
        raise ValueError("logit and label shape mismatch")
    scores = []
    for temperature in TEMPERATURE_GRID:
        scores.append(h8_nll(torch.sigmoid(z / temperature), y, eligible))
    stacked = torch.stack(scores)
    if stacked.numel() != len(TEMPERATURE_GRID):
        raise ValueError("temperature fit must produce one scalar per grid value")
    index = int(
        torch.argmin(stacked).item()
    )  # grid ascending gives smaller-T tie break
    return TEMPERATURE_GRID[index], stacked
