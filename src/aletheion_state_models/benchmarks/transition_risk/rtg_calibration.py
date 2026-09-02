"""Disjoint deterministic temperature and residual calibration for ATTR-RTG."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

TEMPERATURE_GRID_SIZE = 1601
ECE_BINS = 15


@dataclass(frozen=True)
class RtgCalibration:
    """Frozen scalar temperature and empirical absolute-residual band."""

    temperature: float
    q95: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.q95) or not 0 <= self.q95 <= 1:
            raise ValueError("q95 must be finite and in [0, 1]")

    def probability(self, logit: float) -> float:
        """Apply scalar temperature scaling to one raw or smoothed logit."""
        if not math.isfinite(logit):
            raise ValueError("calibration logit must be finite")
        scaled = logit / self.temperature
        if scaled >= 0:
            return 1.0 / (1.0 + math.exp(-scaled))
        exponential = math.exp(scaled)
        return exponential / (1.0 + exponential)


def partition_calibration_worlds(world_ids: Sequence[str | int]) -> tuple[tuple[str | int, ...], tuple[str | int, ...]]:
    """Split the 16 sorted calibration worlds into fixed temperature/residual halves."""
    if not world_ids or type(world_ids[0]) not in {str, int}:
        raise ValueError("world IDs must be homogeneous strings or integers")
    identifier_type = type(world_ids[0])
    if any(type(item) is not identifier_type for item in world_ids):
        raise ValueError("world IDs must be homogeneous strings or integers")
    unique = sorted(set(world_ids))
    if len(unique) != 16 or len(unique) != len(world_ids):
        raise ValueError("calibration requires exactly 16 distinct world IDs")
    return tuple(unique[:8]), tuple(unique[8:])


def _validated(logits_or_probabilities: Sequence[float], labels: Sequence[int], origin_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(logits_or_probabilities, dtype=torch.float64, device="cpu")
    targets = torch.as_tensor(labels, dtype=torch.float64, device="cpu")
    if values.ndim != 1 or targets.ndim != 1 or values.numel() != targets.numel():
        raise ValueError("scores and labels must be aligned one-dimensional sequences")
    if values.numel() < 1 or not torch.isfinite(values).all():
        raise ValueError("calibration scores must be finite and non-empty")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("calibration labels must be binary")
    positives = int(targets.sum().item())
    negatives = targets.numel() - positives
    if type(origin_count) is not int or origin_count < 50:
        raise ValueError("each calibration half requires at least 50 origins")
    if positives < 15 or negatives < 15:
        raise ValueError("each calibration half requires at least 15 labels per class")
    return values, targets


def fit_temperature(logits: Sequence[float], labels: Sequence[int], *, origin_count: int) -> float:
    """Select T on the frozen 1601-point log grid; ties select the smaller T."""
    scores, targets = _validated(logits, labels, origin_count)
    log_temperatures = torch.linspace(-4.0, 4.0, TEMPERATURE_GRID_SIZE, dtype=torch.float64)
    temperatures = torch.exp(log_temperatures)
    signed = targets.mul(2.0).sub(1.0)
    losses = torch.nn.functional.softplus(
        -signed.unsqueeze(0) * scores.unsqueeze(0) / temperatures.unsqueeze(1)
    ).mean(dim=1)
    return float(temperatures[int(torch.argmin(losses))].item())


def calibrated_probabilities(logits: Sequence[float], temperature: float) -> tuple[float, ...]:
    """Apply a frozen positive temperature in CPU/float64."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scores = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("logits must be a finite one-dimensional sequence")
    return tuple(float(value) for value in torch.sigmoid(scores / temperature))


def empirical_q95(probabilities: Sequence[float], labels: Sequence[int], *, origin_count: int) -> float:
    """Return k=min(n,ceil((n+1)*.95)) order statistic of absolute residuals."""
    values, targets = _validated(probabilities, labels, origin_count)
    if not torch.all((0.0 <= values) & (values <= 1.0)):
        raise ValueError("calibrated probabilities must lie in [0, 1]")
    residuals = torch.sort(torch.abs(values - targets)).values
    n = residuals.numel()
    rank = min(n, math.ceil((n + 1) * 0.95))
    return float(residuals[rank - 1].item())


def fit_disjoint_calibration(
    temperature_logits: Sequence[float],
    temperature_labels: Sequence[int],
    temperature_world_ids: Sequence[str | int],
    residual_logits: Sequence[float],
    residual_labels: Sequence[int],
    residual_world_ids: Sequence[str | int],
    *,
    temperature_origin_count: int,
    residual_origin_count: int,
) -> RtgCalibration:
    """Fit T and q95 while enforcing the canonical disjoint eight-world halves."""
    if len(temperature_world_ids) != len(temperature_logits):
        raise ValueError("temperature world IDs must align with temperature logits")
    if len(residual_world_ids) != len(residual_logits):
        raise ValueError("residual world IDs must align with residual logits")
    temperature_worlds = set(temperature_world_ids)
    residual_worlds = set(residual_world_ids)
    if len(temperature_worlds) != 8 or len(residual_worlds) != 8:
        raise ValueError("calibration requires eight worlds in each half")
    if not temperature_worlds.isdisjoint(residual_worlds):
        raise ValueError("temperature and residual worlds must be disjoint")
    expected_temperature, expected_residual = partition_calibration_worlds(
        tuple(temperature_worlds | residual_worlds)
    )
    if temperature_worlds != set(expected_temperature) or residual_worlds != set(expected_residual):
        raise ValueError("worlds do not match the canonical sorted calibration halves")
    temperature = fit_temperature(
        temperature_logits, temperature_labels, origin_count=temperature_origin_count
    )
    residual_probabilities = calibrated_probabilities(residual_logits, temperature)
    q95 = empirical_q95(
        residual_probabilities, residual_labels, origin_count=residual_origin_count
    )
    return RtgCalibration(temperature=temperature, q95=q95)


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Compute count-weighted ECE in the frozen 15 equal-width bins."""
    values = torch.as_tensor(probabilities, dtype=torch.float64, device="cpu")
    targets = torch.as_tensor(labels, dtype=torch.float64, device="cpu")
    if values.ndim != 1 or values.numel() == 0 or values.shape != targets.shape:
        raise ValueError("probabilities and labels must be aligned non-empty vectors")
    if not torch.isfinite(values).all() or not torch.all((0 <= values) & (values <= 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("labels must be binary")
    indices = torch.clamp((values * ECE_BINS).floor().to(torch.int64), max=ECE_BINS - 1)
    error = 0.0
    for index in range(ECE_BINS):
        mask = indices == index
        count = int(mask.sum())
        if count:
            error += count / values.numel() * abs(
                float(values[mask].mean()) - float(targets[mask].mean())
            )
    return error
