"""Deterministic linear probe for information retained in a representation.

The probe is deliberately small and diagnostic-only.  It fits a ridge linear
map on a deterministic train split, then compares its test MSE with a predictor
that always emits the training-target mean.  In particular, the same target can
be probed from an effective state and from declared external memory to detect
an unintended information bypass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class InformationProbeResult:
    """Held-out metrics produced by :func:`linear_information_probe`.

    Attributes:
        mse: Mean squared error of the fitted ridge probe on the test split.
        baseline_mse: Test MSE obtained by predicting the training-target mean.
        recovery_score: Fraction of baseline error removed by the probe, in
            ``[0, 1]``.  A value near zero means that the representation does
            not linearly recover the target; a value near one means that it does.
    """

    mse: float
    baseline_mse: float
    recovery_score: float


def _as_sample_matrix(value: Tensor, *, name: str) -> Tensor:
    """Return ``value`` as a finite, two-dimensional float64 CPU tensor."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 0:
        raise ValueError(f"{name} must have a sample dimension")
    if value.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")

    matrix = value.detach().to(device="cpu", dtype=torch.float64)
    matrix = matrix.reshape(matrix.shape[0], -1)
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature per sample")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _split_indices(
    sample_count: int,
    test_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Build deterministic, disjoint train and test indices."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")

    test_count = round(sample_count * test_fraction)
    test_count = min(max(test_count, 1), sample_count - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(sample_count, generator=generator)
    return permutation[test_count:], permutation[:test_count]


def _fit_centered_ridge(features: Tensor, targets: Tensor, ridge: float) -> tuple[Tensor, Tensor]:
    """Fit ridge weights and an unregularized intercept."""
    feature_mean = features.mean(dim=0, keepdim=True)
    target_mean = targets.mean(dim=0, keepdim=True)
    centered_features = features - feature_mean
    centered_targets = targets - target_mean

    feature_count = features.shape[1]
    gram = centered_features.T @ centered_features
    regularizer = torch.eye(feature_count, dtype=features.dtype) * ridge
    weights = torch.linalg.solve(gram + regularizer, centered_features.T @ centered_targets)
    intercept = target_mean - feature_mean @ weights
    return weights, intercept


def linear_information_probe(
    representation: Tensor,
    target: Tensor,
    *,
    test_fraction: float = 0.25,
    ridge: float = 1e-6,
    seed: int = 0,
) -> InformationProbeResult:
    """Measure how well a representation linearly recovers a target.

    The split is shuffled with a local CPU generator, so the result is
    repeatable and does not mutate PyTorch's global random state.  Computation
    uses CPU float64 to make this diagnostic stable across input dtypes and
    accelerator availability.

    Args:
        representation: Tensor shaped ``[samples, ...]``.  Non-sample axes are
            flattened into probe features.
        target: Tensor shaped ``[samples, ...]``.  Non-sample axes are flattened
            into regression outputs.
        test_fraction: Fraction assigned to the held-out test split.
        ridge: Strictly positive L2 coefficient for probe weights.  The
            intercept is not regularized.
        seed: Seed used only for the deterministic split permutation.

    Returns:
        Held-out probe MSE, mean-baseline MSE, and normalized recovery score.

    Raises:
        TypeError: If either input is not a tensor or ``seed`` is not an integer.
        ValueError: If shapes, values, or hyperparameters are invalid.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(ridge, (int, float)) or isinstance(ridge, bool) or ridge <= 0:
        raise ValueError("ridge must be a strictly positive number")

    features = _as_sample_matrix(representation, name="representation")
    targets = _as_sample_matrix(target, name="target")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("representation and target must have the same sample count")

    train_indices, test_indices = _split_indices(features.shape[0], test_fraction, seed)
    train_features = features[train_indices]
    train_targets = targets[train_indices]
    test_features = features[test_indices]
    test_targets = targets[test_indices]

    weights, intercept = _fit_centered_ridge(train_features, train_targets, float(ridge))
    predictions = test_features @ weights + intercept
    mse = torch.mean(torch.square(predictions - test_targets)).item()

    baseline = train_targets.mean(dim=0, keepdim=True)
    baseline_mse = torch.mean(torch.square(test_targets - baseline)).item()
    if baseline_mse == 0.0:
        recovery_score = 0.0
    else:
        recovery_score = max(0.0, min(1.0, 1.0 - mse / baseline_mse))

    return InformationProbeResult(
        mse=mse,
        baseline_mse=baseline_mse,
        recovery_score=recovery_score,
    )


def probe_information(
    representation: Tensor,
    target: Tensor,
    **kwargs: float | int,
) -> InformationProbeResult:
    """Compatibility alias for :func:`linear_information_probe`."""
    return linear_information_probe(representation, target, **kwargs)
